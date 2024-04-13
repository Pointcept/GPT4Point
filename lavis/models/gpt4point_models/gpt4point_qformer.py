"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.gpt4point_models.gpt4point import (
    GPT4Point_Base,
    disabled_train,
)
from lavis.models.gpt4point_models.gpt4point_outputs import GPT4Point_Output, GPT4Point_OutputFeatures

@registry.register_model("gpt4point")
@registry.register_model("gpt4point_feature_extractor")
class GPT4Point_Qformer(GPT4Point_Base):
    """
    GPT4Point first-stage model with Q-former and Point Encoder.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/gpt4point/gpt4point_pretrain.yaml",
    }

    def __init__(
        self,
        point_model="ulip_point_bert",
        point_encoder_cfg=None,
        freeze_point_encoder=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        ckpt_special_strs=None,
    ):
        super().__init__()

        self.ckpt_special_strs = ckpt_special_strs
        '''Point Qformer'''
        self.Qformer, self.query_tokens = self.init_Qformer(                    
            num_query_token, 1408, cross_attention_freq         # self.point_encoder.num_features: 1408
        )
        '''Bert tokenizer'''
        self.tokenizer = self.init_tokenizer() # Bert tokenizer
        '''Point: PIT'''
        self.point_encoder = GPT4Point_Base.init_point_encoder(point_model, point_encoder_cfg)
        self.pc_projection = nn.Parameter(torch.empty(point_encoder_cfg['trans_dim'], 1408))
        nn.init.normal_(self.pc_projection, std=1408 ** -0.5)
        '''Point Qformer'''
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])
        '''Text and others'''
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)    # not freeze
        self.point_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)   # not freeze

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)             # actually it is ptm head
        self.temp = nn.Parameter(0.07 * torch.ones([]))                           # not freeze
        self.max_txt_len = max_txt_len
        
        if freeze_point_encoder:
            for name, param in self.point_encoder.named_parameters():             # PIT frozen 1
                param.requires_grad = False
            logging.info("freeze point encoder")

    def forward(self, samples):
        '''Input'''
        text = samples["text_input"]                                                        # list: bs
        point = samples['point']                                                            # [bs, 8192, 3]

        '''Point Encoder and Point Q-Former'''
        point_embeds = self.point_encoder(point)                                            # [bs, 8192, 3] -> [bs, 512, trans_dim]
        point_embeds = point_embeds @ self.pc_projection                                    # [bs, 512, trans_dim] -> [bs, 512, 1408]        
        
        point_atts = torch.ones(point_embeds.size()[:-1], dtype=torch.long).to(             # [bs, 257]
            point.device
        )
        query_tokens = self.query_tokens.expand(point_embeds.shape[0], -1, -1)              # [1, 32, 768] -> [bs, 32, 768]
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,              # [bs, 32, 768]
            encoder_hidden_states=point_embeds,     # [bs, 257, ***1408***]
            encoder_attention_mask=point_atts,      # [bs, 257]
            use_cache=True,
            return_dict=True,
        )
        '''point feature'''
        point_feats = F.normalize(
            self.point_proj(query_output.last_hidden_state), dim=-1
        )

        '''Text Bert'''
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(point.device)
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(                                                             # [bs, 256]
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )
        
        ###============== Point-text Contrastive ===================###
        point_feats_all = concat_all_gather(                                                # [bs, 32, 256] -> [bs, 32, 256]
            point_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]     # [bs, 256] -> [bs, 256]

        sim_q2t = torch.matmul(                                                             # [bs, bs, 32]
            point_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # point-text similarity: aggregate across all query tokens
        sim_p2t, _ = sim_q2t.max(-1)
        sim_p2t = sim_p2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), point_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-point similarity: aggregate across all query tokens
        sim_t2p, _ = sim_t2q.max(-1)
        sim_t2p = sim_t2p / self.temp  # [batch_size, batch_size*num_gpu]

        rank = dist.get_rank()
        bs = point.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            point.device
        )

        loss_ptc = (
            F.cross_entropy(sim_p2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2p, targets, label_smoothing=0.1)
        ) / 2

        ###============== Point-text Matching ===================###
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)                 # [bs, 32]
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)       # [bs, 32]
        point_embeds_world = all_gather_with_grad(point_embeds)                         # [bs, 257, 1408]
        with torch.no_grad():  
            sim_t2p[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
            sim_p2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)            
                
            weights_t2p = F.softmax(sim_t2p, dim=1)
            weights_p2t = F.softmax(sim_p2t, dim=1)

        # select a negative point for each text
        point_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2p[b], 1).item()
            point_embeds_neg.append(point_embeds_world[neg_idx])
        point_embeds_neg = torch.stack(point_embeds_neg, dim=0)

        # select a negative text for each point
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_p2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0             # [bs, 32], [bs, 32], [bs, 32] = [3bs, 32]
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],        # [3*bs, 32]
            dim=0,
        )

        query_tokens_ptm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)          # [3*bs, 32, 768]
        query_atts_ptm = torch.ones(query_tokens_ptm.size()[:-1], dtype=torch.long).to(     # [3*bs, 32]
            point.device
        )
        attention_mask_all = torch.cat([query_atts_ptm, text_atts_all], dim=1)              # [3*bs, 32*2]

        point_embeds_all = torch.cat(                                                       # [3*bs, 257, 1408]
            [point_embeds, point_embeds_neg, point_embeds], dim=0
        )  # pos, neg, pos
        point_atts_all = torch.ones(point_embeds_all.size()[:-1], dtype=torch.long).to(     # [3*bs, 257]
            point.device
        )

        output_ptm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_ptm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=point_embeds_all,
            encoder_attention_mask=point_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_ptm.last_hidden_state[:, : query_tokens_ptm.size(1), :]      # [3*bs, 32, 768]
        vl_output = self.itm_head(vl_embeddings)                                            # [3*bs, 32, 768] -> [3*bs, 32, 2]
        logits = vl_output.mean(dim=1)                                                      # [3*bs, 32, 2] -> [300, 2]s

        ptm_labels = torch.cat(                                                             # [3*bs] bs:1, 2*bs
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(point.device)
        loss_ptm = F.cross_entropy(logits, ptm_labels)

        ##================= Point Captioning ========================##
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            point.device
        )
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss

        return GPT4Point_Output(
            loss= loss_ptc + loss_ptm + loss_lm,
            loss_ptc=loss_ptc,
            loss_ptm=loss_ptm,
            loss_lm=loss_lm,
        )

    @classmethod
    def from_config(cls, cfg):
        point_model = cfg.get("point_model", "ulip_point_bert")
        point_encoder_cfg = cfg.get("point_encoder_cfg")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)
        freeze_point_encoder = cfg.get("freeze_point_encoder", True)
        max_txt_len = cfg.get("max_txt_len", 32)
        ckpt_special_strs = cfg.get("ckpt_special_strs", None)

        model = cls(
            point_model=point_model,
            point_encoder_cfg=point_encoder_cfg,
            freeze_point_encoder=freeze_point_encoder,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
            ckpt_special_strs=ckpt_special_strs
        )
        model.load_checkpoint_from_config(cfg)

        return model

