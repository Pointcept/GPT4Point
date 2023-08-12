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
from lavis.models.point_blip_models.point_blip import (
    PointBlipBase,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures
from models_point.pointbert.point_encoder import PointTransformer

import yaml
from easydict import EasyDict
def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config

def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
    merge_new_config(config=config, new_config=new_config)
    return config

@registry.register_model("point_blip")
@registry.register_model("point_blip_feature_extractor")
class PointBlipQformer(PointBlipBase):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/point_blip/point_blip_pretrain_opt2.7b.yaml",
    }

    def __init__(
        self,
        point_model="ulip_point_bert",
        point_num=8192,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_point_encoder=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
    ):
        super().__init__()

        '''Vision: VIT and QFormer'''
        # self.visual_encoder, self.ln_vision = self.init_vision_encoder(             # freeze: 2
        #     vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        # )

        # Q-Former vision branch
        self.Qformer, self.query_tokens = self.init_Qformer(                        # freeze: 3, 4
            num_query_token, 1408, cross_attention_freq         # self.visual_encoder.num_features: 1408
        )

        '''Bert tokenizer'''
        self.tokenizer = self.init_tokenizer()      # Bert tokenizer


        '''Point: PIT'''
        self.point_encoder = PointBlipBase.init_point_encoder(point_model, point_num)                       # freeze: 1
        self.pc_projection = nn.Parameter(torch.empty(384, 1408))      # cannot be frozen cause it is defined by ourselves.
        nn.init.normal_(self.pc_projection, std=1408 ** -0.5)

        '''Point Qformer'''
        # Q-Former vision branch
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        '''Text and others'''
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)    # no freeze
        # self.point_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)   # no freeze
        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)  # no freeze

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)             # no freeze
        self.temp = nn.Parameter(0.07 * torch.ones([]))                           # no freeze
        self.max_txt_len = max_txt_len
        
        
        if freeze_point_encoder:
            # pass
            for name, param in self.point_encoder.named_parameters():           # PIT frozen 1
                param.requires_grad = False
            self.point_encoder = self.point_encoder.eval()
            self.point_encoder.train = disabled_train
            logging.info("freeze point encoder")

            # for name, param in self.visual_encoder.named_parameters():          # VIT frozen 2
            #     param.requires_grad = False
            # self.visual_encoder = self.visual_encoder.eval()
            # self.visual_encoder.train = disabled_train
            # logging.info("freeze vision encoder")

            # for name, param in self.Qformer.named_parameters():               # self.Qformer frozen 3
            #     param.requires_grad = False
            # self.Qformer = self.Qformer.eval()
            # self.Qformer.train = disabled_train
            # logging.info("freeze self.Qformer")                                    
            # self.query_tokens.requires_grad = False                             # self.query_tokens frozen 4
            # logging.info("freeze self.query_tokens")


    def forward(self, samples):
        '''Input'''
        # image = samples["image"]                                                            # [bs, 3, 224, 224]
        text = samples["text_input"]                                                        # list: bs
        image = samples['point']                                                            # [bs, 8192, 3]

        '''PIT and Point Q-Former'''
        '''We do not use the Point_Q-Former, cause we make the Q-Former be the point branch'''
        # point_embeds = self.point_encoder(point)                                            # [bs, 8192, 3] -> [bs, 512, 384]
        # point_embeds = point_embeds @ self.pc_projection                                    # [bs, 512, 384] -> [bs, 512, 768]
        # point_atts = torch.ones(point_embeds.size()[:-1], dtype=torch.long).to(             # [bs, 512]
        #     point.device
        # )
        # query_tokens_point = self.query_tokens_point.expand(point_embeds.shape[0], -1, -1)              # [1, 32, 768] -> [bs, 32, 768]
        # query_output_point = self.Qformer_point.bert(
        #     query_embeds=query_tokens_point,                # [bs, 32, 768]
        #     encoder_hidden_states=point_embeds,             # [bs, 512, ***768***]
        #     encoder_attention_mask=point_atts,              # [bs, 512]
        #     use_cache=True, 
        #     return_dict=True,
        # )

        # point_feats = F.normalize(                          # [bs, 32, 256]
        #     self.point_proj(query_output_point.last_hidden_state), dim=-1
        # )

        '''VIT and Vision Q-Former'''
        # image_embeds = self.ln_vision(self.visual_encoder(image))                           # [bs, 3, 224, 224] -> [bs, 257, 1408]

        # make the point as the image
        image_embeds = self.point_encoder(image)                                            # [bs, 8192, 3] -> [bs, 512, 384]
        image_embeds = image_embeds @ self.pc_projection                                    # [bs, 512, 384] -> [bs, 512, 768]        
        
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(             # [bs, 257]
            image.device
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)              # [1, 32, 768] -> [bs, 32, 768]
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,              # [bs, 32, 768]
            encoder_hidden_states=image_embeds,     # [bs, 257, ***1408***]
            encoder_attention_mask=image_atts,      # [bs, 257]
            use_cache=True,
            return_dict=True,
        )

        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )

        '''Text Bert'''
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(                                                             # [bs, 256]
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )
        
        
        '''============== Image-Point Contrastive ==================='''
        # temperature = 0.07
        # image_query = torch.max(query_output.last_hidden_state, dim=1)[0]                                          # [bs, num_query, dim] -> [bs, dim]
        # point_query = torch.max(query_output_point.last_hidden_state, dim=1)[0]                                   # [bs, num_query, dim] -> [bs, dim]

        # image_query = image_query / (torch.norm(image_query, p=2, dim=1, keepdim=True) + 1e-7)       
        # point_query = point_query / (torch.norm(point_query, p=2, dim=1, keepdim=True) + 1e-7)

        # sim_i2p = torch.mm(image_query, point_query.transpose(1, 0))                                  # [bs, bs]
        # sim_p2i = torch.mm(point_query, image_query.transpose(1, 0))                                  # [bs, bs]
        # labels = torch.arange(sim_i2p.shape[0], device=image_query.device).long()

        # loss_ipc = (torch.nn.CrossEntropyLoss(reduction="mean")(torch.div(sim_i2p, temperature), labels) + \
        #     torch.nn.CrossEntropyLoss(reduction="mean")(torch.div(sim_p2i, temperature), labels)) / 2


        
        ###============== Image-text Contrastive ===================###
        image_feats_all = concat_all_gather(                                                # [bs, 32, 256] -> [bs, 32, 256]
            image_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]     # [bs, 256] -> [bs, 256]

        sim_q2t = torch.matmul(                                                             # [bs, bs, 32]
            image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        rank = dist.get_rank()
        bs = image.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            image.device
        )

        if "image_id" in samples.keys(): #coco retrieval finetuning
            image_ids = samples["image_id"].view(-1,1)
            image_ids_all = concat_all_gather(image_ids)
            pos_idx = torch.eq(image_ids, image_ids_all.t()).float()       
            sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)   
            sim_targets = 0.9 * sim_targets + 0.1 * torch.ones_like(sim_targets) / sim_targets.size(1)

            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean()
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()     
            loss_itc = (loss_t2i+loss_i2t)/2  
        else:                     
            loss_itc = (
                F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
                + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
            ) / 2

        ###============== Image-text Matching ===================###
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)                 # [bs, 32]
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)       # [bs, 32]
        image_embeds_world = all_gather_with_grad(image_embeds)                         # [bs, 257, 1408]
        with torch.no_grad():
            if "image_id" in samples.keys():
                mask = torch.eq(image_ids, image_ids_all.t())
                sim_t2i.masked_fill_(mask, -10000)
                sim_i2t.masked_fill_(mask, -10000)
            else:    
                sim_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
                sim_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)            
                
            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_i2t = F.softmax(sim_i2t, dim=1)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
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

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)          # [3*bs, 32, 768]
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(     # [3*bs, 32]
            image.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)              # [3*bs, 32*2]

        image_embeds_all = torch.cat(                                                       # [3*bs, 257, 1408]
            [image_embeds, image_embeds_neg, image_embeds], dim=0
        )  # pos, neg, pos
        image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(     # [3*bs, 257]
            image.device
        )

        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]      # [3*bs, 32, 768]
        vl_output = self.itm_head(vl_embeddings)                                            # [3*bs, 32, 768] -> [3*bs, 32, 2]
        logits = vl_output.mean(dim=1)                                                      # [3*bs, 32, 2] -> [300, 2]s

        itm_labels = torch.cat(                                                             # [3*bs] bs:1, 2*bs
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(image.device)
        loss_itm = F.cross_entropy(logits, itm_labels)

        ##================= Image Captioning ========================##
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image.device
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

        return BlipOutput(
            loss= loss_itc + loss_itm + loss_lm,
            loss_itc=loss_itc,      # 4.7318
            loss_itm=loss_itm,      # 0.6541
            loss_lm=loss_lm,        # 9.2334
        )
        

        # return BlipOutput(
        #     loss=loss_ipc,
        #     loss_ipc=loss_ipc,      # 8.0712
        # )

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        point_model = cfg.get("point_model", "ulip_point_bert")
        point_num = cfg.get("point_num")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_point_encoder = cfg.get("freeze_point_encoder", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            point_model=point_model,
            point_num=point_num,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_point_encoder=freeze_point_encoder,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
