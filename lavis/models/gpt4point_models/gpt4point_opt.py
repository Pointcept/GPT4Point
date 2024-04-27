"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from lavis.common.registry import registry
from lavis.models.gpt4point_models.gpt4point import GPT4Point_Base, disabled_train
from transformers import AutoTokenizer, OPTForCausalLM
import transformers

from transformers.utils import logging as transformers_logging
transformers_logging.set_verbosity_error()

@registry.register_model("gpt4point_opt")
class GPT4Point_OPT(GPT4Point_Base):
    """
    GPT4Point OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "gpt4point_opt2.7b": "configs/models/gpt4point/gpt4point_caption_opt2.7b.yaml",
    }

    def __init__(
        self,
        point_model="ulip_point_bert",
        point_encoder_cfg=None,
        freeze_point_encoder=True,
        num_query_token=32,
        opt_model="facebook/opt-2.7b",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
        ckpt_special_strs=None,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.27"), "GPT4Point OPT requires transformers>=4.27"
        
        self.ckpt_special_strs = ckpt_special_strs
        self.tokenizer = self.init_tokenizer()

        '''Point: PIT'''
        self.point_encoder = GPT4Point_Base.init_point_encoder(point_model, point_encoder_cfg)
        '''Point projection'''
        self.pc_projection = nn.Parameter(torch.empty(point_encoder_cfg['trans_dim'], 1408))      # cannot be frozen cause it is defined by ourselves.

        '''Q-former and delete some of it.'''
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, 1408
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        '''OPT and freeze it'''
        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model, torch_dtype=torch.float16
        )
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        '''Projection'''
        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )

        '''Others'''
        self.max_txt_len = max_txt_len
        self.prompt = prompt # actually no prompt at all
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)
        
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        '''freeze PIT, pc_projection'''
        if freeze_point_encoder:
            for name, param in self.point_encoder.named_parameters():
                param.requires_grad = False
            logging.info("freeze point encoder")


    def forward(self, samples):
        point = samples["point"]
        with self.maybe_autocast():
            point_embeds = self.point_encoder(point)
            point_embeds = point_embeds @ self.pc_projection
        
        point_atts = torch.ones(point_embeds.size()[:-1], dtype=torch.long).to(
            point.device
        )

        query_tokens = self.query_tokens.expand(point_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=point_embeds,
            encoder_attention_mask=point_atts,
            return_dict=True,
        )

        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(point.device)

        self.opt_tokenizer.padding_side = "right"

        text = [t + "\n" for t in samples["text_input"]]

        opt_tokens = self.opt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(point.device)

        targets = opt_tokens.input_ids.masked_fill(
            opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

        empty_targets = (
            torch.ones(atts_opt.size(), dtype=torch.long).to(point.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                = pcd_id (list): the id of the point cloud
                - point (torch.Tensor): A tensor of shape (batch_size, num_points, 6)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each point cloud.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """

        point = samples["point"]
        with self.maybe_autocast():
            point_embeds = self.point_encoder(point)                                            # [bs, 8192, 3] -> [bs, 512, 384]
            point_embeds = point_embeds @ self.pc_projection                                    # [bs, 512, 384] -> [bs, 512, 768]  
            point_atts = torch.ones(point_embeds.size()[:-1], dtype=torch.long).to(
                point.device
            )
            query_tokens = self.query_tokens.expand(point_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=point_embeds,
                encoder_attention_mask=point_atts,
                return_dict=True,
            )

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                point.device
            )

            if "text_input" in samples.keys():
                prompt = samples["text_input"]
            else:
                self.prompt = 'a 3D point cloud of'
                prompt = self.prompt
                prompt = [prompt] * point.size(0)

            opt_tokens = self.opt_tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(point.device)
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            
            # new version for transformers>=4.27
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt,inputs_embeds],dim=1)
            
            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds, 
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

            output_text = [text.strip() for text in output_text]
            return output_text
        
    
    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer
        
    @classmethod
    def from_config(cls, cfg):
        point_model = cfg.get("point_model", "ulip_point_bert")
        point_encoder_cfg = cfg.get("point_encoder_cfg")
        num_query_token = cfg.get("num_query_token")
        opt_model = cfg.get("opt_model")
        freeze_point_encoder = cfg.get("freeze_point_encoder", True)
        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)
        ckpt_special_strs = cfg.get("ckpt_special_strs", None)

        model = cls(
            point_model=point_model,
            point_encoder_cfg=point_encoder_cfg,
            freeze_point_encoder=freeze_point_encoder,
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            ckpt_special_strs=ckpt_special_strs
        )
        model.load_checkpoint_from_config(cfg)

        return model