"""
 Copyright (c) 2023, pjlab.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)


@dataclass
class GPT4Point_Similarity(ModelOutput):
    sim_p2t: torch.FloatTensor = None
    sim_t2p: torch.FloatTensor = None

    sim_p2t_m: Optional[torch.FloatTensor] = None
    sim_t2p_m: Optional[torch.FloatTensor] = None

    sim_p2t_targets: Optional[torch.FloatTensor] = None
    sim_t2p_targets: Optional[torch.FloatTensor] = None


@dataclass
class GPT4Point_IntermediateOutput(ModelOutput):
    """
    Data class for intermediate outputs of BLIP models.

    point_embeds (torch.FloatTensor): point embeddings, shape (batch_size, num_patches, embed_dim).
    text_embeds (torch.FloatTensor): Text embeddings, shape (batch_size, seq_len, embed_dim).

    point_embeds_m (torch.FloatTensor): point embeddings from momentum visual encoder, shape (batch_size, num_patches, embed_dim).
    text_embeds_m (torch.FloatTensor): Text embeddings from momentum text encoder, shape (batch_size, seq_len, embed_dim).

    encoder_output (BaseModelOutputWithPoolingAndCrossAttentions): output from the point-grounded text encoder.
    encoder_output_neg (BaseModelOutputWithPoolingAndCrossAttentions): output from the point-grounded text encoder for negative pairs.

    decoder_output (CausalLMOutputWithCrossAttentions): output from the point-grounded text decoder.
    decoder_labels (torch.LongTensor): labels for the captioning loss.

    ptm_logits (torch.FloatTensor): logits for the point-text matching loss, shape (batch_size * 3, 2).
    ptm_labels (torch.LongTensor): labels for the point-text matching loss, shape (batch_size * 3,)

    """

    # uni-modal features
    point_embeds: torch.FloatTensor = None
    text_embeds: Optional[torch.FloatTensor] = None

    point_embeds_m: Optional[torch.FloatTensor] = None
    text_embeds_m: Optional[torch.FloatTensor] = None

    # intermediate outputs of multimodal encoder
    encoder_output: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None
    encoder_output_neg: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None

    ptm_logits: Optional[torch.FloatTensor] = None
    ptm_labels: Optional[torch.LongTensor] = None

    # intermediate outputs of multimodal decoder
    decoder_output: Optional[CausalLMOutputWithCrossAttentions] = None
    decoder_labels: Optional[torch.LongTensor] = None


@dataclass
class GPT4Point_Output(ModelOutput):
    # some finetuned models (e.g. BlipVQA) do not compute similarity, thus optional.
    sims: Optional[GPT4Point_Similarity] = None

    intermediate_output: GPT4Point_IntermediateOutput = None

    loss: Optional[torch.FloatTensor] = None

    loss_ptc: Optional[torch.FloatTensor] = None

    loss_ptm: Optional[torch.FloatTensor] = None

    loss_lm: Optional[torch.FloatTensor] = None


@dataclass
class GPT4Point_OutputFeatures(ModelOutput):
    """
    Data class of features from BlipFeatureExtractor.

    Args:
        point_embeds: (torch.FloatTensor) of shape (batch_size, num_patches+1, embed_dim), optional
        point_features: (torch.FloatTensor) of shape (batch_size, num_patches+1, feature_dim), optional
        text_embeds: (torch.FloatTensor) of shape (batch_size, sequence_length+1, embed_dim), optional
        text_features: (torch.FloatTensor) of shape (batch_size, sequence_length+1, feature_dim), optional

        The first embedding or feature is for the [CLS] token.

        Features are obtained by projecting the corresponding embedding into a normalized low-dimensional space.
    """

    point_embeds: Optional[torch.FloatTensor] = None
    point_embeds_proj: Optional[torch.FloatTensor] = None

    text_embeds: Optional[torch.FloatTensor] = None
    text_embeds_proj: Optional[torch.FloatTensor] = None

    multimodal_embeds: Optional[torch.FloatTensor] = None