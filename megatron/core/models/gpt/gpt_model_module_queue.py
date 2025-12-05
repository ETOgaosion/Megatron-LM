# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from collections import OrderedDict
from typing import Dict, Literal, Optional

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings import YarnRotaryEmbedding
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    MultimodalRotaryEmbedding,
    RotaryEmbedding,
)
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.quantization.utils import get_quant_config_or_none
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.multi_token_prediction import (
    MTPLossAutoScaler,
    MTPLossLoggingHelper,
    MultiTokenPredictionBlock,
    roll_tensor,
    tie_output_layer_state_dict,
    tie_word_embeddings_state_dict,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import WrappedTensor, deprecate_inference_params
from megatron.core.models.gpt.gpt_model import GPTModel


class GPTModelModuleQueue(GPTModel):
    
    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal[
            'learned_absolute', 'rope', 'mrope', 'yarn', 'none'
        ] = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
        mtp_block_spec: Optional[ModuleSpec] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            rope_scaling=rope_scaling,
            rope_scaling_factor=rope_scaling_factor,
            scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )
    
        assert self.post_process, "GPTModelModuleQueue only applys to post_process models."