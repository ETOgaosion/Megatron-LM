# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""GPT Model selector that chooses between GPTModelNormal and GPTModelModuleQueue.

This module provides the GPTModel class which acts as a factory to select
the appropriate implementation based on configuration:
- If post_process=True AND pipeline_parallel > 1 AND enable_module_queue=True:
    Use GPTModelModuleQueue for memory-efficient training on last pipeline stage
- Otherwise:
    Use GPTModelNormal (standard implementation)
"""

from typing import Literal, Optional

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

# Import both implementations
from megatron.core.models.gpt.gpt_model_normal import GPTModelNormal
from megatron.core.models.gpt.gpt_model_module_queue import GPTModelModuleQueue


def GPTModel(
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
):
    """Factory function to create appropriate GPT model implementation.

    Selects between GPTModelModuleQueue and GPTModelNormal based on configuration:
    - GPTModelModuleQueue: Used when post_process=True, PP > 1, and enable_module_queue=True
    - GPTModelNormal: Used in all other cases

    Args:
        config (TransformerConfig): Transformer config
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers
        vocab_size (int): Vocabulary size
        max_sequence_length (int): Maximum size of sequence for positional embedding
        pre_process (bool): Include embedding layer (pipeline parallelism). Defaults to True.
        post_process (bool): Include output layer (pipeline parallelism). Defaults to True.
        fp16_lm_cross_entropy (bool): Defaults to False.
        parallel_output (bool): Keep outputs split across tensor parallel ranks. Defaults to True.
        share_embeddings_and_output_weights (bool): Share input embeddings and output weights. Defaults to False.
        position_embedding_type: Position embedding type. Defaults to 'learned_absolute'.
        rotary_percent (float): Percent of rotary dimension for RoPE. Defaults to 1.0.
        rotary_base (int): Base period for RoPE. Defaults to 10000.
        rope_scaling (bool): Toggle RoPE scaling. Defaults to False.
        rope_scaling_factor (float): RoPE scaling factor. Defaults to 8.0.
        scatter_embedding_sequence_parallel (bool): Scatter embeddings in SP. Defaults to True.
        seq_len_interpolation_factor (Optional[float]): RoPE interpolation factor. Defaults to None.
        mtp_block_spec (Optional[ModuleSpec]): Multi-token prediction block spec. Defaults to None.
        pg_collection (Optional[ProcessGroupCollection]): Process groups. Defaults to None.
        vp_stage (Optional[int]): Virtual pipeline stage. Defaults to None.

    Returns:
        Union[GPTModelModuleQueue, GPTModelNormal]: The appropriate GPT model implementation.
    """
    # Determine if we should use GPTModelModuleQueue
    # Conditions: post_process=True AND PP > 1 AND enable_module_queue=True
    use_module_queue = False

    if post_process and getattr(config, 'enable_module_queue', False):
        # Check if pipeline parallel is enabled (PP > 1)
        pp_size = 1
        if pg_collection is not None and pg_collection.pp is not None:
            pp_size = pg_collection.pp.size()
        elif parallel_state.is_initialized():
            pp_size = parallel_state.get_pipeline_model_parallel_world_size()

        if pp_size > 1:
            use_module_queue = True

    # Select and instantiate the appropriate model
    model_class = GPTModelModuleQueue if use_module_queue else GPTModelNormal

    return model_class(
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
        mtp_block_spec=mtp_block_spec,
        pg_collection=pg_collection,
        vp_stage=vp_stage,
    )


# For backward compatibility, also export GPTModelNormal
__all__ = ['GPTModel', 'GPTModelNormal', 'GPTModelModuleQueue']
