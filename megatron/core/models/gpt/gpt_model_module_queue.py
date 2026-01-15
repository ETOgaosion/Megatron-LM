# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from collections import deque
from typing import Literal, Optional

import torch
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.gpt_model_normal import GPTModelNormal
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


class GPTModelModuleQueue(GPTModelNormal):
    """GPT Model with Module Queue for memory-efficient training on last pipeline stage.

    This class extends GPTModelNormal to implement a module queue mechanism that:
    1. Keeps post-process (output layer) weights on CPU initially
    2. In forward pass: offloads computed transformer layers to CPU while loading
       chunks of post-process weights to GPU
    3. In backward pass: offloads post-process weights after computation and loads
       transformer layers back from CPU

    This approach reduces peak GPU memory usage by overlapping computation with
    data transfers between CPU and GPU.

    Only used when post_process=True (last pipeline stage).

    Args:
        See GPTModel for standard arguments.
        Additional behavior controlled by TransformerConfig.enable_module_queue
        and TransformerConfig.module_queue_num_chunks.
    """

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
        # Only enable module queue for post_process stage
        if not post_process or not config.enable_module_queue:
            # If not post_process or module queue disabled, just use regular GPTModel
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
                mtp_block_spec=mtp_block_spec,
                pg_collection=pg_collection,
                vp_stage=vp_stage,
            )
            self._module_queue_enabled = False
            return

        # Initialize parent class
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
            mtp_block_spec=mtp_block_spec,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )

        self._module_queue_enabled = True

        # Initialize module queue state
        self.num_chunks = config.module_queue_num_chunks
        self.cpu_queue = deque()  # Queue of modules on CPU
        self.gpu_queue = deque()  # Queue of modules on GPU

        # Initialize streams for async transfers
        self.h2d_stream = torch.cuda.Stream()  # Host to Device (CPU to GPU)
        self.d2h_stream = torch.cuda.Stream()  # Device to Host (GPU to CPU)

        # Track which layers are on CPU vs GPU
        self.num_layers = len(self.decoder.layers)
        self.layers_on_cpu = [False] * self.num_layers
        self.output_layer_chunks_on_gpu = [False] * self.num_chunks

        # Store original output layer on CPU and create chunks
        self._initialize_output_layer_chunks()

        # Register hooks for automatic weight movement during backward pass
        self._forward_hooks = []
        self._backward_hooks = []
        self._register_hooks()

    def _initialize_output_layer_chunks(self):
        """Initialize output layer chunks on CPU."""
        if not hasattr(self, 'output_layer'):
            return

        # Move output layer to CPU and create chunks
        output_layer_weight = self.output_layer.weight.data.cpu()
        self.output_layer.weight.data = torch.empty(
            0, dtype=output_layer_weight.dtype, device='cuda'
        )

        # Split weight into chunks along output dimension (vocab_size)
        vocab_size_per_chunk = (output_layer_weight.size(0) + self.num_chunks - 1) // self.num_chunks

        self.output_layer_weight_chunks = []
        for i in range(self.num_chunks):
            start_idx = i * vocab_size_per_chunk
            end_idx = min((i + 1) * vocab_size_per_chunk, output_layer_weight.size(0))
            chunk = output_layer_weight[start_idx:end_idx].clone()
            self.output_layer_weight_chunks.append(chunk)

        # Keep track of which chunks are loaded
        self.output_layer_weight_chunks_gpu = [None] * self.num_chunks
        self.chunk_start_indices = []
        for i in range(self.num_chunks):
            start_idx = i * vocab_size_per_chunk
            self.chunk_start_indices.append(start_idx)

    def _register_hooks(self):
        """Register forward and backward hooks for automatic weight management."""
        # Register forward hooks on transformer layers
        for layer_idx, layer in enumerate(self.decoder.layers):
            hook = layer.register_forward_hook(
                self._create_layer_forward_hook(layer_idx)
            )
            self._forward_hooks.append(hook)

        # Register backward hook on output layer
        if hasattr(self, 'output_layer') and hasattr(self.output_layer, 'weight'):
            hook = self.output_layer.weight.register_hook(
                self._create_output_layer_backward_hook()
            )
            self._backward_hooks.append(hook)

    def _create_layer_forward_hook(self, layer_idx):
        """Create forward hook for a transformer layer."""
        def hook(module, input, output):
            if self.training and self._module_queue_enabled:
                self._offload_layer_after_forward(layer_idx)
                self._load_output_layer_chunk_if_needed(layer_idx)
            return output
        return hook

    def _create_output_layer_backward_hook(self):
        """Create backward hook for output layer."""
        def hook(grad):
            if self.training and self._module_queue_enabled:
                self._offload_output_layer_after_backward()
                self._load_layers_for_backward()
            return grad
        return hook

    def _offload_layer_after_forward(self, layer_idx):
        """Offload transformer layer to CPU after forward computation."""
        if self.layers_on_cpu[layer_idx]:
            return  # Already on CPU

        layer = self.decoder.layers[layer_idx]

        # Use async transfer
        with torch.cuda.stream(self.d2h_stream):
            for name, param in layer.named_parameters():
                if param.data.device.type == 'cuda':
                    param_cpu = param.data.cpu()
                    param.data = param_cpu
                    # Keep gradient on GPU for now

        self.layers_on_cpu[layer_idx] = True
        self.cpu_queue.append(('layer', layer_idx))

    def _load_output_layer_chunk_if_needed(self, layer_idx):
        """Load a chunk of output layer weights from CPU to GPU."""
        # Calculate which chunk to load based on layer progress
        chunk_idx = (layer_idx * self.num_chunks) // self.num_layers
        chunk_idx = min(chunk_idx, self.num_chunks - 1)

        if self.output_layer_chunks_on_gpu[chunk_idx]:
            return  # Already loaded

        # Load chunk to GPU asynchronously
        with torch.cuda.stream(self.h2d_stream):
            chunk_gpu = self.output_layer_weight_chunks[chunk_idx].cuda(non_blocking=True)
            self.output_layer_weight_chunks_gpu[chunk_idx] = chunk_gpu

        self.output_layer_chunks_on_gpu[chunk_idx] = True
        self.gpu_queue.append(('output_chunk', chunk_idx))

    def _offload_output_layer_after_backward(self):
        """Offload output layer chunks from GPU to CPU after backward."""
        for chunk_idx in range(self.num_chunks):
            if self.output_layer_chunks_on_gpu[chunk_idx]:
                # Move chunk back to CPU
                with torch.cuda.stream(self.d2h_stream):
                    chunk_gpu = self.output_layer_weight_chunks_gpu[chunk_idx]
                    if chunk_gpu is not None:
                        self.output_layer_weight_chunks[chunk_idx] = chunk_gpu.cpu()
                        self.output_layer_weight_chunks_gpu[chunk_idx] = None

                self.output_layer_chunks_on_gpu[chunk_idx] = False

    def _load_layers_for_backward(self):
        """Load transformer layers back to GPU for backward pass."""
        # Load layers in reverse order (last layer first for backward)
        for layer_idx in reversed(range(self.num_layers)):
            if self.layers_on_cpu[layer_idx]:
                self._load_layer_to_gpu(layer_idx)

    def _load_layer_to_gpu(self, layer_idx):
        """Load a transformer layer from CPU to GPU."""
        if not self.layers_on_cpu[layer_idx]:
            return  # Already on GPU

        layer = self.decoder.layers[layer_idx]

        # Use async transfer
        with torch.cuda.stream(self.h2d_stream):
            for name, param in layer.named_parameters():
                if param.data.device.type != 'cuda':
                    param_gpu = param.data.cuda(non_blocking=True)
                    param.data = param_gpu

        self.layers_on_cpu[layer_idx] = False

    def _ensure_layers_on_gpu(self):
        """Ensure all transformer layers are on GPU before forward pass."""
        for layer_idx in range(self.num_layers):
            if self.layers_on_cpu[layer_idx]:
                self._load_layer_to_gpu(layer_idx)
        # Synchronize to ensure all transfers are complete
        torch.cuda.current_stream().wait_stream(self.h2d_stream)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with module queue management.

        If module queue is not enabled or not in training mode, this delegates
        to the parent GPTModel.forward(). Otherwise, it manages layer offloading
        and output layer chunk loading during the forward pass.
        """
        # For non-first pipeline stages (pre_process=False), TransformerBlock expects
        # input via input_tensor because it ignores the hidden_states argument.
        # Set the decoder's input tensor when decoder_input is provided.
        if decoder_input is not None and not self.pre_process:
            self.decoder.set_input_tensor(decoder_input)

        # Ensure all layers are on GPU before forward pass
        if self._module_queue_enabled:
            self._ensure_layers_on_gpu()

        if not self._module_queue_enabled or not self.training:
            return super().forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=decoder_input,
                labels=labels,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
                extra_block_kwargs=extra_block_kwargs,
                runtime_gather_output=runtime_gather_output,
                inference_params=inference_params,
                loss_mask=loss_mask,
            )

        # Ensure all output layer chunks are loaded before forward pass
        self._ensure_output_layer_ready()

        # Call parent forward - hooks will handle layer offloading
        return super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
            runtime_gather_output=runtime_gather_output,
            inference_params=inference_params,
            loss_mask=loss_mask,
        )

    def _ensure_output_layer_ready(self):
        """Ensure output layer is ready for forward pass by assembling all chunks."""
        if not hasattr(self, 'output_layer'):
            return

        # Load all chunks if not already loaded
        for chunk_idx in range(self.num_chunks):
            if not self.output_layer_chunks_on_gpu[chunk_idx]:
                chunk_gpu = self.output_layer_weight_chunks[chunk_idx].cuda()
                self.output_layer_weight_chunks_gpu[chunk_idx] = chunk_gpu
                self.output_layer_chunks_on_gpu[chunk_idx] = True

        # Assemble full weight from chunks
        weight_list = [chunk for chunk in self.output_layer_weight_chunks_gpu if chunk is not None]
        if weight_list:
            full_weight = torch.cat(weight_list, dim=0)
            self.output_layer.weight.data = full_weight

    def __del__(self):
        """Cleanup hooks when object is destroyed."""
        for hook in self._forward_hooks:
            if isinstance(hook, RemovableHandle):
                hook.remove()
        for hook in self._backward_hooks:
            if isinstance(hook, RemovableHandle):
                hook.remove()
