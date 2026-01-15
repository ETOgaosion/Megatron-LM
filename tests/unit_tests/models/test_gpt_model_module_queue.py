# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import os

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model_normal import GPTModelNormal
from megatron.core.models.gpt.gpt_model_module_queue import GPTModelModuleQueue
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestGPTModelModuleQueuePostProcess:
    """Test GPTModelModuleQueue for post_process training only.

    GPTModelModuleQueue is designed specifically for the last pipeline stage:
    - pre_process=False (no embedding layer, receives hidden states from previous stage)
    - post_process=True (has output layer for final logits)

    This enables memory-efficient training by offloading transformer layers to CPU
    while loading output layer chunks.
    """

    def setup_method(self, method):
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _create_module_queue_model(
        self, num_layers=2, hidden_size=12, num_attention_heads=4, num_chunks=2, enable_module_queue=True
    ):
        """Create a GPTModelModuleQueue model for last pipeline stage (pre_process=False, post_process=True)."""
        transformer_config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            use_cpu_initialization=True,
            enable_module_queue=enable_module_queue,
            module_queue_num_chunks=num_chunks,
        )
        model = GPTModelModuleQueue(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
            pre_process=False,  # Last pipeline stage: no embedding layer
            post_process=True,  # Last pipeline stage: has output layer
        )
        return model

    @pytest.mark.internal
    def test_constructor_with_post_process(self):
        """Test that GPTModelModuleQueue initializes correctly for last pipeline stage."""
        model = self._create_module_queue_model()

        assert isinstance(model, GPTModelModuleQueue)
        assert isinstance(model, GPTModelNormal)  # GPTModelModuleQueue inherits from GPTModelNormal
        assert model._module_queue_enabled is True
        # Verify last pipeline stage configuration
        assert model.pre_process is False, "GPTModelModuleQueue should have pre_process=False (last stage)"
        assert model.post_process is True, "GPTModelModuleQueue should have post_process=True (last stage)"
        # Verify no embedding layer (pre_process=False)
        assert not hasattr(model, 'embedding') or model.embedding is None
        # Verify has output layer (post_process=True)
        assert hasattr(model, 'output_layer')
        assert model.max_sequence_length == 4
        assert model.num_chunks == 2
        assert len(model.decoder.layers) == 2
        assert len(model.output_layer_weight_chunks) == 2

    @pytest.mark.internal
    def test_module_queue_disabled_when_no_post_process(self):
        """Test that GPTModelModuleQueue falls back to regular GPTModelNormal behavior when post_process=False."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_module_queue=True,
            module_queue_num_chunks=2,
        )
        # Middle pipeline stage: no embedding, no output layer
        model = GPTModelModuleQueue(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
            pre_process=False,
            post_process=False,  # Not post_process stage - module queue should be disabled
        )

        assert isinstance(model, GPTModelModuleQueue)
        assert model._module_queue_enabled is False

    @pytest.mark.internal
    def test_module_queue_disabled_when_config_disabled(self):
        """Test that GPTModelModuleQueue is disabled when enable_module_queue=False in config."""
        model = self._create_module_queue_model(enable_module_queue=False)

        assert isinstance(model, GPTModelModuleQueue)
        assert model._module_queue_enabled is False
        # Still verify last pipeline stage configuration
        assert model.pre_process is False
        assert model.post_process is True

    @pytest.mark.internal
    def test_output_layer_chunks_initialization(self):
        """Test that output layer chunks are correctly initialized on CPU."""
        model = self._create_module_queue_model(num_chunks=4)

        assert len(model.output_layer_weight_chunks) == 4
        assert len(model.output_layer_weight_chunks_gpu) == 4
        assert all(chunk is None for chunk in model.output_layer_weight_chunks_gpu)
        assert all(chunk.device.type == 'cpu' for chunk in model.output_layer_weight_chunks)

    @pytest.mark.internal
    def test_forward_post_process_training(self):
        """Test forward pass with post_process training mode."""
        model = self._create_module_queue_model()
        model.cuda()
        model.train()

        config = model.config
        sequence_length = model.max_sequence_length
        micro_batch_size = 2

        # Since pre_process=False, we provide decoder_input (hidden states) instead of input_ids
        # Shape: [sequence_length, batch_size, hidden_size]
        decoder_input = torch.randn(
            sequence_length, micro_batch_size, config.hidden_size, dtype=torch.float32
        ).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        logits = model.forward(
            input_ids=None, position_ids=None, attention_mask=attention_mask, decoder_input=decoder_input
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == model.vocab_size

    @pytest.mark.internal
    def test_forward_backward_post_process_training(self):
        """Test forward and backward pass with post_process training mode."""
        model = self._create_module_queue_model()
        model.cuda()
        model.train()

        config = model.config
        sequence_length = model.max_sequence_length
        micro_batch_size = 2

        # Since pre_process=False, we provide decoder_input (hidden states) instead of input_ids
        decoder_input = torch.randn(
            sequence_length, micro_batch_size, config.hidden_size, dtype=torch.float32
        ).cuda()
        decoder_input.requires_grad = True
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        logits = model.forward(
            input_ids=None, position_ids=None, attention_mask=attention_mask, decoder_input=decoder_input
        )

        # Compute loss and backward
        loss = logits.sum()
        loss.backward()

        # Verify gradients exist for model parameters
        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads, "Model should have gradients after backward pass"

    @pytest.mark.internal
    def test_output_layer_assembly(self):
        """Test that output layer is correctly assembled from chunks during forward."""
        model = self._create_module_queue_model(num_chunks=2)
        model.cuda()
        model.train()

        # Manually ensure output layer is ready
        model._ensure_output_layer_ready()

        # Verify output layer weight is assembled
        assert model.output_layer.weight.data.device.type == 'cuda'
        assert model.output_layer.weight.data.shape[0] == model.vocab_size

    @pytest.mark.internal
    def test_multiple_forward_passes(self):
        """Test multiple forward passes work correctly."""
        model = self._create_module_queue_model()
        model.cuda()
        model.train()

        config = model.config
        sequence_length = model.max_sequence_length
        micro_batch_size = 2

        # Since pre_process=False, we provide decoder_input (hidden states) instead of input_ids
        decoder_input = torch.randn(
            sequence_length, micro_batch_size, config.hidden_size, dtype=torch.float32
        ).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        # Run multiple forward passes
        for _ in range(3):
            logits = model.forward(
                input_ids=None, position_ids=None, attention_mask=attention_mask, decoder_input=decoder_input
            )
            assert logits.shape == (micro_batch_size, sequence_length, model.vocab_size)

