# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import os

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.module_queue_gpt_model import ModuleQueue
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestModuleQueuePostProcess:
    """Test ModuleQueue for post_process training only.

    ModuleQueue is designed specifically for the last pipeline stage (post_process=True)
    to enable memory-efficient training by offloading transformer layers to CPU
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
        """Create a ModuleQueue model with post_process=True."""
        transformer_config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            use_cpu_initialization=True,
            enable_module_queue=enable_module_queue,
            module_queue_num_chunks=num_chunks,
        )
        model = ModuleQueue(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
            pre_process=True,
            post_process=True,  # ModuleQueue only works with post_process=True
        )
        return model

    @pytest.mark.internal
    def test_constructor_with_post_process(self):
        """Test that ModuleQueue initializes correctly with post_process=True."""
        model = self._create_module_queue_model()

        assert isinstance(model, ModuleQueue)
        assert model._module_queue_enabled is True
        assert model.max_sequence_length == 4
        assert model.num_chunks == 2
        assert len(model.decoder.layers) == 2
        assert hasattr(model, 'output_layer')
        assert len(model.output_layer_weight_chunks) == 2

    @pytest.mark.internal
    def test_module_queue_disabled_when_no_post_process(self):
        """Test that ModuleQueue falls back to regular GPTModel when post_process=False."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_module_queue=True,
            module_queue_num_chunks=2,
        )
        model = ModuleQueue(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
            pre_process=True,
            post_process=False,  # Not post_process stage
        )

        assert isinstance(model, ModuleQueue)
        assert model._module_queue_enabled is False

    @pytest.mark.internal
    def test_module_queue_disabled_when_config_disabled(self):
        """Test that ModuleQueue is disabled when enable_module_queue=False in config."""
        model = self._create_module_queue_model(enable_module_queue=False)

        assert isinstance(model, ModuleQueue)
        assert model._module_queue_enabled is False

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

        sequence_length = model.max_sequence_length
        micro_batch_size = 2

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        logits = model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
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

        sequence_length = model.max_sequence_length
        micro_batch_size = 2

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        logits = model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
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

        sequence_length = model.max_sequence_length
        micro_batch_size = 2

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        # Run multiple forward passes
        for _ in range(3):
            logits = model.forward(
                input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
            )
            assert logits.shape == (micro_batch_size, sequence_length, model.vocab_size)

