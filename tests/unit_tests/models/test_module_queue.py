# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import os

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.module_queue_gpt_model import ModuleQueue
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestModuleQueue:
    """Unit tests for ModuleQueue class."""

    def setup_method(self, method):
        """Set up test environment."""
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        """Clean up test environment."""
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_module_queue_disabled(self):
        """Test ModuleQueue with module queue disabled behaves like GPTModel."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_module_queue=False,  # Disabled
        )
        module_queue_model = ModuleQueue(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
            post_process=True,
        )

        # Should behave like regular GPTModel when disabled
        assert isinstance(module_queue_model, GPTModel)
        assert not module_queue_model._module_queue_enabled

    @pytest.mark.internal
    def test_module_queue_enabled_post_process(self):
        """Test ModuleQueue with module queue enabled on post_process stage."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_module_queue=True,  # Enabled
            module_queue_num_chunks=2,
        )
        module_queue_model = ModuleQueue(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
            post_process=True,
        )

        # Should enable module queue
        assert isinstance(module_queue_model, GPTModel)
        assert module_queue_model._module_queue_enabled
        assert module_queue_model.num_chunks == 2
        assert hasattr(module_queue_model, 'cpu_queue')
        assert hasattr(module_queue_model, 'gpu_queue')
        assert hasattr(module_queue_model, 'output_layer_weight_chunks')

    @pytest.mark.internal
    def test_module_queue_not_enabled_without_post_process(self):
        """Test ModuleQueue doesn't enable without post_process=True."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_module_queue=True,  # Enabled in config
        )
        module_queue_model = ModuleQueue(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
            post_process=False,  # Not post_process
        )

        # Should NOT enable module queue
        assert isinstance(module_queue_model, GPTModel)
        assert not module_queue_model._module_queue_enabled

    @pytest.mark.internal
    def test_configuration_parameters(self):
        """Test that configuration parameters are properly defined."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_module_queue=True,
            module_queue_num_chunks=4,
        )

        assert hasattr(transformer_config, 'enable_module_queue')
        assert hasattr(transformer_config, 'module_queue_num_chunks')
        assert transformer_config.enable_module_queue == True
        assert transformer_config.module_queue_num_chunks == 4

    @pytest.mark.internal
    def test_output_layer_chunking(self):
        """Test that output layer is properly chunked."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_module_queue=True,
            module_queue_num_chunks=3,
        )
        module_queue_model = ModuleQueue(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=99,  # Not divisible by 3 to test chunking edge cases
            max_sequence_length=4,
            post_process=True,
        )

        # Check chunks were created
        assert len(module_queue_model.output_layer_weight_chunks) == 3
        assert len(module_queue_model.output_layer_weight_chunks_gpu) == 3
        assert len(module_queue_model.output_layer_chunks_on_gpu) == 3

        # All chunks should initially be on CPU (None in GPU list)
        for chunk_gpu in module_queue_model.output_layer_weight_chunks_gpu:
            assert chunk_gpu is None

        # Check chunk sizes
        total_vocab = 0
        for chunk in module_queue_model.output_layer_weight_chunks:
            assert chunk.device.type == 'cpu'
            total_vocab += chunk.size(0)
        assert total_vocab == 99

    @pytest.mark.internal
    def test_forward_inference_mode(self):
        """Test forward pass in inference mode (module queue should be inactive)."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_module_queue=True,
            module_queue_num_chunks=2,
        )
        module_queue_model = ModuleQueue(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
            post_process=True,
        )

        sequence_length = 4
        micro_batch_size = 2

        module_queue_model.cuda()
        module_queue_model.eval()  # Set to eval mode

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        # Should work in inference mode without module queue active
        with torch.no_grad():
            logits = module_queue_model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

        assert logits is not None
        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == 100

    @pytest.mark.internal
    def test_layer_tracking(self):
        """Test that layer tracking state is properly initialized."""
        transformer_config = TransformerConfig(
            num_layers=3,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_module_queue=True,
            module_queue_num_chunks=2,
        )
        module_queue_model = ModuleQueue(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
            post_process=True,
        )

        # Check layer tracking initialization
        assert module_queue_model.num_layers == 3
        assert len(module_queue_model.layers_on_cpu) == 3
        assert all(layer_cpu == False for layer_cpu in module_queue_model.layers_on_cpu)

    @pytest.mark.internal
    def test_inheritance(self):
        """Test that ModuleQueue properly inherits from GPTModel."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_module_queue=True,
        )
        module_queue_model = ModuleQueue(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
            post_process=True,
        )

        # Should be instance of both ModuleQueue and GPTModel
        assert isinstance(module_queue_model, ModuleQueue)
        assert isinstance(module_queue_model, GPTModel)

        # Should have all GPTModel attributes
        assert hasattr(module_queue_model, 'decoder')
        assert hasattr(module_queue_model, 'embedding')
        assert hasattr(module_queue_model, 'output_layer')

    @pytest.mark.internal
    def test_hooks_registered(self):
        """Test that forward and backward hooks are registered."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_module_queue=True,
        )
        module_queue_model = ModuleQueue(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
            post_process=True,
        )

        # Check hooks are registered
        assert hasattr(module_queue_model, '_forward_hooks')
        assert hasattr(module_queue_model, '_backward_hooks')
        assert len(module_queue_model._forward_hooks) == 2  # One per layer

    @pytest.mark.internal
    def test_streams_initialized(self):
        """Test that CUDA streams are initialized."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_module_queue=True,
        )
        module_queue_model = ModuleQueue(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
            post_process=True,
        )

        # Check streams are created
        assert hasattr(module_queue_model, 'h2d_stream')
        assert hasattr(module_queue_model, 'd2h_stream')
        assert isinstance(module_queue_model.h2d_stream, torch.cuda.Stream)
        assert isinstance(module_queue_model.d2h_stream, torch.cuda.Stream)

    @pytest.mark.internal
    def test_ensure_output_layer_ready(self):
        """Test _ensure_output_layer_ready method."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_module_queue=True,
            module_queue_num_chunks=2,
        )
        module_queue_model = ModuleQueue(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
            post_process=True,
        )

        module_queue_model.cuda()

        # Initially chunks should not be on GPU
        assert not any(module_queue_model.output_layer_chunks_on_gpu)

        # Call _ensure_output_layer_ready
        module_queue_model._ensure_output_layer_ready()

        # Now all chunks should be on GPU
        assert all(module_queue_model.output_layer_chunks_on_gpu)

        # Output layer weight should be assembled
        assert module_queue_model.output_layer.weight.data.shape[0] == 100

    @pytest.mark.internal
    def test_default_config_values(self):
        """Test default configuration values."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )

        # Check defaults
        assert transformer_config.enable_module_queue == False
        assert transformer_config.module_queue_num_chunks == 4
