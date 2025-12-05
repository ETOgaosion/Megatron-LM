# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.language_model_cpu_embedding import LanguageModelCPUEmbedding
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestBaseEmbedding:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True, cpu_embedding=True
        )
        self.base_embedding = LanguageModelEmbedding(
            config=transformer_config,
            vocab_size=100,
            max_sequence_length=4,
            position_embedding_type='learned_absolute',
        )
        self.base_embedding_cpu = LanguageModelCPUEmbedding(
            config=transformer_config,
            vocab_size=100,
            max_sequence_length=4,
            position_embedding_type='learned_absolute',
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.base_embedding, LanguageModelEmbedding)
        assert isinstance(self.base_embedding_cpu, LanguageModelCPUEmbedding)
        num_weights = sum([p.numel() for p in self.base_embedding.parameters()])
        num_weights_cpu = sum([p.numel() for p in self.base_embedding_cpu.parameters()])
        assert num_weights == 1248
        assert num_weights_cpu == 1248

    def test_zero_parameters(self):
        sum_weights = sum([p.sum() for p in self.base_embedding.parameters()])
        sum_weights_cpu = sum([p.sum() for p in self.base_embedding_cpu.parameters()])
        assert sum_weights != 0
        assert sum_weights_cpu != 0
        self.base_embedding.zero_parameters()
        self.base_embedding_cpu.zero_parameters()
        sum_weights = sum([p.sum() for p in self.base_embedding.parameters()])
        sum_weights_cpu = sum([p.sum() for p in self.base_embedding_cpu.parameters()])
        assert sum_weights == 0
        assert sum_weights_cpu == 0

    def test_cpu_forward(self):
        input_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64).repeat((2, 1))
        position_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64).repeat((2, 1))
        embeddings = self.base_embedding(input_ids, position_ids)
        embeddings_cpu = self.base_embedding_cpu(input_ids, position_ids)
        assert embeddings.device.type == 'cpu'
        assert embeddings.shape[0] == self.base_embedding.max_sequence_length
        assert embeddings.shape[1] == input_ids.shape[0]
        assert embeddings.shape[2] == self.base_embedding.config.hidden_size
        assert embeddings_cpu.device.type == 'cpu'
        assert embeddings_cpu.shape[0] == self.base_embedding_cpu.max_sequence_length
        assert embeddings_cpu.shape[1] == input_ids.shape[0]
        assert embeddings_cpu.shape[2] == self.base_embedding_cpu.config.hidden_size

    def test_gpu_forward(self):
        self.base_embedding.cuda()
        input_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64).repeat((2, 1)).cuda()
        position_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64).repeat((2, 1)).cuda()
        embeddings = self.base_embedding(input_ids, position_ids)
        assert embeddings.device.type == 'cuda'
        assert embeddings.shape[0] == self.base_embedding.max_sequence_length
        assert embeddings.shape[1] == input_ids.shape[0]
        assert embeddings.shape[2] == self.base_embedding.config.hidden_size
    
    def test_cpu_forward(self):
        input_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64).repeat((2, 1)).cpu()
        position_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64).repeat((2, 1)).cpu()
        embeddings = self.base_embedding_cpu(input_ids, position_ids)
        assert embeddings.device.type == 'cuda'
        assert embeddings.shape[0] == self.base_embedding_cpu.max_sequence_length
        assert embeddings.shape[1] == input_ids.shape[0]
        assert embeddings.shape[2] == self.base_embedding_cpu.config.hidden_size
