import torch
from torch import nn
import torch.nn.functional as f
from typing import List, Optional, Tuple, Union


class BertEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size,config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values_length: int=0):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

