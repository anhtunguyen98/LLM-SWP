import torch
import torch.nn as nn
from models.bert_layer import BertLayer, BertPooler
from models.bert_embedding import BertEmbeddings
from typing import List, Optional, Tuple, Union
from models.bert_output import BertOutput

class BertModel(nn.Module):

    def __init__(self, config):
        super(BertModel, self).__init__()

        self.num_blocks = config.num_hidden_layers
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embedding = BertEmbeddings(config)
        self.encoder = nn.ModuleList([BertLayer(config) for _ in range(self.num_blocks)])
        self.pooler = BertPooler(config)

    def forward(self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        ):           
        x_enc = self.embedding(input_ids, position_ids)
        for encoder_layer in self.encoder:
            x_enc = encoder_layer(x_enc, attention_mask)

        sequence_output = x_enc
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BertOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output
        )
