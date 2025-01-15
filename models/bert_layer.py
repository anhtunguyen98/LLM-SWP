import torch
import torch.nn as nn
from models.bert_attention import MultiHeadAttention
import torch.nn.functional as F


class FeedForward(nn.Module):


    def __init__(self, config):

        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_fc_size = self.hidden_size * 4
        self.hidden_dropout_prob = config.hidden_dropout_prob

        self.fc1 = nn.Linear(self.hidden_size, self.intermediate_fc_size)
        self.fc2 = nn.Linear(self.intermediate_fc_size, self.hidden_size)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self,
        hidden_state: torch.Tensor):

        hidden_state = self.fc1(hidden_state)
        hidden_state = F.gelu(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state= self.fc2(hidden_state)
        
        return hidden_state


class BertLayer(nn.Module):


    def __init__(self, config):

        super().__init__()

        self.hidden_size = config.hidden_size
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.multihead_attention = MultiHeadAttention(config)
        self.norm1 = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.feed_forward = FeedForward(config)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self,
        hidden_state: torch.Tensor,
        mask: torch.Tensor = None,
        ):
        x_norm1 = self.norm1(hidden_state)
        attention_output = self.multihead_attention(x_norm1, x_norm1, x_norm1, mask)
        hidden_state = attention_output + hidden_state
        
        x_norm2 = self.norm2(hidden_state)
        feed_forward_output = self.feed_forward(x_norm2)
        x_enc = feed_forward_output + hidden_state
        hidden_state = self.dropout(x_enc)
        
        return hidden_state

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.predictions = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        sequence_output = self.dense(sequence_output)
        sequence_output = F.gelu(sequence_output)
        sequence_output = self.LayerNorm(sequence_output)
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score
