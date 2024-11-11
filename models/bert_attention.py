import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):

    def __init__(self,config):

        super().__init__()
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.query_weights = nn.Linear(config.hidden_size, self.attention_head_size)
        self.key_weights = nn.Linear(config.hidden_size, self.attention_head_size)
        self.value_weights = nn.Linear(config.hidden_size, self.attention_head_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor = None
                ):

        query = self.query_weights(query)
        key = self.key_weights(key)
        value = self.value_weights(value)

        att_scores = torch.matmul(query, key.transpose(1, 2)) / self.attention_head_size ** 0.5

        if mask is not None:
            mask = mask.to(torch.int)
            att_scores = att_scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        att_weights = F.softmax(att_scores, dim=-1)
        att_weights = self.dropout(att_weights) 
        n_value = torch.matmul(att_weights, value)

        return n_value

class MultiHeadAttention(nn.Module):

    def __init__(self, config):

        super().__init__()
        self.num_heads = config.num_attention_heads
        self.attention_heads: nn.ModuleList = nn.ModuleList([Attention(config) for _ in range(self.num_heads)])
        self.fc: nn.Linear = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor = None,
                ):

        attention_outputs = [attention_head(query, key, value, mask=mask) for attention_head in self.attention_heads]
        hidden_state = torch.cat(attention_outputs, dim=-1)
        hidden_state = self.fc(hidden_state)
        return hidden_state