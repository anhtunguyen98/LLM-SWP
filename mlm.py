from models.bert_config import BertConfig
from models.bert_model import BertModel
import torch

config = BertConfig()
model = BertModel(config)

input_ids = torch.tensor([[0,1,2,3,4]])
attention_mask = torch.tensor([[1,1,1,1,1]])
print(model)
output = model(input_ids,attention_mask)
print(output.last_hidden_state.shape)