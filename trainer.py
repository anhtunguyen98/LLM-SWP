from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from models.bert_model import BertForMaskedLM, BertModel
from models.bert_config import BertConfig


tokenizer = PreTrainedTokenizerFast.from_pretrained('checkpoints/checkpoint-220000')

model = BertModel.from_pretrained('checkpoints/checkpoint-220000')

print(model)
