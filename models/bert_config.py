import copy
import json
import os

class BertConfig:

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        pad_token_id=0,
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
    ):

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.classifier_dropout = classifier_dropout
        self.pad_token_id = pad_token_id
    
    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output
    
    def to_json_string(self):
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def __repr__(self):
        return self.to_json_string()

    @classmethod
    def from_pretrained(cls,pretrained_model_name_or_path):
        config_file_path = os.path.join(pretrained_model_name_or_path,'config.json')
        with open(config_file_path, "r", encoding="utf-8") as reader:
            text = reader.read()
        config_dict = json.loads(text)
        config = cls(**config_dict)

        return config



    
    def save_pretrained(self,save_directory):

        config_dict = self.to_dict()
        os.makedirs(save_directory, exist_ok=True)
        
        json_file_path = os.path.join(save_directory,'config.json')
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(config_dict, indent=2, sort_keys=True) + "\n")