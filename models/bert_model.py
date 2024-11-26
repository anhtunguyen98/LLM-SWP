import torch
import torch.nn as nn
from models.bert_layer import BertLayer, BertPooler, BertOnlyMLMHead
from models.bert_embedding import BertEmbeddings
from models.bert_config import BertConfig
from typing import List, Optional, Tuple, Union
from models.bert_output import BertOutput, MaskedLMOutput
from transformers.modeling_utils import unwrap_model
import os

class BertModel(nn.Module):

    config_class = BertConfig

    def __init__(self, config, add_pooling_layer=False):
        super(BertModel, self).__init__()
        self.config = config

        self.num_blocks = config.num_hidden_layers
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embedding = BertEmbeddings(config)
        self.encoder = nn.ModuleList([BertLayer(config) for _ in range(self.num_blocks)])

        self.pooler = BertPooler(config) if add_pooling_layer else None
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):

        config = cls.config_class.from_pretrained(pretrained_model_name_or_path)
        model = cls(config)
        checkpoint_path = os.path.join(pretrained_model_name_or_path,'pytorch_model.bin')
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

        return model


    def save_pretrained(self,save_directory):

        os.makedirs(save_directory, exist_ok=True)
        model_to_save = unwrap_model(self)
        model_to_save.config.save_pretrained(save_directory)

        output_model_file = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(model_to_save.state_dict(), output_model_file)

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
class BertForMaskedLM(nn.Module):

    config_class = BertConfig

    def __init__(self, config):
        super(BertForMaskedLM, self).__init__()
        self.config = config

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):

        config = cls.config_class.from_pretrained(pretrained_model_name_or_path)
        model = cls(config)
        checkpoint_path = os.path.join(pretrained_model_name_or_path,'pytorch_model.bin')
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

        return model
    
    def save_pretrained(self,save_directory):

        os.makedirs(save_directory, exist_ok=True)
        model_to_save = unwrap_model(self)
        model_to_save.config.save_pretrained(save_directory)

        output_model_file = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(model_to_save.state_dict(), output_model_file)    

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        sequence_output = outputs.last_hidden_state
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss() 
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + (outputs.last_hidden_state,)
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
        )

