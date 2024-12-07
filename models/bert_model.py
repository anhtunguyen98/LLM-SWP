import os
from functools import partial
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_utils import unwrap_model
from safetensors.torch import load_file as safe_load_file
from models.bert_config import BertConfig
from typing import List, Optional, Tuple, Union
from models.bert_output import BertOutput, MaskedLMOutput
from models.bert_layer import BertLayer, BertPooler, BertOnlyMLMHead
from models.bert_embedding import BertEmbeddings

class BaseModel(nn.Module):
    """Base model class providing common functionalities for model loading and saving."""
    
    config_class = None  # This should be defined in subclasses

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, num_labels=None):
        """Load a pre-trained model from a specified path."""
        config = cls.config_class.from_pretrained(pretrained_model_name_or_path)
        if num_labels:
            config.num_labels = num_labels
        model = cls(config)

        checkpoint_path = os.path.join(pretrained_model_name_or_path, 'pytorch_model.bin')
        if not os.path.isfile(checkpoint_path):
            checkpoint_path = os.path.join(pretrained_model_name_or_path, 'model.safetensors')
            loader = safe_load_file
        else:
            loader = partial(torch.load, map_location='cpu')

        state_dict = loader(checkpoint_path)
        model.load_state_dict(state_dict, strict=False)

        return model

    def save_pretrained(self, save_directory):
        """Save the model and its configuration to the specified directory."""
        os.makedirs(save_directory, exist_ok=True)
        model_to_save = unwrap_model(self)
        model_to_save.config.save_pretrained(save_directory)

        output_model_file = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(model_to_save.state_dict(), output_model_file)


class BertModel(BaseModel):
    """BERT model consisting of an embedding layer and a stack of transformer layers."""
    
    config_class = BertConfig  # Define the config class for BERT

    def __init__(self, config, add_pooling_layer=False):
        super(BertModel, self).__init__()
        self.config = config
        self.num_blocks = config.num_hidden_layers
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embedding = BertEmbeddings(config)
        self.encoder = nn.ModuleList([BertLayer(config) for _ in range(self.num_blocks)])
        self.pooler = BertPooler(config) if add_pooling_layer else None

    def forward(self, input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> BertOutput:
        """Forward pass through the BERT model."""
        x_enc = self.embedding(input_ids, position_ids)
        for encoder_layer in self.encoder:
            x_enc = encoder_layer(x_enc, attention_mask)

        sequence_output = x_enc
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BertOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output
        )


class BertForMaskedLM(BaseModel):
    """BERT model for Masked Language Modeling (MLM)."""
    
    config_class = BertConfig  # Define the config class for MLM

    def __init__(self, config):
        super(BertForMaskedLM, self).__init__()
        self.config = config
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

    def forward(self, input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                return_dict: Optional[bool] = None) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        """Forward pass for MLM with optional loss computation."""
        outputs = self.bert(input_ids, attention_mask=attention_mask, position_ids=position_ids)

        sequence_output = outputs.last_hidden_state
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
        ) if return_dict else (masked_lm_loss, prediction_scores, outputs.last_hidden_state)


class BertForSequenceClassification(BaseModel):

    config_class = BertConfig

    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs = self.bert(input_ids, attention_mask=attention_mask, position_ids=position_ids)

        pooled_output = outputs.last_hidden_state[:,0,:]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits, pooled_output)
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
class BertForMultipleChoice(BaseModel):
    config_class = BertConfig  # Define config class for Multiple Choice

    def __init__(self, config):
        super(BertForMultipleChoice, self).__init__()
        self.config = config
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        pooled_output = outputs.last_hidden_state[:, 0, :]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits, pooled_output)
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
        )
