from typing import List, Optional, Tuple, Union
import torch

class BertOutput:
    def __init__(self,
                last_hidden_state: torch.FloatTensor = None,
                pooler_output: torch.FloatTensor = None,
                hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None):
                
        self.last_hidden_state = last_hidden_state
        self.pooler_output = pooler_output
        self.hidden_states = hidden_states

class MaskedLMOutput:
    def __init__(self,
                loss: torch.FloatTensor = None,
                logits: torch.FloatTensor = None,
                hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None):
                
        self.loss = loss
        self.logits = logits
        self.hidden_states = hidden_states