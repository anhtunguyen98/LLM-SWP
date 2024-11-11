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