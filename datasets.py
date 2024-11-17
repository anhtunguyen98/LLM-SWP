import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
'''since we have three validation tasks -- three datasets,
better to handle it separately from dataloader'''


# helper functions

def tokenize(texts, max_length=512) -> dict:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # add [CLS] and [SEP] tokens to the beginning and end of each document
    # set max_length to truncate long documents


    """
    Tokenizes input texts and returns token IDs and attention masks.

    Args:
        texts (list of str): A list of input sentences or documents.
        max_length (int): Maximum sequence length (default is 128).

    Returns:
        dict: A dictionary containing:
            - 'input_ids': Padded token IDs.
            - 'attention_mask': Attention masks.
            
    """
    encoding = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"  # tensors
    )

    # Return the tokenized results
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"]
    }



class FillInBlanksDataset(Dataset):

    pass

class TextClassificationDataset(Dataset):
    pass

class MultipleChoiceQuestionDataset(Dataset):
    pass