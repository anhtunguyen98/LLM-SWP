from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from datasets import load_dataset
from medmcqa import MCQADataset
import torch
import pandas as pd
import random

class Tokenizer:
    """
    A class to handle tokenization and preparation of MedMCQA and MedWiki datasets.
    """

    def __init__(self, medmcqa_dataset: object, medwiki_dataset: str):
        """
        Initializes the Tokenizer with MedMCQA and MedWiki datasets.

        Args:
            medmcqa_dataset (object): Instance of the MedMCQA dataset class.
            medwiki_dataset_name (str): Name of the MedWiki dataset (e.g., 'medwiki.py').
            medwiki_split (str): The data split to load from MedWiki (default is "train").
        """
        self.medmcqa_dataset = medmcqa_dataset

        self.medwiki_dataset = load_dataset(medwiki_dataset)

    def tokenize_medmcqa(self):
        """
        Tokenizes the MedMCQA dataset into a unified text format.

        Returns:
            List[Dict]: A list of dictionaries with tokenized MedMCQA entries.
        """
        tokenized_data = []
        for idx in range(len(self.medmcqa_data)):
            # Extract entries from MedMCQA
            data = self.medmcqa_data[idx]
            if len(data) == 3:  # Case where context is not used
                context, question, options, label = None, *data
            else:  # Case where context is used
                context, question, options, label = data

            # Build tokenized text
            qa_text = "<qa>"
            if context:
                qa_text += f" <context> {context}"
            qa_text += (
                f" <question> {question} <options> {'; '.join(options)} <answer> {options[label]}"
            )
            tokenized_data.append({"text": qa_text})
        return tokenized_data

    def tokenize_medwiki(self):
        """
        Tokenizes the MedWiki dataset into a unified text format.

        Returns:
            List[Dict]: A list of dictionaries with tokenized MedWiki entries.
        """
        tokenized_data = []
        for idx in range(len(self.medwiki_dataset)):
            # Extract entries from MedWiki
            data = self.medwiki_dataset[idx]
            title, content = data["document.title"], data["document.text"]

            # Build tokenized text
            wiki_text = f"<wiki> <title> {title.strip()} <content> {content.strip()}"
            tokenized_data.append({"text": wiki_text})
        return tokenized_data

    def combine_datasets(self):
        """
        Combines and shuffles tokenized MedMCQA and MedWiki datasets.

        Returns:
            List[Dict]: A shuffled list of tokenized entries from both datasets.
        """
        # Tokenize individual datasets
        medmcqa_data = self.tokenize_medmcqa()
        medwiki_data = self.tokenize_medwiki()

        # Combine the datasets
        combined_data = medmcqa_data + medwiki_data

        # Shuffle the combined dataset for multitask training
        random.shuffle(combined_data)
        return combined_data

class CombinedDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        """
        Custom dataset for combining MedMCQA and MedWiki data.

        Args:
            dataframe (pd.DataFrame): Combined DataFrame with MedMCQA and MedWiki data.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer for encoding the text.
            max_length (int): The maximum sequence length for tokenization.
        """
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Retrieves the tokenized version of the text and the associated label for the task.
        """
        row = self.dataframe.iloc[idx]
        text = row['text']
        label = row['label']  # Assuming you have a label column in combined_df

        # Tokenize the input text
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")

        # Extract the tokenized input IDs and attention mask
        input_ids = encoding['input_ids'].squeeze()  # Shape: [max_length]
        attention_mask = encoding['attention_mask'].squeeze()  # Shape: [max_length]

        # Return tokenized input and label
        return input_ids, attention_mask, label


tokenizer = AutoTokenizer.from_pretrained("gpt2")

medmcqa_dataset = MCQADataset(csv_path="train.csv")

medwiki_dataset = "medwiki.py"
combined_df = Tokenizer(medmcqa_dataset=medmcqa_dataset, medwiki_dataset=medwiki_dataset).combine_datasets()

dataset = CombinedDataset(dataframe=combined_df, tokenizer=tokenizer)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for input_ids, attention_mask, labels in dataloader:
    print(input_ids.shape)  # Shape will be [batch_size, max_length]
    print(attention_mask.shape)  # Shape will be [batch_size, max_length]
    print(labels.shape)  # Shape will be [batch_size]