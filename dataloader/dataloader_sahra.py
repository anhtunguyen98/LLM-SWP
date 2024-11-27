from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
import os
from pubmed import PubMedProcessor

class Tokenizer:
    """
    A class to handle tokenization and preparation of PubMed and MedWiki datasets.
    """

    def __init__(self, pubmed_processor, medwiki_dataset: str, pubmed_output_csv: str):
        """
        Initializes the Tokenizer with PubMedProcessor and MedWiki datasets.

        Args:
            pubmed_processor (PubMedProcessor): An instance of the PubMedProcessor class.
            medwiki_dataset (str): Name of the MedWiki script from huggingface (e.g., 'medwiki.py').
            pubmed_output_csv (str): Path to save the generated PubMed CSV file.
        """
        self.pubmed_processor = pubmed_processor
        self.medwiki_dataset = load_dataset(medwiki_dataset)  # Load the MedWiki dataset
        self.pubmed_output_csv = pubmed_output_csv

        # Check if the PubMed CSV already exists
        if not os.path.exists(pubmed_output_csv):
            print(f"Generating PubMed CSV file at {pubmed_output_csv}...")
            self.pubmed_processor.process_xml_files()  # Generate CSV from XML files
        else:
            print(f"PubMed CSV file already exists at {pubmed_output_csv}.")

        # Load the generated PubMed CSV into a pandas DataFrame
        self.pubmed_data = pd.read_csv(pubmed_output_csv)

    def tokenize_pubmed(self):
        """
        Tokenizes the PubMed dataset into a unified text format.

        Returns:
            List[Dict]: A list of dictionaries with tokenized PubMed entries.
        """
        tokenized_data = []
        for idx in range(len(self.pubmed_data)):
            # Extract entries from the PubMed dataset (assuming columns 'title', 'abstract', and 'text')
            title = self.pubmed_data.iloc[idx]['title']
            abstract = self.pubmed_data.iloc[idx]['abstract']
            full_text = self.pubmed_data.iloc[idx]['text']

            # Build tokenized text in a specific format
            pubmed_text = f"<pubmed> <title> {title.strip()} <abstract> {abstract.strip()} <text> {full_text.strip()}"
            tokenized_data.append({"text": pubmed_text})
        return tokenized_data

    def tokenize_medwiki(self):
        """
        Tokenizes the MedWiki dataset into a unified text format.

        Returns:
            List[Dict]: A list of dictionaries with tokenized MedWiki entries.
        """
        tokenized_data = []
        for idx in range(len(self.medwiki_dataset['train'])):  # Use the 'train' split; adjust if needed
            data = self.medwiki_dataset['train'][idx]
            title, content = data["document.title"], data["document.text"]

            # Build tokenized text in a specific format
            wiki_text = f"<wiki> <title> {title.strip()} <content> {content.strip()}"
            tokenized_data.append({"text": wiki_text})
        return tokenized_data

    def combine_datasets(self):
        """
        Combines and shuffles tokenized PubMed and MedWiki datasets.

        Returns:
            List[Dict]: A shuffled list of tokenized entries from both datasets.
        """
        pubmed_data = self.tokenize_pubmed()  # Tokenize PubMed data
        medwiki_data = self.tokenize_medwiki()  # Tokenize MedWiki data

        combined_data = pubmed_data + medwiki_data  # Combine both datasets
        random.shuffle(combined_data)  # Shuffle for better training distribution
        return combined_data


class CombinedDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        """
        Custom dataset for combining PubMed and MedWiki data.

        Args:
            dataframe (pd.DataFrame): Combined DataFrame with PubMed and MedWiki data.
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
        Retrieves the tokenized version of the text and associated label for the task.

        Returns:
            Tuple[Tensor, Tensor]: Tokenized input IDs and attention mask.
        """
        row = self.dataframe.iloc[idx]
        text = row['text']

        # Tokenize the input text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding['input_ids'].squeeze()  # Convert to tensor format
        attention_mask = encoding['attention_mask'].squeeze()  # Convert to tensor format

        return input_ids, attention_mask


# Example Usage:

# Initialize the PubMedProcessor to process XML files and generate a CSV
pubmed_processor = PubMedProcessor()

# Initialize the Tokenizer
tokenizer_instance = Tokenizer(
    pubmed_processor=pubmed_processor,
    medwiki_dataset="medwiki.py",  # Name of the MedWiki script
    pubmed_output_csv="pubmed.csv",  # Generated CSV from PubMed
)

# Combine PubMed and MedWiki datasets
combined_data = tokenizer_instance.combine_datasets()

# Create a DataFrame for combined data
combined_df = pd.DataFrame(combined_data)

# Initialize GPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Create a PyTorch Dataset
dataset = CombinedDataset(dataframe=combined_df, tokenizer=tokenizer)

# Initialize DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate over the DataLoader
for input_ids, attention_mask in dataloader:
    print(input_ids.shape)  # Shape: [batch_size, max_length]
    print(attention_mask.shape)  # Shape: [batch_size, max_length]
