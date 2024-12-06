import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, BertForMultipleChoice, BertTokenizer
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
import pandas as pd

class MCQADataset(Dataset):

    def __init__(self, csv_path, tokenizer, max_length=512, use_context=True):
        self.dataset = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_context = use_context

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.use_context:
            context = self.dataset.loc[idx, 'exp']
        question = self.dataset.loc[idx, 'question']
        options = self.dataset.loc[idx, ['opa', 'opb', 'opc', 'opd']].values
        label = self.dataset.loc[idx, 'cop'] - 1

        # Tokenize
        encodings = []
        for option in options:
            if self.use_context:
                # Combine context, question and option
                text = f"{context} {question} {option}"
            else:
                # Combine question and option only
                text = f"{question} {option}"

            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            encodings.append(encoding)

        # Stack the different encodings into a batch (in this case with 4 options)
        input_ids = torch.cat([e['input_ids'] for e in encodings], dim=0)
        attention_mask = torch.cat([e['attention_mask'] for e in encodings], dim=0)

        return input_ids, attention_mask, label

# Load BERT-Tokenizer and pretrained model
tokenizer = PreTrainedTokenizerFast.from_pretrained('checkpoints/checkpoint-220000')
model = BertForMultipleChoice.from_pretrained('checkpoints/checkpoint-220000')

# Set the model to evaluation mode
model.eval()

# Load our dataset
dataset = MCQADataset(csv_path='medmcqa_data.csv', tokenizer=tokenizer, use_context=True)

# DataLoader
data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

def evaluate(model, data_loader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            input_ids = input_ids.cuda() if torch.cuda.is_available() else input_ids
            attention_mask = attention_mask.cuda() if torch.cuda.is_available() else attention_mask
            labels = labels.cuda() if torch.cuda.is_available() else labels

            # Forward Pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Predictions
            preds = torch.argmax(logits, dim=1)

            # Save predictions
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

accuracy = evaluate(model, data_loader)
print(f'Accuracy: {accuracy * 100:.2f}%')



