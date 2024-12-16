import torch
from torch.utils.data import Subset, Dataset
from models.bert_model import BertForMultipleChoice
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from transformers import PreTrainedTokenizerFast

class BlanksDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=512, use_context=True):
        self.dataset = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_context = use_context

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.use_context:
            context = self.dataset.loc[idx, 'Abstract']
        options = self.dataset.loc[idx, ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']].values
        label = self.dataset.loc[idx, 'cop'] - 1

        encodings = []
        for option in options:
            text = context.replace("[MASK]", option)
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            encodings.append(encoding)

        input_ids = torch.stack([e['input_ids'].squeeze(0) for e in encodings])
        attention_mask = torch.stack([e['attention_mask'].squeeze(0) for e in encodings])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(int(label), dtype=torch.long)
        }

max_length = 512
model_name_or_path = 'checkpoints/checkpoint-220000'
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name_or_path)

def compute_metrics(pred):
    preds = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    labels = pred.label_ids
    preds = preds.argmax(-1)
    f1 = f1_score(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'f1': f1,
        'acc': acc
    }

if __name__ == '__main__':
    dataset_name = 'updated_file.csv'
    dataset = BlanksDataset(csv_path=dataset_name, tokenizer=tokenizer, use_context=True)

    indices = list(range(len(dataset)))
    train_indices, temp_indices = train_test_split(indices, test_size=0.2, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    model = BertForMultipleChoice.from_pretrained(model_name_or_path, num_labels=10)

    for param in model.bert.parameters():
        param.requires_grad = False

    training_args = TrainingArguments(
        output_dir='./classification_checkpoint',
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=0,
        weight_decay=0.01,
        logging_dir='./logs',
        report_to='tensorboard',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        logging_steps=5,
        save_strategy="epoch",
        evaluation_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print(trainer.evaluate(eval_dataset=test_dataset))
