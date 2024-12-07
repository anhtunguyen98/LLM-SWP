import torch
from torch.utils.data import Subset
from torch.utils.data import Dataset
from models.bert_model import BertForMaskedLM, BertModel, BertForMultipleChoice
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from transformers import PreTrainedTokenizerFast




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

        encodings = []
        for option in options:
            text = f"{context} {question} {option}" if self.use_context else f"{question} {option}"
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            encodings.append(encoding)

        input_ids = torch.stack([e['input_ids'].squeeze(0) for e in encodings])  # [4, max_length]
        attention_mask = torch.stack([e['attention_mask'].squeeze(0) for e in encodings])  # [4, max_length]

        # Rückgabe als Dictionary
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long)
        }

max_length = 256
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

    dataset_name = 'medmcqa_data.csv'
    dataset = MCQADataset(csv_path=dataset_name, tokenizer=tokenizer, use_context=False)

    indices = list(range(len(dataset)))
    train_indices, temp_indices = train_test_split(indices, test_size=0.2, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    model = BertForMultipleChoice.from_pretrained(model_name_or_path, num_labels=5)

    # frozen bert encoder
    for param in model.bert.parameters():
        param.requires_grad = False

    training_args = TrainingArguments(
        output_dir='./classification_checkpoint',  # output directory
        num_train_epochs=10,  # total number of training epochs
        per_device_train_batch_size=64,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=0,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',
        report_to='tensorboard',  # directory for storing logs
        load_best_model_at_end=True,  # load the best model when finished training (default metric is loss)
        metric_for_best_model='f1',
        # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
        logging_steps=100,  # log & save weights each logging_steps
        save_strategy="epoch",
        evaluation_strategy="epoch",# evaluate each `logging_steps`
    )

    trainer = Trainer(
        model=model,  # the instantiated Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # the callback that computes metrics of interest
    )
    trainer.train()
    print(trainer.evaluate(eval_dataset=test_dataset))