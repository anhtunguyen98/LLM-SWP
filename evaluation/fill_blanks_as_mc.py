import torch
from models.bert_model import BertForMaskedLM, BertModel, BertForMultipleChoice
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from transformers import PreTrainedTokenizerFast
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import pandas as pd, numpy as np, torch
from dataclasses import dataclass
from datasets import load_dataset

max_length = 512
model_name_or_path = 'checkpoints/checkpoint-220000'
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name_or_path)


def preprocess_function(example):
    abstracts = example['Abstract']
    missing_words = [example[option] for option in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']]

    processed_abstracts = []
    for i in range(10):
        processed_abstracts.append(abstracts.replace("[MASK]", missing_words[i]))

    tokenized_example = tokenizer(processed_abstracts, truncation=True,
                                  max_length=max_length, add_special_tokens=False)

    tokenized_example['label'] = int(example['cop'])-1

    return tokenized_example


def acc(predictions, labels):
    count = 0
    pred = np.argsort(-1 * np.array(predictions), axis=1)[:, 0]
    for x, y in zip(pred, labels):
        if x == y:
            count += 1
    return count / len(predictions)


def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    predictions = preds.tolist()
    labels = p.label_ids.tolist()
    return {"acc": acc(predictions, labels)}


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch


if __name__ == '__main__':

    dataset = load_dataset("csv", data_files={"train": "fill_in_the_blanks.csv"})
    raw_datasets = dataset.map(
        preprocess_function,
        num_proc=4,
        desc="Running tokenizer on dataset",
    )

    # Split dataset
    train_test_split = raw_datasets["train"].train_test_split(test_size=0.2)
    train_val_split = train_test_split['train'].train_test_split(test_size=0.1)
    train_dataset = train_val_split['train']
    val_dataset = train_val_split['test']
    test_dataset = train_test_split['test']

    model = BertForMultipleChoice.from_pretrained(model_name_or_path)

    # frozen bert encoder
    for param in model.bert.parameters():
        param.requires_grad = False

    training_args = TrainingArguments(
        output_dir='./fb_checkpoint',  # output directory
        num_train_epochs=10,  # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=8,  # batch size for evaluation
        warmup_steps=0,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./fb_checkpoint/runs',
        report_to='tensorboard',  # directory for storing logs
        load_best_model_at_end=True,  # load the best model when finished training (default metric is loss)
        metric_for_best_model='acc',
        # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
        logging_steps=10,  # log & save weights each logging_steps
        save_strategy="epoch",
        evaluation_strategy="epoch",  # evaluate each `logging_steps`
    )

    trainer = Trainer(
        model=model,  # the instantiated Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        compute_metrics=compute_metrics,  # the callback that computes metrics of interest
    )

    print(trainer.evaluate(eval_dataset=test_dataset))
    trainer.train()
    print(trainer.evaluate(eval_dataset=test_dataset))
