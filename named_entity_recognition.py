from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from models.bert_model import BertForMaskedLM, BertModel, BertForTokenClassification
from transformers import  PreTrainedTokenizerFast
from torch.utils.data import Dataset
import torch
import json
import numpy as np
from sklearn.model_selection import train_test_split
import evaluate
from datasets import load_dataset
from transformers import DataCollatorForTokenClassification

max_length = 512
model_name_or_path = 'checkpoints/checkpoint-220000'
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name_or_path)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
seqeval = evaluate.load("seqeval")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], padding='max_length', max_length=max_length, truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions[0] if isinstance(predictions, tuple) else predictions
    predictions = np.argmax(predictions, axis=2)


    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]


    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


if __name__ == '__main__':


    dataset_name = 'spyysalo/bc2gm_corpus'
    raw_datasets = load_dataset(dataset_name)
    raw_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=20,
        desc="Running tokenizer on dataset",
    )
    
    label_list = raw_datasets["train"].features[f"ner_tags"].feature.names

    train_dataset = raw_datasets["train"]
    train_val_dataset = train_dataset.train_test_split(test_size=0.1)
    
    train_dataset = train_val_dataset['train']
    val_dataset = train_val_dataset['test']
    test_dataset = raw_datasets["test"]

    model = BertForTokenClassification.from_pretrained(model_name_or_path,num_labels = 3)

    # frozen bert encoder
    for param in model.bert.parameters():
        param.requires_grad = False


    training_args = TrainingArguments(
        output_dir='./ner_classification',          # output directory
        num_train_epochs=10,               # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=8,   # batch size for evaluation
        warmup_steps=0,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./ner_classification/runs',
        report_to='tensorboard',            # directory for storing logs
        load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
        metric_for_best_model='f1',
        # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
        logging_steps=100,               # log & save weights each logging_steps
        save_strategy="epoch",
        evaluation_strategy="epoch",     # evaluate each `logging_steps`
    )

    trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,            # evaluation dataset
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
    )
    trainer.train()
    print(trainer.evaluate(eval_dataset=test_dataset))
