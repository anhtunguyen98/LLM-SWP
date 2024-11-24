from models.bert_config import BertConfig
from models.bert_model import BertForMaskedLM
import torch
import evaluate
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast
from tokenizers import Tokenizer
import argparse
from datasets import load_dataset
from transformers import TrainingArguments, Trainer

metric = evaluate.load("accuracy")

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return metric.compute(predictions=preds, references=labels)


def train(training_args):

    # init model and data

    tokenizer = Tokenizer.from_file(training_args.tokenizer_path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                pad_token='[PAD]',
                unk_token='[UNK]',
                mask_token='[MASK]',
                sep_token='[SEP]',
                cls_token='[CLS]')
    model_config = BertConfig(
        vocab_size=tokenizer.vocab_size
    )
    model = BertForMaskedLM(model_config)

    # load dataset

    data_files = {}
    data_files["train"] = training_args.train_file
    extension = training_args.train_file.split(".")[-1]

    if training_args.validation_file is not None:
        data_files["validation"] = training_args.validation_file
        extension = training_args.validation_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
    )

    #preprocess dataset

    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    max_seq_length = training_args.max_seq_length
    

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column_name],
            padding='max_length',
            truncation=True,
            max_length=max_seq_length,
            return_special_tokens_mask=True,
        )
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=[text_column_name],
        num_proc=training_args.preprocessing_num_workers,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=training_args.mlm_probability,
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]


    args = TrainingArguments(
                        warmup_ratio=training_args.warmup_ratio, 
                        learning_rate=training_args.learning_rate,
                        per_device_train_batch_size=training_args.per_device_train_batch_size,
                        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
                        num_train_epochs=training_args.num_train_epochs,
                        report_to='tensorboard',
                        output_dir = training_args.output_dir,
                        overwrite_output_dir=True,
                        fp16=False,
                        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
                        logging_steps=training_args.logging_steps,
                        evaluation_strategy='steps',
                        eval_steps=training_args.eval_steps,
                        save_strategy="steps",
                        save_steps=training_args.save_steps,
                        load_best_model_at_end=True,
                        lr_scheduler_type='cosine',
                        weight_decay=0.01,
                        save_total_limit=10,
            )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
    )
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training arguments")
    #dataset
    parser.add_argument("--tokenizer_path", type=str, default='assets/tokenizer.json')
    parser.add_argument("--train_file", type=str, default='data/train.txt')
    parser.add_argument("--validation_file", type=str, default='data/val.txt')
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--preprocessing_num_workers", type=int, default=20)
    parser.add_argument("--mlm_probability", type=float, default=0.15)

    #hyperparams
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default='checkpoints/')
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = parser.parse_args()

    train(args)
    


