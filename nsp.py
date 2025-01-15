from models.bert_config import BertConfig
from models.bert_model import BertForMaskedLM, BertForNextSentencePrediction
import torch
import evaluate
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast
from tokenizers import Tokenizer
import argparse
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from dataloader.nsp_dataset import TextDatasetForNextSentencePrediction, DataCollatorWithPaddingAndTracating




metric = evaluate.load("accuracy")

def compute_metrics(pred):
    preds = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    labels = pred.label_ids
    preds = preds.argmax(-1)
    return metric.compute(predictions=preds, references=labels)


def train(training_args):

    # init model and data

    tokenizer = PreTrainedTokenizerFast.from_pretrained(training_args.model_name_or_path)
    model = BertForNextSentencePrediction.from_pretrained(training_args.model_name_or_path)


    train_dataset = TextDatasetForNextSentencePrediction(file_path=training_args.train_file,
                                                        tokenizer=tokenizer,
                                                        block_size=200)
    
    eval_dataset = TextDatasetForNextSentencePrediction(file_path=training_args.validation_file,
                                                        tokenizer=tokenizer,
                                                        block_size=200)
    


    data_collator = DataCollatorWithPaddingAndTracating(
        tokenizer=tokenizer,
        max_length=training_args.max_seq_length,
        padding='max_length'
    )


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
        compute_metrics=compute_metrics,
    )
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training arguments")
    #dataset
    parser.add_argument("--model_name_or_path", type=str, default='checkpoints/checkpoint-280000')
    
    parser.add_argument("--train_file", type=str, default='data/train_nsp.txt')
    parser.add_argument("--validation_file", type=str, default='data/val_nsp.txt')
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--preprocessing_num_workers", type=int, default=20)
    parser.add_argument("--mlm_probability", type=float, default=0.15)

    #hyperparams
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default='checkpoints_nsp/')
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=20000)
    parser.add_argument("--save_steps", type=int, default=20000)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = parser.parse_args()

    train(args)
    


