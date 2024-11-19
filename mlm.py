from models.bert_config import BertConfig
from models.bert_model import BertForMaskedLM
import torch
import evaluate
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast
from tokenizers import Tokenizer

metric = evaluate.load("accuracy")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    preds = preds.argmax(dim=-1)
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return metric.compute(predictions=preds, references=labels)


def train(training_config, data_config):

    # init model and data

    tokenizer = Tokenizer.from_file(training_config.tokenizer_path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,pad_token='[STOP]',unk_token='[UNK]',)
    model_config = BertConfig(
        vocab_size=tokenizer.vocab_size
    )
    model = BertForMaskedLM(model_config)

    # load dataset

    data_files = {}
    data_files["train"] = data_config.train_file
    extension = data_config.train_file.split(".")[-1]

    if data_config.validation_file is not None:
        data_files["validation"] = data_config.validation_file
        extension = data_config.validation_file.split(".")[-1]
    
    
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=training_config.cache_dir,
        token=training_config.token,
    )

    #preprocess dataset

    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    max_seq_length = data_config.max_seq_length
    

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
            return_special_tokens_mask=True,
        )
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=[text_column_name],
        num_proc=data_config.preprocessing_num_workers,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=data_config.mlm_probability,
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]


    training_args = TrainingArguments(
                        warmup_ratio=0.1, 
                        learning_rate=2e-5,
                        per_device_train_batch_size=8,
                        per_device_eval_batch_size=8,
                        num_train_epochs=2,
                        report_to='wandb',
                        output_dir = OUTPUT_DIR,
                        overwrite_output_dir=True,
                        fp16=True,
                        gradient_accumulation_steps=8,
                        logging_steps=25,
                        evaluation_strategy='steps',
                        eval_steps=500,
                        save_strategy="steps",
                        save_steps=500,
                        load_best_model_at_end=True,
                        metric_for_best_model='acc',
                        lr_scheduler_type='cosine',
                        weight_decay=0.01,
                        save_total_limit=2,
                    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    train_result = trainer.train(resume_from_checkpoint=checkpoint)


