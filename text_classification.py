from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from models.bert_model import BertForMaskedLM, BertModel, BertForSequenceClassification
from transformers import  PreTrainedTokenizerFast
from datasets import load_dataset


max_length = 256
model_name_or_path = 'checkpoints/checkpoint-220000'
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name_or_path)


def preprocess_function(examples):

    result = tokenizer(examples["medical_abstract"], padding='max_length', max_length=max_length, truncation=True)

    result["label"] = [l-1 for l in examples["condition_label"]] 
    return result


def compute_metrics(pred):
    preds = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    labels = pred.label_ids
    preds = preds.argmax(-1)
    f1 = f1_score(labels, preds, average = 'micro')
    acc = accuracy_score(labels, preds)
    return {
        'f1': f1,
        'acc': acc
    }



if __name__ == '__main__':



    dataset_name = 'TimSchopf/medical_abstracts'
    raw_datasets = load_dataset(dataset_name)
    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=20,
        desc="Running tokenizer on dataset",
    )
    train_dataset = raw_datasets["train"]
    train_val_dataset = train_dataset.train_test_split(test_size=0.1)
    
    train_dataset = train_val_dataset['train']
    val_dataset = train_val_dataset['test']
    test_dataset = raw_datasets["test"]


    
    model = BertForSequenceClassification.from_pretrained(model_name_or_path,num_labels = 5)

    # frozen bert encoder
    for param in model.bert.parameters():
        param.requires_grad = False


    

    training_args = TrainingArguments(
        output_dir='./classification_checkpoint',          # output directory
        num_train_epochs=10,               # total number of training epochs
        per_device_train_batch_size=64,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=0,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',
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
    eval_dataset=val_dataset,          # evaluation dataset
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
    )
    trainer.train()
    print(trainer.evaluate(eval_dataset=test_dataset))

