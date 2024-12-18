import torch
from models.bert_model import BertModel, BertForMaskedLM
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

# Load the BERT model and tokenizer
model_name_or_path = "checkpoints/checkpoint-220000"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name_or_path)  # Load tokenizer
model = BertForMaskedLM.from_pretrained(model_name_or_path)  # Load masked language model
model.eval()  # Set the model to evaluation mode

# Load the MeDAL dataset
dataset_name = "mcgill-nlp/medal"
raw_datasets = load_dataset(dataset_name, split="test[:5000]")  # Use a subset of the test split for inference

# Function to preprocess a dataset
def preprocess_function(examples):
    # Tokenize text with truncation and padding to a maximum length of 512 tokens
    tokenized = tokenizer(examples['text'], padding='max_length', max_length=512, truncation=True, return_tensors="pt")

    # Access tokenized input IDs
    input_ids = torch.tensor(tokenized["input_ids"], device=device)  # Ensure it's treated as a batch

    # Initialize the masked token's position
    mask_idx = None
    if isinstance(examples['location'], list):  # If 'location' is a list, take the first index
        mask_idx = min(examples['location'][0], 511)  # Ensure the index is within the valid range
    else:  # If 'location' is a single number
        mask_idx = min(examples['location'], 511)

    # Mask the token at the determined position
    input_ids[0, mask_idx] = tokenizer.mask_token_id  # Mask the first token at this location

    # Prepare the labels, taking only the first label (if there are multiple)
    labels = examples['label'][0]  # Use the first label if multiple are provided

    # Convert the label to token IDs if it's a string
    if isinstance(labels, str):
        labels = tokenizer.encode(labels, add_special_tokens=False)

    # Ensure the masked index is within the label list range
    if len(labels) > mask_idx:
        label_ids = labels
        label_ids[mask_idx] = tokenizer.convert_tokens_to_ids(examples['label'][0][0])  # Set the first label token
    else:
        label_ids = [tokenizer.convert_tokens_to_ids(examples['label'][0][0])]  # Set the first label as fallback

    # Return the processed data
    return {
        "input_ids": inputs,
        "labels": labels,  # The labels are now token IDs for the masked tokens
    }


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

    dataset_name = 'mcgill-nlp/medal'
    raw_datasets = load_dataset(dataset_name)

    # Reduce the dataset size to 5000 samples from the training set
    reduced_datasets = raw_datasets['train'].shuffle(seed=42).select([i for i in range(5000)])
    # Beispiel: Überprüfe das Format der Labels
    print(reduced_datasets["label"][:10])  # Zeige die ersten 10 Labels an

    # Preprocess the entire reduced dataset first
    reduced_datasets = reduced_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=20,
        desc="Running tokenizer on reduced dataset"
    )

    train_val_temp = reduced_datasets.train_test_split(test_size=0.2)

    val_test_split = train_val_temp['test'].train_test_split(test_size=0.5)

    train_dataset = train_val_temp['train']
    val_dataset = val_test_split['train']
    test_dataset = val_test_split['test']

    model = BertForMaskedLM.from_pretrained(model_name_or_path, num_labels=5)

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
        evaluation_strategy="epoch",  # evaluate each `logging_steps`
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