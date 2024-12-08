from models.bert_model import BertForMaskedLM, BertModel, BertForMultipleChoice
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

max_length = 256
model_name_or_path = 'checkpoints/checkpoint-220000'
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name_or_path)


def preprocess_function(examples):
    inputs = []  # List to store input_ids (tokenized inputs)
    labels = []  # List to store labels (tokenized masked words)

    # Loop through each example (abstract text, locations of the masked tokens, and their corresponding labels)
    for abstract, locations, label in zip(examples["text"], examples["location"], examples["label"]):
        # Tokenize the text (abstract) while padding and truncating to the maximum length
        tokenized = tokenizer(abstract, padding='max_length', max_length=max_length, truncation=True)
        input_ids = tokenized["input_ids"]  # Get the tokenized input IDs

        # Ensure that the location indices are within the bounds of max_length
        mask_indices = [i for i in locations if i < max_length]

        # Mask the tokens at the specified indices
        for idx in mask_indices:
            input_ids[idx] = tokenizer.mask_token_id  # Set the token at the specified index to the mask token ID

        label_ids = [-100] * len(input_ids)  # Initialize the labels with -100 (tokens that are not masked)

        # For each masked token, convert the label (word) into token IDs
        for idx, word in zip(mask_indices, label):
            word_ids = tokenizer.encode(word,
                                        add_special_tokens=False)  # Tokenize the word into its token IDs (without special tokens)
            if word_ids:
                label_ids[idx] = word_ids[0]  # Assign the first token ID as the label for the masked position

        # Append the processed input_ids and label_ids to their respective lists
        inputs.append(input_ids)
        labels.append(label_ids)

    # Return the final tokenized inputs and corresponding labels
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