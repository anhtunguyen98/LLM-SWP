from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from models.bert_model import BertForMaskedLM, BertModel, BertForTokenClassification
from transformers import  PreTrainedTokenizerFast
from torch.utils.data import Dataset
import torch
import json
from sklearn.model_selection import train_test_split


max_length = 512
model_name_or_path = 'checkpoints/checkpoint-220000'
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name_or_path)


def preprocess_data(data):
    ner_formatted_data = []

    for example in data["examples"]:
        text = example["content"]
        entities = []
        for annotation in example["annotations"]:
            start = annotation["start"]
            end = annotation["end"]
            entity_type = annotation["tag_name"]
            entities.append((start, end, entity_type))

        ner_formatted_data.append({
            "text": text,
            "labels": {"entities": entities}
        })
    return ner_formatted_data

entity_types = ["O", "B-MedicalCondition", "I-MedicalCondition", "B-Medicine", "I-Medicine", "B-Pathogen", "I-Pathogen"]
# Set num_labels
num_labels = len(entity_types)
def tokenize_and_format_data(dataset, tokenizer):
    tokenized_data = []
    for sample in dataset:
        text = sample["text"]
        entities = sample["labels"]["entities"]
        # Tokenize the input text using the BERT tokenizer

        tokens =  tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
        # Initialize labels for each token as 'O' (Outside)
        labels = ['O'] * len(tokens)
        # Update labels for entity spans
        for start, end, entity_type in entities:
            # Tokenize the prefix to get the correct offset
            prefix_tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text[:start])))
            start_token = len(prefix_tokens)
            # Tokenize the entity to get its length
            entity_tokens = tokenizer.tokenize(text[start:end])
            end_token = start_token + len(entity_tokens) - 1
            labels[start_token] = f"B-{entity_type}"

            for i in range(start_token + 1, end_token +1):
                labels[i] = f"I-{entity_type}"

        # Convert tokens and labels to input IDs and label IDs
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        label_ids = [entity_types.index(label) for label in labels]
        # Pad input_ids and label_ids to the maximum sequence length
        padding_length = max_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        label_ids += [entity_types.index('O')] * padding_length
        attention_mask = [1] * len(tokens) + [0] * padding_length
        tokenized_data.append({
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(label_ids)
        })
    return tokenized_data

class NERDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]

def compute_metrics(pred):
    # Extract predictions and labels from the pred object
    logits, labels = pred.predictions, pred.label_ids
    # Convert logits to predictions
    
    if isinstance(logits, tuple):
        logits = logits[0]  # Assuming the first element is the logits tensor

    # Get the index of the max log-probability
    predictions = logits.argmax(axis=-1)

    # Flatten the predictions and labels arrays
    pred_flat = predictions.flatten()
    labels_flat = labels.flatten()

    # Filter out the padding tokens (assuming padding tokens are labeled as -100)
    mask = labels_flat != -100
    pred_flat = pred_flat[mask]
    labels_flat = labels_flat[mask]

    # Compute the metrics
    f1 = f1_score(labels_flat, pred_flat, average='micro')
    acc = accuracy_score(labels_flat, pred_flat)
    
    return {
        'f1': f1,
        'acc': acc
    }




if __name__ == '__main__':



    dataset_name = 'Corona2.json'
    with open(dataset_name, 'r') as file:
        data = json.load(file)
    
    formatted_data = preprocess_data(data)
    tokenized_data = tokenize_and_format_data(formatted_data, tokenizer)
    processed_dataset = NERDataset(tokenized_data)

    train_dataset, test_val_dataset = train_test_split(processed_dataset, test_size=0.2, random_state=42)
    test_dataset, val_dataset = train_test_split(test_val_dataset, test_size=0.5, random_state=42)

    model = BertForTokenClassification.from_pretrained(model_name_or_path,num_labels=num_labels)

    # frozen bert encoder
    for param in model.bert.parameters():
        param.requires_grad = False
  

    training_args = TrainingArguments(
        output_dir='./classification_checkpoint',          # output directory
        num_train_epochs=10,               # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=8,   # batch size for evaluation
        warmup_steps=0,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./classification_checkpoint/runs',
        report_to='tensorboard',            # directory for storing logs
        load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
        metric_for_best_model='f1',
        # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
        logging_steps=10,               # log & save weights each logging_steps
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
