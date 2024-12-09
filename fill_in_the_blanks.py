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
        'input_ids': input_ids,
        'labels': label_ids,
        'mask_idx': mask_idx  # Save the masked token's index for later use
    }

# Define the device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocess the entire dataset
processed_data = [preprocess_function(example) for example in raw_datasets]

# Function to calculate the Mean Average Precision (MAP)
def compute_map(data, model, tokenizer, device):
    model.to(device)  # Move model to the specified device

    average_precisions = []

    for sample in data:
        # Prepare inputs
        input_ids = sample["input_ids"].to(device)  # Add a batch dimension
        true_label = sample["labels"][0]  # Always use the first label

        # Find the position of the first <mask> token
        mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
        if mask_token_index.numel() == 0:  # Skip if no <mask> token is found
            continue

        # Get model logits
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        logits = outputs[1]

        # Extract logits for the <mask> token
        mask_logits = logits[0, mask_token_index[0], :]

        # Predicted tokens
        predicted_token_ids = torch.argsort(mask_logits, descending=True)
        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_ids.tolist())

        # Calculate precision for the true label
        true_rank = predicted_tokens.index(true_label) if true_label in predicted_tokens else -1
        if true_rank != -1:
            precision = 1 / (true_rank + 1)  # Precision = 1 / (rank + 1)
            average_precisions.append(precision)

    # Return the mean average precision
    return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0

# Calculate MAP
map_score = compute_map(processed_data, model, tokenizer, device)
print(f"Mean Average Precision (MAP): {map_score:.4f}")
