# LLM Pretraining Software Project

This repository contains a script for training a BERT using a provided dataset.

## Prerequisites

Make sure you have the following installed:

- Python 3.x
- Necessary Python libraries (e.g., `transformers`, `torch`, etc.)
  
You can install the required libraries with:

```bash
pip install -r requirements.txt
```
# Download the dataset on this link 
https://huggingface.co/datasets/anhtunguyen98/LLM-SWP
# File Structure
```
.
├── assets
│   └── tokenizer.json
├── data
│   ├── train.txt
│   └── val.txt
├── mlm.py
├── nsp.py
└── mlm_nsp.py
```
# Preprocess

```
python preprocess_corpus.py
```

# Pretraining MLM
```
#!/bin/bash

# Set default values for the arguments
TOKENIZER_PATH='assets/tokenizer.json'
TRAIN_FILE='data/train.txt'
VALIDATION_FILE='data/val.txt'
MAX_SEQ_LENGTH=384
PREPROCESSING_NUM_WORKERS=20
MLM_PROBABILITY=0.15
WARMUP_RATIO=0.1
LEARNING_RATE=5e-5
PER_DEVICE_TRAIN_BATCH_SIZE=64
PER_DEVICE_EVAL_BATCH_SIZE=64
NUM_TRAIN_EPOCHS=5
GRADIENT_ACCUMULATION_STEPS=1
OUTPUT_DIR='checkpoints/'
LOGGING_STEPS=100
EVAL_STEPS=20000
SAVE_STEPS=20000

# Execute the Python script with the specified arguments
python mlm.py \
  --tokenizer_path "$TOKENIZER_PATH" \
  --train_file "$TRAIN_FILE" \
  --validation_file "$VALIDATION_FILE" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --preprocessing_num_workers "$PREPROCESSING_NUM_WORKERS" \
  --mlm_probability "$MLM_PROBABILITY" \
  --warmup_ratio "$WARMUP_RATIO" \
  --learning_rate "$LEARNING_RATE" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --output_dir "$OUTPUT_DIR" \
  --logging_steps "$LOGGING_STEPS" \
  --eval_steps "$EVAL_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --resume_from_checkpoint /home/ubuntu/tuna/saarland/LLM-SWP/checkpoints/checkpoint-220000 \
```
# Pretraining NSP
```
#!/bin/bash

# Set default values for the arguments
TOKENIZER_PATH='assets/tokenizer.json'
TRAIN_FILE='data/train.txt'
VALIDATION_FILE='data/val.txt'
MAX_SEQ_LENGTH=384
PREPROCESSING_NUM_WORKERS=20
MLM_PROBABILITY=0.15
WARMUP_RATIO=0.1
LEARNING_RATE=5e-5
PER_DEVICE_TRAIN_BATCH_SIZE=64
PER_DEVICE_EVAL_BATCH_SIZE=64
NUM_TRAIN_EPOCHS=5
GRADIENT_ACCUMULATION_STEPS=1
OUTPUT_DIR='checkpoints/'
LOGGING_STEPS=100
EVAL_STEPS=20000
SAVE_STEPS=20000

# Execute the Python script with the specified arguments
python nsp.py \
  --tokenizer_path "$TOKENIZER_PATH" \
  --train_file "$TRAIN_FILE" \
  --validation_file "$VALIDATION_FILE" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --preprocessing_num_workers "$PREPROCESSING_NUM_WORKERS" \
  --mlm_probability "$MLM_PROBABILITY" \
  --warmup_ratio "$WARMUP_RATIO" \
  --learning_rate "$LEARNING_RATE" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --output_dir "$OUTPUT_DIR" \
  --logging_steps "$LOGGING_STEPS" \
  --eval_steps "$EVAL_STEPS" \
  --save_steps "$SAVE_STEPS" \
```
# Joint Pretraining MLM and NSP
```
#!/bin/bash

# Set default values for the arguments
TOKENIZER_PATH='assets/tokenizer.json'
TRAIN_FILE='data/train.txt'
VALIDATION_FILE='data/val.txt'
MAX_SEQ_LENGTH=384
PREPROCESSING_NUM_WORKERS=20
MLM_PROBABILITY=0.15
WARMUP_RATIO=0.1
LEARNING_RATE=5e-5
PER_DEVICE_TRAIN_BATCH_SIZE=64
PER_DEVICE_EVAL_BATCH_SIZE=64
NUM_TRAIN_EPOCHS=5
GRADIENT_ACCUMULATION_STEPS=1
OUTPUT_DIR='checkpoints/'
LOGGING_STEPS=100
EVAL_STEPS=20000
SAVE_STEPS=20000

# Execute the Python script with the specified arguments
python mlm_nsp.py \
  --tokenizer_path "$TOKENIZER_PATH" \
  --train_file "$TRAIN_FILE" \
  --validation_file "$VALIDATION_FILE" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --preprocessing_num_workers "$PREPROCESSING_NUM_WORKERS" \
  --mlm_probability "$MLM_PROBABILITY" \
  --warmup_ratio "$WARMUP_RATIO" \
  --learning_rate "$LEARNING_RATE" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --output_dir "$OUTPUT_DIR" \
  --logging_steps "$LOGGING_STEPS" \
  --eval_steps "$EVAL_STEPS" \
  --save_steps "$SAVE_STEPS" \
```




