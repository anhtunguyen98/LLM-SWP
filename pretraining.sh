#!/bin/bash

# Set default values for the arguments
TOKENIZER_PATH='assets/tokenizer.json'
TRAIN_FILE='data/train.txt'
VALIDATION_FILE='data/val.txt'
MAX_SEQ_LENGTH=256
PREPROCESSING_NUM_WORKERS=20
MLM_PROBABILITY=0.15
WARMUP_RATIO=0.1
LEARNING_RATE=5e-5
PER_DEVICE_TRAIN_BATCH_SIZE=32
PER_DEVICE_EVAL_BATCH_SIZE=32
NUM_TRAIN_EPOCHS=5
GRADIENT_ACCUMULATION_STEPS=1
OUTPUT_DIR='checkpoints/'
LOGGING_STEPS=100
EVAL_STEPS=200
SAVE_STEPS=200

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