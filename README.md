# MLM Training Script

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
└── run_mlm.sh
```
# Preprocess

```
python preprocess_corpus.py
```

# Pretraining
```
bash pretraining.sh
