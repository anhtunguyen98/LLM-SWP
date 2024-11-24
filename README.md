# MLM Training Script

This repository contains a script for training a masked language model (MLM) using a provided dataset. The training script leverages the `mlm.py` Python file and can be executed using the included Bash script.

## Prerequisites

Make sure you have the following installed:

- Python 3.x
- Necessary Python libraries (e.g., `transformers`, `torch`, etc.)
  
You can install the required libraries with:

```bash
pip install -r requirements.txt
```
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