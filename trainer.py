# from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast
# from tokenizers import Tokenizer

# tokenizer = Tokenizer.from_file('assets/tokenizer.json')
# tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

# text = 'good morning how are you today?'

# print(tokenizer.tokenize(text))

# from datasets import load_dataset

# medwiki = load_dataset('VOD-LM/medwiki')
# print(medwiki)

def count_words_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        words = text.split()  # Split the text into words
        return len(words)

# Example usage
file_path = 'data/train.txt'  # Replace with your file path
word_count = count_words_in_file(file_path)
print(f'The file contains {word_count} words.')