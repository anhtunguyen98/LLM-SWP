import nltk
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
import multiprocessing
from tqdm import tqdm

# Download the necessary NLTK resources
nltk.download('punkt')

# Load your dataset
 

# Function to concatenate sentences
def concatenate_sentences(sentences, min_words=100, max_words=150):
    result = []
    current_segment = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence = sentence.replace('\n',' ').strip()
        sentence_word_count = len(sentence.split())
        
        if current_word_count + sentence_word_count > max_words:
            if current_segment:  # Avoid empty segments
                
                result.append(' '.join(current_segment).strip())
                current_segment = []
                current_word_count = 0
        
        current_segment.append(sentence)
        current_word_count += sentence_word_count

        # If we reach the minimum word count, store the segment
        if current_word_count >= min_words:
            result.append(' '.join(current_segment).strip())
            current_segment = []
            current_word_count = 0

    # Add any remaining sentences as a segment
    if current_segment:
        result.append(' '.join(current_segment))

    return result

# Function to process each document
def process_document(example):
    try:
        text = example['document.text']
    except:
        text = example['text']
    # text = text.replace('\n', ' ').strip()
    sentences = sent_tokenize(text)
    segments = concatenate_sentences(sentences)
    return segments

# Main processing function
def main():
    # Create a pool of workers
    dataset_medwiki = load_dataset('VOD-LM/medwiki') 
    # dataset_pubmed = load_dataset('casinca/PUBMED_title_abstracts_2019_baseline')  
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Use tqdm to show progress
        results1 = list(tqdm(pool.imap(process_document, dataset_medwiki['train']), total=len(dataset_medwiki['train']), desc='Processing Documents'))

    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #     # Use tqdm to show progress
    #     results2 = list(tqdm(pool.imap(process_document, dataset_pubmed['train']), total=len(dataset_pubmed['train']), desc='Processing Documents'))

    # Flatten the list of results
    new_segments1 = [segment for sublist in results1 for segment in sublist]
    # new_segments2 = [segment for sublist in results2 for segment in sublist]
    print(len(new_segments1))

    # # Save as a plain text file
    with open('data/train.txt', 'w', encoding='utf-8') as txt_file:
        txt_file.write('\n'.join(new_segments1[:-5000]))

    with open('data/val.txt', 'w', encoding='utf-8') as txt_file:
        txt_file.write('\n'.join(new_segments1[-5000:]))

if __name__ == "__main__":
    main()