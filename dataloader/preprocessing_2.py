import nltk
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
import multiprocessing
from tqdm import tqdm
import os
import ftplib
import tarfile
import xml.etree.ElementTree as ET

# Download the necessary NLTK resources
nltk.download('punkt')


# Load your dataset


# Function to concatenate sentences
def concatenate_sentences(sentences, min_words=100, max_words=150):
    result = []
    current_segment = []
    current_word_count = 0

    for sentence in sentences:
        sentence = sentence.replace('\n', ' ').strip()
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

# Function for downloading and extractiv pubmed texts
def download_and_process_pubmed():
    """
    Downloads a PubMed archive file from an FTP server, extracts it,
    and stores the extracted text from XML files in a text file in the current directory.
    """
    # FTP URL and local file names
    ftp_url = '/pub/pmc/oa_bulk/oa_comm/xml/oa_comm_xml.PMC003xxxxxx.baseline.2024-06-18.tar.gz'
    local_tar_file = 'oa_comm_xml.tar.gz'  # Local file to store the downloaded archive
    extracted_dir = 'oa_comm_xml_extracted'  # Directory to store extracted files
    output_file = 'pubmed_texts.txt'  # Output text file to save the extracted text

    # Step 1: Download the file from the FTP server
    print("Downloading the file from the FTP server...")
    with ftplib.FTP('ftp.ncbi.nlm.nih.gov') as ftp:
        ftp.login()  # Anonymous FTP login
        with open(local_tar_file, 'wb') as local_file:
            ftp.retrbinary(f"RETR {ftp_url}", local_file.write)  # Download the file in binary mode
    print(f"File {local_tar_file} has been downloaded.")

    # Step 2: Extract the archive
    os.makedirs(extracted_dir, exist_ok=True)  # Create the directory for the extracted files if it doesn't exist
    print("Extracting the archive...")
    with tarfile.open(local_tar_file, 'r:gz') as tar:
        tar.extractall(path=extracted_dir)  # Extract all files to the specified directory
    print(f"Files have been extracted to {extracted_dir}.")

    # Step 3: Extract text from XML files
    print("Extracting text from the XML files...")
    with open(output_file, 'w', encoding='utf-8') as out_file:
        # Iterate over the files in the extracted directory
        for file_name in os.listdir(extracted_dir):
            if file_name.endswith('.xml'):  # Check if it's an XML file
                file_path = os.path.join(extracted_dir, file_name)  # Get the full path of the XML file
                try:
                    tree = ET.parse(file_path)  # Parse the XML file
                    root = tree.getroot()  # Get the root element of the XML tree

                    # Extract title, abstract, and text from <title>, <abstract>, and <p> tags
                    title = ""
                    abstract = ""
                    text_segments = []

                    for elem in root.iter():  # Iterate through all elements in the XML
                        if elem.tag.endswith('title'):  # Check for the title tag
                            title = elem.text.strip() if elem.text else ""
                        elif elem.tag.endswith('abstract'):  # Check for the abstract tag
                            abstract = elem.text.strip() if elem.text else ""
                        elif elem.tag.endswith('p'):  # Check for paragraph tags
                            if elem.text:
                                text_segments.append(elem.text.strip())  # Add paragraph text

                    # Combine the extracted data into a formatted string
                    full_text = f"title: {title}\nabstract: {abstract}\ntext: {' '.join(text_segments)}\n"
                    out_file.write(full_text + "\n" + "="*80 + "\n")  # Write the text to the output file

                except Exception as e:
                    print(f"Error processing the file {file_name}: {e}")  # Handle any errors

    print(f"Extracted texts have been saved to {output_file}.")  # Final message


# Function to process each document
def process_document(example):
    try:
        text = example['document.text']
    except KeyError:
        text = example['text']
    # text = text.replace('\n', ' ').strip()
    sentences = sent_tokenize(text)
    segments = concatenate_sentences(sentences)
    return segments


# Main processing function
def main():
    # Create a pool of workers
    # dataset_medwiki = load_dataset('VOD-LM/medwiki')
    # dataset_pubmed = load_dataset('casinca/PUBMED_title_abstracts_2019_baseline')
    # Download and process PubMed dataset
    download_and_process_pubmed()
    # Read data
    with open('pubmed_texts.txt', 'r', encoding='utf-8') as file:
        dataset_pubmed = [{"document.text": line.strip()} for line in file.readlines()]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Use tqdm to show progress
        # results1 = list(tqdm(pool.imap(process_document, dataset_medwiki['train']), total=len(dataset_medwiki['train']),
        #                      desc='Processing MedWiki Documents'))
        # Verarbeite PubMed-Daten
        results2 = list(tqdm(pool.imap(process_document, dataset_pubmed), total=len(dataset_pubmed),
                             desc='Processing PubMed Documents'))

    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #     # Use tqdm to show progress
    #     results2 = list(tqdm(pool.imap(process_document, dataset_pubmed['train']), total=len(dataset_pubmed['train']), desc='Processing Documents'))

    # Flatten the list of results
    # new_segments1 = [segment for sublist in results1 for segment in sublist]
    new_segments2 = [segment for sublist in results2 for segment in sublist]
    # print(len(new_segments1))
    # print(len(new_segments2))

    # Combine MedWiki and PubMed texts
    # combined_results = new_segments1 + new_segments2
    # Save as a plain text file
    with open('data/train_pubmed.txt', 'w', encoding='utf-8') as txt_file:
        txt_file.write('\n'.join(new_segments2))

    # with open('data/val.txt', 'w', encoding='utf-8') as txt_file:
    #     txt_file.write('\n'.join(combined_results[-5000:]))


if __name__ == "__main__":
    main()