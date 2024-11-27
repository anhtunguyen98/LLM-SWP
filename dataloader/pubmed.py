import os
import xml.etree.ElementTree as ET
import csv
import ftplib

class PubMedProcessor:
    # set default url to download 500 million tokens
    DEFAULT_URL = '/pub/pmc/oa_bulk/oa_comm/xml/oa_comm_xml.PMC003xxxxxx.baseline.2024-06-18.tar.gz'
    DEFAULT_EXTRACTED_DIR = 'PMC003xxxxxx'
    def __init__(self, input_dir=None, output_file='pubmed.csv', file_limit=210000):
        self.input_dir = input_dir or self.DEFAULT_EXTRACTED_DIR # Directory containing XML files
        self.output_file = output_file  # Path to save the CSV file
        self.file_limit = file_limit  # Limit for files to process
        self.processed_files = 0  # Counter for processed files

    # Function to extract metadata and text from XML files
    def extract_metadata_and_text(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Extract title, abstract, and text
        title = ""
        abstract = ""
        text = []

        # Adjust based on XML structure
        for elem in root.iter():
            if elem.tag.endswith('title'):
                title = elem.text.strip() if elem.text else ""
            elif elem.tag.endswith('abstract'):
                abstract = elem.text.strip() if elem.text else ""
            elif elem.tag.endswith('p'):  # Paragraphs are often in <p> tags
                if elem.text:
                    text.append(elem.text.strip())

        # Combine the text
        full_text = ' '.join(text)
        return title, abstract, full_text

    # Function to download data from an FTP server
    def download_ftp_file(self, ftp_url, local_filename):
        # Connect to the FTP server and download the file
        ftp = ftplib.FTP('ftp.ncbi.nlm.nih.gov')
        ftp.login()  # Anonymous login
        with open(local_filename, 'wb') as local_file:
            ftp.retrbinary(f"RETR {ftp_url}", local_file.write)
        ftp.quit()

    # Function to download and extract the data (if needed)
    def download_and_extract_data(self, output_dir, url=None):
        url = url or self.DEFAULT_URL

        local_filename = os.path.join(output_dir, os.path.basename(url))
        self.download_ftp_file(url, local_filename)
        if local_filename.endswith('.tar.gz'):
            import tarfile
            with tarfile.open(local_filename, 'r:gz') as tar:
                tar.extractall(path=output_dir)

    # Function to process XML files and save them into a CSV
    def process_xml_files(self):
        with open(self.output_file, mode='w', newline='', encoding='utf-8') as csv_file:
            fieldnames = ['title', 'abstract', 'text']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            # Write the header row
            writer.writeheader()

            # Iterate over all XML files and extract text
            for file in os.listdir(self.input_dir):
                if file.endswith('.xml'):  # Ensure it's an XML file
                    file_path = os.path.join(self.input_dir, file)
                    title, abstract, full_text = self.extract_metadata_and_text(file_path)

                    # Write the data to the CSV file
                    writer.writerow({'title': title, 'abstract': abstract, 'text': full_text})

                    # Count the processed files
                    self.processed_files += 1
                    if self.processed_files >= self.file_limit:
                        print(f"Limit of {self.file_limit} files reached. Processing stopped.")
                        break

        print(f"Extracted data from {self.processed_files} files has been saved to {self.output_file}.")

