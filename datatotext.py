import pdfplumber
import os
import re
from dotenv import load_dotenv
def create_text(directory, output_directory_name, output_normal_directory_name):
    """
    Processes PDF files from the './data' directory and extracts their text content.
    
    This function:
    1. Reads all PDF files in the './data' directory
    2. Extracts text from each page
    3. Cleans the extracted text by removing hidden characters
    4. Saves both cleaned and original versions to separate output directories

    Args: 
    directory - Directory where the original PDF files are present in.
    output_directory_name - Directory where the processed output of the PDF documents should be sent.
    output_normal_directory_name - Directory where the unprocessed output of the PDF documents should be written.
    
    Output files are saved in:
    - './output/' for cleaned text
    - './output_normal/' for original text
    """
    c = ''
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            print(filename)
            c = filename[:-4]
        # Open the PDF file
        with pdfplumber.open(directory + "/" + filename) as pdf:

            # Iterate through all pages
            
            text = ''
            pages = pdf.pages
            text_total = ''
            total_text_normal = ''
            for page in pages:
                # Extract text from the page
                text = page.extract_text()
                
                text_normal = text
                # Remove common hidden characters, but keep newlines
                text = re.sub(r'[\x00-\x09\x0B-\x1F\x7F-\x9F]', '', text)
            
                # Remove zero-width characters
                text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
            
                # Remove other invisible separator characters, but keep newlines
                text = re.sub(r'[\u2000-\u200F\u2028-\u202E\u205F-\u206F]', '', text)
            
                # Remove control characters, but keep newlines
                text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')

                text = re.sub(r'\n', ' ', text)
            
                text_total += text 
                total_text_normal += text_normal
            
            # write total text without any line seperation to the output directory - modified
            with open(output_directory_name +'/output' + str(c) + '.txt', 'w') as file:      
                file.write(text_total)
            
            # write total text - normal without modification to the output_normal directory
            with open(output_normal_directory_name + '/output_normal' + str(c) + '.txt', 'w') as file:      
                file.write(total_text_normal)