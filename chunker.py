import os
import json
import requests
import re

def extract_number_from_filename(filename):
    """Extract number from filename like 'output100.txt'."""
    match = re.search(r'output(\d+)\.txt', filename)
    if match:
        return match.group(1)
    return None

def read_files(folder_path):
    """Read all text files from the specified folder and sort them."""
    texts = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                texts.append((filename, f.read()))
    return texts

def get_existing_chunks(output_folder):
    """Get list of already processed document numbers."""
    if not os.path.exists(output_folder):
        return set()
    
    existing_numbers = set()
    for filename in os.listdir(output_folder):
        if filename.endswith('_chunks.json'):
            doc_number = filename.replace('_chunks.json', '')
            if doc_number.isdigit():
                existing_numbers.add(doc_number)
    return existing_numbers

def chunk_by_sentences(text, max_sentences=5):
    """Chunk text by number of sentences."""
    sentences = text.replace('!', '.').replace('?', '.').split('. ')
    chunks = []
    current_chunk = []
    
    for sentence in sentences:
        current_chunk.append(sentence)
        if len(current_chunk) >= max_sentences:
            chunks.append('. '.join(current_chunk).strip() + '.')
            current_chunk = []
    
    if current_chunk:
        chunks.append('. '.join(current_chunk).strip() + '.')
    
    return chunks

def create_embeddings(chunks_data, api_key):
    """Create embeddings using OpenAI API and update chunks_data in place."""
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    for chunk in chunks_data['chunks']:
        data = {
            "input": chunk['text'],
            "model": "text-embedding-ada-002"
        }
        
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            embedding = response.json()['data'][0]['embedding']
            chunk['embedding'] = embedding
        else:
            print(f"Error creating embedding for chunk {chunk['chunk_id']}: {response.text}")
            chunk['embedding'] = None

def save_chunks_to_json(chunks, doc_number, output_folder, openai_api_key=None):
    """Save chunks with optional embeddings to a JSON file."""
    os.makedirs(output_folder, exist_ok=True)
    
    pdf_name = f"{doc_number}.pdf"
    
    # Create JSON structure with PDF name
    chunks_data = {
        "source_file": pdf_name,
        "chunks": [
            {
                "chunk_id": i,
                "text": chunk,
                "metadata": {
                    "position": i,
                    "source_file": pdf_name
                }
            }
            for i, chunk in enumerate(chunks, 1)
        ]
    }
    
    # Add embeddings if API key is provided
    if openai_api_key:
        create_embeddings(chunks_data, openai_api_key)
    
    # Save as JSON file using the document number
    json_filename = f"{doc_number}_chunks.json"
    json_path = os.path.join(output_folder, json_filename)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    
    return chunks_data

def process_folder(input_folder, openai_api_key=None):
    """Process all texts in folder using sentence chunking, skipping existing files."""
    texts = read_files(input_folder)
    all_documents = []
    output_folder = "chunks_sentences"
    
    # Get list of already processed documents
    existing_chunks = get_existing_chunks(output_folder)
    print(f"Found {len(existing_chunks)} existing processed files")
    
    # Process files
    files_processed = 0
    files_skipped = 0
    
    for filename, text in texts:
        doc_number = extract_number_from_filename(filename)
        if not doc_number:
            print(f"Warning: Could not extract number from filename {filename}")
            continue
            
        # Skip if already processed
        if doc_number in existing_chunks:
            print(f"Skipping {filename} (already processed)")
            files_skipped += 1
            # Load existing data for the summary
            json_path = os.path.join(output_folder, f"{doc_number}_chunks.json")
            with open(json_path, 'r', encoding='utf-8') as f:
                document_data = json.load(f)
                all_documents.append(document_data)
            continue
            
        print(f"Processing {filename}")
        chunks = chunk_by_sentences(text)
        document_data = save_chunks_to_json(chunks, doc_number, output_folder, openai_api_key)
        all_documents.append(document_data)
        files_processed += 1
    
    # Save all documents summary
    summary_path = os.path.join(output_folder, 'all_documents.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total_documents": len(all_documents),
            "documents": all_documents,
            "processing_summary": {
                "files_processed": files_processed,
                "files_skipped": files_skipped,
                "total_files": files_processed + files_skipped
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessing complete:")
    print(f"- Files processed: {files_processed}")
    print(f"- Files skipped: {files_skipped}")
    print(f"- Total files: {files_processed + files_skipped}")
    
    return all_documents

process_folder('output', openai_api_key="key-to-be-entered")