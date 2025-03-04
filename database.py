import os
import json
import uuid
from tqdm import tqdm
from pinecone import Pinecone

# Configuration
CHUNKS_DIR = "chunks_sentences"
PINECONE_API_KEY = "pinecone-key"  # Replace with your actual API key
PINECONE_ENVIRONMENT = "aws"   # Replace with your Pinecone environment
INDEX_NAME = "document-embeddings"          # Replace with your desired index name
DIMENSION = 1536                            # Adjust based on your embedding dimension (OpenAI ada-002 is 1536)
BATCH_SIZE = 100                            # Number of vectors to upsert in each batch
ID_OUTPUT_FILE = "embedding_ids.txt"        # File to save all embedding IDs

# Initialize Pinecone with the new API format
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to the existing index
index = pc.Index(INDEX_NAME)

# List to store all embedding IDs
all_embedding_ids = []

def process_files():
    # Get all JSON files from the directory
    json_files = [f for f in os.listdir(CHUNKS_DIR) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {CHUNKS_DIR}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Process each file
    for file_name in json_files:
        file_path = os.path.join(CHUNKS_DIR, file_name)
        print(f"Processing {file_path}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            source_file = data.get("source_file", "unknown")
            chunks = data.get("chunks", [])
            
            if not chunks:
                print(f"No chunks found in {file_path}")
                continue
                
            print(f"Found {len(chunks)} chunks from source: {source_file}")
            
            # Process chunks in batches
            batch_vectors = []
            
            for chunk in tqdm(chunks, desc=f"Processing chunks from {source_file}"):
                embedding = chunk.get("embedding")
                
                # Skip if no embedding
                if not embedding:
                    print(f"Skipping chunk {chunk.get('chunk_id')} - no embedding found")
                    continue
                
                # Create a unique ID for the chunk
                # Format: sourcefilename-uuid
                unique_id = f"{source_file.replace('.pdf', '')}-{str(uuid.uuid4())}"
                
                # Add the ID to our list
                all_embedding_ids.append(unique_id)
                
                # Prepare metadata
                metadata = {
                    "text": chunk.get("text", ""),
                    "source_file": source_file,
                    "original_chunk_id": chunk.get("chunk_id"),
                    "position": chunk.get("metadata", {}).get("position")
                }
                
                # Add to batch
                batch_vectors.append((unique_id, embedding, metadata))
                
                # If batch is full, upsert and clear
                if len(batch_vectors) >= BATCH_SIZE:
                    upsert_batch(batch_vectors)
                    batch_vectors = []
            
            # Upsert any remaining vectors
            if batch_vectors:
                upsert_batch(batch_vectors)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

def upsert_batch(vectors):
    """Upsert a batch of vectors to Pinecone"""
    try:
        # Format for Pinecone upsert
        # New format requires dictionaries with 'id', 'values', and 'metadata' keys
        upsert_data = [
            {
                "id": id, 
                "values": vec, 
                "metadata": meta
            } 
            for id, vec, meta in vectors
        ]
        
        # Upsert to Pinecone
        index.upsert(vectors=upsert_data)
        print(f"Successfully upserted batch of {len(vectors)} vectors")
    except Exception as e:
        print(f"Error upserting batch to Pinecone: {e}")

def save_embedding_ids():
    """Save all embedding IDs to a file as comma-separated strings with quotes"""
    try:
        # Format each ID as a quoted string
        formatted_ids = [f'"{id}"' for id in all_embedding_ids]
        
        # Join with comma and space
        ids_string = ", ".join(formatted_ids)
        
        # Write to file
        with open(ID_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(ids_string)
        
        print(f"Successfully saved {len(all_embedding_ids)} embedding IDs to {ID_OUTPUT_FILE}")
    except Exception as e:
        print(f"Error saving embedding IDs to file: {e}")

if __name__ == "__main__":
    print("Starting Pinecone vector database upload...")
    process_files()
    
    # Save the IDs to a file
    if all_embedding_ids:
        save_embedding_ids()
    else:
        print("No embedding IDs were collected")
    
    print("Upload process complete")