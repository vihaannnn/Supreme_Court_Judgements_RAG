import streamlit as st
from pinecone import Pinecone
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv
from collections import defaultdict

# Load environment variables from .env file
load_dotenv()

# Function to initialize Pinecone
def init_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    
    # Create Pinecone client
    pc = Pinecone(api_key=api_key)
    
    # Get index by name
    index_name = os.getenv("PINECONE_INDEX")
    return pc.Index(index_name)

# Function to load embedding IDs from a comma-separated format
def load_embedding_ids(file_path='embedding_ids.txt'):
    with open(file_path, 'r') as file:
        content = file.read()
        # Parse the comma-separated string with quotes
        # This handles the format: "id1", "id2", "id3"
        embedding_ids = [id.strip().strip('"\'') for id in content.split(',')]
    return embedding_ids

# Function to compute cosine similarity
def cosine_similarity(vec1, vec2):
    # Compute dot product
    dot_product = np.dot(vec1, vec2)
    # Compute norms
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    # Compute cosine similarity
    return dot_product / (norm1 * norm2)

# Function to find nearest neighbors
def find_nearest_neighbors(query_vector, all_vectors, top_k=5):
    similarities = []
    
    for vector_id, vector_data in all_vectors.items():
        vector = vector_data['vector']
        similarity = cosine_similarity(query_vector, vector)
        similarities.append((vector_id, similarity, vector_data))
    
    # Sort by similarity in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k results
    return similarities[:top_k]

# Function to group vectors by source file
def group_by_source_file(vectors):
    grouped = defaultdict(list)
    for vector_id, vector_data in vectors.items():
        source_file = vector_data['metadata'].get('source_file', 'unknown')
        grouped[source_file].append((vector_id, vector_data))
    return grouped

# Function to get embeddings for a text
def get_embedding(text):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# Function to get response from OpenAI for a single source
def get_openai_response(query, context_texts, source_file):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Prepare the prompt with context
    combined_context = "\n\n".join(context_texts)
    prompt = f"""
    Context information from source '{source_file}':
    {combined_context}
    
    Based only on the above context from '{source_file}', please answer the following question:
    {query}
    
    If the context doesn't contain enough information to answer, please state so clearly.
    """
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

# Main app
def main():
    st.title("Semantic Search with Pinecone and OpenAI")
    
    # Initialize Pinecone
    with st.spinner("Initializing Pinecone..."):
        try:
            index = init_pinecone()
            st.success("Pinecone initialized successfully!")
        except Exception as e:
            st.error(f"Failed to initialize Pinecone: {str(e)}")
            return
    
    # User input
    query = st.text_input("Enter your query:")
    
    if query:
        with st.spinner("Processing your query..."):
            # Get embedding for the query
            query_vector = get_embedding(query)
            
            # Load embedding IDs
            try:
                embedding_ids = load_embedding_ids()
                # Remove any empty strings that might result from trailing commas
                embedding_ids = [id for id in embedding_ids if id]
                st.info(f"Loaded {len(embedding_ids)} embedding IDs")
            except Exception as e:
                st.error(f"Failed to load embedding IDs: {str(e)}")
                return
            
            # Fetch all vectors by ID
            try:
                all_vectors = {}
                # Process in batches of 100
                for i in range(0, len(embedding_ids), 100):
                    batch_ids = embedding_ids[i:i+100]
                    
                    # Fetch batch
                    response = index.fetch(ids=batch_ids)
                    
                    # Process each vector in the response
                    for vector_id, vector_data in response.vectors.items():
                        all_vectors[vector_id] = {
                            'vector': vector_data.values,
                            'metadata': vector_data.metadata
                        }
                
                st.info(f"Retrieved {len(all_vectors)} vectors from Pinecone")
                
            except Exception as e:
                st.error(f"Failed to retrieve vectors: {str(e)}")
                return
            
            # Find nearest neighbors
            try:
                nearest_neighbors = find_nearest_neighbors(query_vector, all_vectors, top_k=5)
                st.subheader("Top Nearest Neighbors:")
                for i, (vector_id, similarity, vector_data) in enumerate(nearest_neighbors):
                    st.write(f"{i+1}. ID: {vector_id}, Similarity: {similarity:.4f}")
                    st.write(f"   Source: {vector_data['metadata']['source_file']}")
            except Exception as e:
                st.error(f"Failed to find nearest neighbors: {str(e)}")
                return
            
            # Process by source file
            try:
                # Get unique source files from top results
                top_sources = set([nn[2]['metadata']['source_file'] for nn in nearest_neighbors])
                
                # Group all vectors by source file
                grouped_vectors = group_by_source_file(all_vectors)
                
                # Process each source file separately
                st.subheader("Responses by Source:")
                for source_file in top_sources:
                    st.markdown(f"### Source: {source_file}")
                    
                    # Get vectors for this source
                    source_vectors = grouped_vectors[source_file]
                    st.info(f"Found {len(source_vectors)} vectors from this source")
                    
                    # Extract texts from these vectors (limited to 50 for performance)
                    context_texts = []
                    for _, vector_data in source_vectors[:50]:  # Limit to first 50 vectors
                        if 'text' in vector_data['metadata']:
                            context_texts.append(vector_data['metadata']['text'])
                    
                    # Send to OpenAI for response
                    if context_texts:
                        with st.spinner(f"Getting response for source: {source_file}..."):
                            response = get_openai_response(query, context_texts, source_file)
                            st.write(response)
                    else:
                        st.warning("No text content found in the vectors from this source.")
            except Exception as e:
                st.error(f"Failed to process sources: {str(e)}")
                return

if __name__ == "__main__":
    main()