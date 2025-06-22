import streamlit as st
from pinecone import Pinecone
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

# Function to get embeddings for a text
def get_embedding(text):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# Function to perform semantic search using Pinecone's query method
def semantic_search(index, query_vector, top_k=5, namespace=None):
    """
    Perform semantic search using Pinecone's built-in query functionality
    """
    try:
        # Use Pinecone's query method for efficient semantic search
        search_results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            include_values=False,  # We don't need the actual vectors back
            namespace=namespace
        )
        
        return search_results.matches
    except Exception as e:
        st.error(f"Error during semantic search: {str(e)}")
        return []

# Function to group search results by source file
def group_results_by_source(search_results):
    """
    Group search results by source file
    """
    grouped = defaultdict(list)
    for match in search_results:
        source_file = match.metadata.get('source_file', 'unknown')
        grouped[source_file].append(match)
    return grouped

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

# Function to enhance query using OpenAI
def enhance_query(original_query):
    """
    Enhance the user's query to make it more suitable for semantic search
    """
    prompt = f"""
    We have built a system, that can bring out relevant documents related to the legal domain, based on a user's query.
    Given a query, enhance it to make it suitable for semantic search. Do not add any extra context or any surrounding words, just give me the enhanced query.
    The original query is - '{original_query}'
    """
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that enhances queries for semantic search."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

# Function to get additional context from the same source
def get_additional_context_from_source(index, source_file, query_vector, exclude_ids, top_k=20):
    """
    Get additional context from the same source file by performing a filtered search
    """
    try:
        # Perform a filtered search for the specific source file
        additional_results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            include_values=False,
            filter={"source_file": {"$eq": source_file}}
        )
        
        # Filter out results that we already have
        filtered_results = [
            match for match in additional_results.matches 
            if match.id not in exclude_ids
        ]
        
        return filtered_results
    except Exception as e:
        st.warning(f"Could not get additional context from {source_file}: {str(e)}")
        return []

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
        # Step 1: Enhance the query
        with st.spinner("Enhancing your query..."):
            try:
                enhanced_query = enhance_query(query)
                st.info(f"Enhanced query: {enhanced_query}")
            except Exception as e:
                st.warning(f"Could not enhance query, using original: {str(e)}")
                enhanced_query = query
        
        # Step 2: Get embedding for the enhanced query
        with st.spinner("Getting query embedding..."):
            try:
                query_vector = get_embedding(enhanced_query)
                st.success("Query embedding generated successfully!")
            except Exception as e:
                st.error(f"Failed to get query embedding: {str(e)}")
                return
        
        # Step 3: Perform semantic search using Pinecone's query method
        with st.spinner("Performing semantic search..."):
            try:
                # Get top results from semantic search
                search_results = semantic_search(index, query_vector, top_k=15)
                
                if not search_results:
                    st.warning("No results found for your query.")
                    return
                
                st.success(f"Found {len(search_results)} relevant documents")
                
                # Display top results
                st.subheader("Top Search Results:")
                for i, match in enumerate(search_results[:5]):
                    st.write(f"{i+1}. **Score**: {match.score:.4f}")
                    st.write(f"   **Source**: {match.metadata.get('source_file', 'Unknown')}")
                    st.write(f"   **ID**: {match.id}")
                    if 'text' in match.metadata:
                        preview_text = match.metadata['text'][:200] + "..." if len(match.metadata['text']) > 200 else match.metadata['text']
                        st.write(f"   **Preview**: {preview_text}")
                    st.write("---")
                
            except Exception as e:
                st.error(f"Failed to perform semantic search: {str(e)}")
                return
        
        # Step 4: Group results by source and generate responses
        with st.spinner("Generating responses by source..."):
            try:
                # Group results by source file
                grouped_results = group_results_by_source(search_results)
                
                st.subheader("Responses by Source:")
                
                for source_file, matches in grouped_results.items():
                    st.markdown(f"### 📄 Source: {source_file}")
                    
                    # Get text content from the matches
                    context_texts = []
                    match_ids = set()
                    
                    for match in matches:
                        match_ids.add(match.id)
                        if 'text' in match.metadata:
                            context_texts.append(match.metadata['text'])
                    
                    # Get additional context from the same source if needed
                    if len(context_texts) < 10:  # If we have fewer than 10 contexts
                        additional_matches = get_additional_context_from_source(
                            index, source_file, query_vector, match_ids, top_k=15
                        )
                        
                        for match in additional_matches:
                            if len(context_texts) >= 15:  # Limit total contexts
                                break
                            if 'text' in match.metadata:
                                context_texts.append(match.metadata['text'])
                    
                    st.info(f"Using {len(context_texts)} text segments from this source")
                    
                    # Generate response using OpenAI
                    if context_texts:
                        with st.spinner(f"Generating response for {source_file}..."):
                            try:
                                response = get_openai_response(query, context_texts, source_file)
                                st.write(response)
                            except Exception as e:
                                st.error(f"Failed to generate response for {source_file}: {str(e)}")
                    else:
                        st.warning(f"No text content found for {source_file}")
                    
                    st.write("---")
                
            except Exception as e:
                st.error(f"Failed to process results: {str(e)}")
                return

if __name__ == "__main__":
    main()