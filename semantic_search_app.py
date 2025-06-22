import streamlit as st
from pinecone import Pinecone
import os
from openai import OpenAI
from dotenv import load_dotenv
from collections import defaultdict
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple
import time
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('semantic_search_app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Load environment variables from .env file
try:
    load_dotenv()
    logger.info("Environment variables loaded successfully")
except Exception as e:
    logger.error(f"Failed to load environment variables: {str(e)}")
    st.error("Failed to load environment variables. Please check your .env file.")

# Decorator for error handling and logging
def handle_exceptions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"{func.__name__} completed successfully in {end_time - start_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    return wrapper

# Function to validate environment variables
def validate_environment_variables() -> Tuple[bool, List[str]]:
    """
    Validate that all required environment variables are present
    """
    required_vars = ["PINECONE_API_KEY", "PINECONE_INDEX", "OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing environment variables: {missing_vars}")
        return False, missing_vars
    
    logger.info("All required environment variables are present")
    return True, []

# Function to initialize Pinecone
@handle_exceptions
def init_pinecone() -> Optional[Any]:
    """
    Initialize Pinecone connection with comprehensive error handling
    """
    try:
        # Validate environment variables first
        is_valid, missing_vars = validate_environment_variables()
        if not is_valid:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX")
        
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        if not index_name:
            raise ValueError("PINECONE_INDEX not found in environment variables")
        
        logger.info("Initializing Pinecone client...")
        
        # Create Pinecone client
        pc = Pinecone(api_key=api_key)
        logger.info("Pinecone client created successfully")
        
        # Get index by name
        index = pc.Index(index_name)
        logger.info(f"Connected to Pinecone index: {index_name}")
        
        # Test the connection by getting index stats
        try:
            stats = index.describe_index_stats()
            logger.info(f"Index stats - Total vectors: {stats.total_vector_count}")
        except Exception as e:
            logger.warning(f"Could not retrieve index stats: {str(e)}")
        
        return index
        
    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        st.error(f"Configuration error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {str(e)}")
        st.error(f"Failed to initialize Pinecone: {str(e)}")
        return None

# Function to get embeddings for a text
@handle_exceptions
def get_embedding(text: str) -> Optional[List[float]]:
    """
    Get embeddings for text with error handling and validation
    """
    try:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Truncate text if too long (OpenAI has token limits)
        if len(text) > 8000:
            text = text[:8000]
            logger.warning("Text truncated to 8000 characters for embedding")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        client = OpenAI(api_key=api_key)
        logger.info("Getting embedding for text...")
        
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        
        if not response.data or len(response.data) == 0:
            raise ValueError("No embedding data received from OpenAI")
        
        embedding = response.data[0].embedding
        logger.info(f"Embedding generated successfully, dimension: {len(embedding)}")
        
        return embedding
        
    except ValueError as e:
        logger.error(f"Input validation error: {str(e)}")
        st.error(f"Input validation error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Failed to get embedding: {str(e)}")
        st.error(f"Failed to get embedding: {str(e)}")
        return None

# Function to perform semantic search using Pinecone's query method
@handle_exceptions
def semantic_search(index: Any, query_vector: List[float], top_k: int = 5, namespace: Optional[str] = None) -> List[Any]:
    """
    Perform semantic search using Pinecone's built-in query functionality
    """
    try:
        if not query_vector:
            raise ValueError("Query vector cannot be empty")
        
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        
        logger.info(f"Performing semantic search with top_k={top_k}")
        
        # Use Pinecone's query method for efficient semantic search
        search_results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            include_values=False,  # We don't need the actual vectors back
            namespace=namespace
        )
        
        if not search_results or not search_results.matches:
            logger.warning("No search results returned from Pinecone")
            return []
        
        logger.info(f"Found {len(search_results.matches)} matches")
        
        # Validate search results
        valid_matches = []
        for match in search_results.matches:
            if hasattr(match, 'score') and hasattr(match, 'metadata'):
                valid_matches.append(match)
            else:
                logger.warning(f"Invalid match format: {match}")
        
        return valid_matches
        
    except ValueError as e:
        logger.error(f"Search parameter error: {str(e)}")
        st.error(f"Search parameter error: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Error during semantic search: {str(e)}")
        st.error(f"Error during semantic search: {str(e)}")
        return []

# Function to group search results by source file
@handle_exceptions
def group_results_by_source(search_results: List[Any]) -> Dict[str, List[Any]]:
    """
    Group search results by source file with error handling
    """
    try:
        if not search_results:
            logger.warning("No search results to group")
            return {}
        
        grouped = defaultdict(list)
        
        for match in search_results:
            try:
                if not hasattr(match, 'metadata') or not match.metadata:
                    logger.warning(f"Match missing metadata: {match}")
                    continue
                
                source_file = match.metadata.get('source_file', 'unknown')
                grouped[source_file].append(match)
                
            except Exception as e:
                logger.warning(f"Error processing match: {str(e)}")
                continue
        
        logger.info(f"Grouped results into {len(grouped)} sources")
        return dict(grouped)
        
    except Exception as e:
        logger.error(f"Error grouping results by source: {str(e)}")
        return {}

# Function to get response from OpenAI for a single source
@handle_exceptions
def get_openai_response(query: str, context_texts: List[str], source_file: str) -> Optional[str]:
    """
    Get response from OpenAI with comprehensive error handling
    """
    try:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if not context_texts:
            raise ValueError("Context texts cannot be empty")
        
        if not source_file:
            source_file = "unknown"
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Prepare the prompt with context
        combined_context = "\n\n".join(context_texts)
        
        # Truncate context if too long
        max_context_length = 12000  # Leave room for prompt and response
        if len(combined_context) > max_context_length:
            combined_context = combined_context[:max_context_length]
            logger.warning(f"Context truncated to {max_context_length} characters")
        
        prompt = f"""
        Context information from source '{source_file}':
        {combined_context}
        
        Based only on the above context from '{source_file}', please answer the following question:
        {query}
        
        If the context doesn't contain enough information to answer, please state so clearly.
        """
        
        client = OpenAI(api_key=api_key)
        logger.info(f"Generating response for source: {source_file}")
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        if not response.choices or len(response.choices) == 0:
            raise ValueError("No response choices received from OpenAI")
        
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response content from OpenAI")
        
        logger.info(f"Response generated successfully for {source_file}")
        return content
        
    except ValueError as e:
        logger.error(f"Input validation error: {str(e)}")
        st.error(f"Input validation error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Failed to get OpenAI response for {source_file}: {str(e)}")
        st.error(f"Failed to get OpenAI response for {source_file}: {str(e)}")
        return None

# Function to enhance query using OpenAI
@handle_exceptions
def enhance_query(original_query: str) -> Optional[str]:
    """
    Enhance the user's query to make it more suitable for semantic search
    """
    try:
        if not original_query or not original_query.strip():
            raise ValueError("Original query cannot be empty")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        prompt = f"""
        We have built a system, that can bring out relevant documents related to the legal domain, based on a user's query.
        Given a query, enhance it to make it suitable for semantic search. Do not add any extra context or any surrounding words, just give me the enhanced query.
        The original query is - '{original_query}'
        """
        
        client = OpenAI(api_key=api_key)
        logger.info("Enhancing query for better semantic search")
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that enhances queries for semantic search."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        
        if not response.choices or len(response.choices) == 0:
            raise ValueError("No response choices received from OpenAI")
        
        enhanced_query = response.choices[0].message.content
        if not enhanced_query:
            raise ValueError("Empty enhanced query from OpenAI")
        
        enhanced_query = enhanced_query.strip()
        logger.info(f"Query enhanced: '{original_query}' -> '{enhanced_query}'")
        
        return enhanced_query
        
    except ValueError as e:
        logger.error(f"Query enhancement validation error: {str(e)}")
        st.warning(f"Could not enhance query: {str(e)}")
        return original_query
    except Exception as e:
        logger.error(f"Failed to enhance query: {str(e)}")
        st.warning(f"Could not enhance query, using original: {str(e)}")
        return original_query

# Function to get additional context from the same source
@handle_exceptions
def get_additional_context_from_source(index: Any, source_file: str, query_vector: List[float], 
                                     exclude_ids: set, top_k: int = 20) -> List[Any]:
    """
    Get additional context from the same source file by performing a filtered search
    """
    try:
        if not source_file:
            raise ValueError("Source file cannot be empty")
        
        if not query_vector:
            raise ValueError("Query vector cannot be empty")
        
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        
        logger.info(f"Getting additional context from source: {source_file}")
        
        # Perform a filtered search for the specific source file
        additional_results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            include_values=False,
            filter={"source_file": {"$eq": source_file}}
        )
        
        if not additional_results or not additional_results.matches:
            logger.info(f"No additional results found for source: {source_file}")
            return []
        
        # Filter out results that we already have
        filtered_results = []
        for match in additional_results.matches:
            try:
                if hasattr(match, 'id') and match.id not in exclude_ids:
                    filtered_results.append(match)
                else:
                    logger.debug(f"Excluding match ID: {match.id}")
            except Exception as e:
                logger.warning(f"Error processing additional match: {str(e)}")
                continue
        
        logger.info(f"Found {len(filtered_results)} additional matches for {source_file}")
        return filtered_results
        
    except ValueError as e:
        logger.error(f"Additional context parameter error: {str(e)}")
        return []
    except Exception as e:
        logger.warning(f"Could not get additional context from {source_file}: {str(e)}")
        return []

# Main app
def main():
    try:
        st.title("Semantic Search with Pinecone and OpenAI")
        logger.info("Starting Semantic Search application")
        
        # Initialize Pinecone
        with st.spinner("Initializing Pinecone..."):
            index = init_pinecone()
            if index is None:
                st.error("Failed to initialize Pinecone. Please check your configuration.")
                logger.error("Pinecone initialization failed, stopping application")
                return
            
            st.success("Pinecone initialized successfully!")
        
        # User input
        query = st.text_input("Enter your query:")
        
        if query:
            try:
                # Step 1: Enhance the query
                with st.spinner("Enhancing your query..."):
                    enhanced_query = enhance_query(query)
                    if enhanced_query and enhanced_query != query:
                        st.info(f"Enhanced query: {enhanced_query}")
                    else:
                        enhanced_query = query
                        st.info("Using original query")
                
                # Step 2: Get embedding for the enhanced query
                with st.spinner("Getting query embedding..."):
                    query_vector = get_embedding(enhanced_query)
                    if query_vector is None:
                        st.error("Failed to get query embedding")
                        return
                    
                    st.success("Query embedding generated successfully!")
                
                # Step 3: Perform semantic search using Pinecone's query method
                with st.spinner("Performing semantic search..."):
                    search_results = semantic_search(index, query_vector, top_k=15)
                    
                    if not search_results:
                        st.warning("No results found for your query.")
                        return
                    
                    st.success(f"Found {len(search_results)} relevant documents")
                    
                    # Display top results
                    st.subheader("Top Search Results:")
                    for i, match in enumerate(search_results[:5]):
                        try:
                            st.write(f"{i+1}. **Score**: {match.score:.4f}")
                            st.write(f"   **Source**: {match.metadata.get('source_file', 'Unknown')}")
                            st.write(f"   **ID**: {match.id}")
                            
                            if 'text' in match.metadata:
                                text_content = match.metadata['text']
                                if text_content:
                                    preview_text = text_content[:200] + "..." if len(text_content) > 200 else text_content
                                    st.write(f"   **Preview**: {preview_text}")
                            
                            st.write("---")
                        except Exception as e:
                            logger.warning(f"Error displaying search result {i}: {str(e)}")
                            continue
                
                # Step 4: Group results by source and generate responses
                with st.spinner("Generating responses by source..."):
                    # Group results by source file
                    grouped_results = group_results_by_source(search_results)
                    
                    if not grouped_results:
                        st.error("Could not group results by source")
                        return
                    
                    st.subheader("Responses by Source:")
                    
                    for source_file, matches in grouped_results.items():
                        try:
                            st.markdown(f"### 📄 Source: {source_file}")
                            
                            # Get text content from the matches
                            context_texts = []
                            match_ids = set()
                            
                            for match in matches:
                                try:
                                    if hasattr(match, 'id'):
                                        match_ids.add(match.id)
                                    
                                    if hasattr(match, 'metadata') and match.metadata and 'text' in match.metadata:
                                        text_content = match.metadata['text']
                                        if text_content and text_content.strip():
                                            context_texts.append(text_content)
                                except Exception as e:
                                    logger.warning(f"Error processing match for {source_file}: {str(e)}")
                                    continue
                            
                            # Get additional context from the same source if needed
                            if len(context_texts) < 10:  # If we have fewer than 10 contexts
                                additional_matches = get_additional_context_from_source(
                                    index, source_file, query_vector, match_ids, top_k=15
                                )
                                
                                for match in additional_matches:
                                    if len(context_texts) >= 15:  # Limit total contexts
                                        break
                                    
                                    try:
                                        if hasattr(match, 'metadata') and match.metadata and 'text' in match.metadata:
                                            text_content = match.metadata['text']
                                            if text_content and text_content.strip():
                                                context_texts.append(text_content)
                                    except Exception as e:
                                        logger.warning(f"Error processing additional match: {str(e)}")
                                        continue
                            
                            st.info(f"Using {len(context_texts)} text segments from this source")
                            
                            # Generate response using OpenAI
                            if context_texts:
                                with st.spinner(f"Generating response for {source_file}..."):
                                    response = get_openai_response(query, context_texts, source_file)
                                    if response:
                                        st.write(response)
                                    else:
                                        st.error(f"Failed to generate response for {source_file}")
                            else:
                                st.warning(f"No text content found for {source_file}")
                            
                            st.write("---")
                            
                        except Exception as e:
                            logger.error(f"Error processing source {source_file}: {str(e)}")
                            st.error(f"Error processing source {source_file}: {str(e)}")
                            continue
                
            except Exception as e:
                logger.error(f"Error in main query processing: {str(e)}")
                st.error(f"An error occurred while processing your query: {str(e)}")
                return
                
    except Exception as e:
        logger.error(f"Critical error in main application: {str(e)}")
        st.error(f"A critical error occurred: {str(e)}")
        st.error("Please check the logs for more details.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Application startup failed: {str(e)}")
        st.error(f"Application startup failed: {str(e)}")
        st.error("Please check your configuration and try again.")