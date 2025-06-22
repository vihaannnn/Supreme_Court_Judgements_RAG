# Supreme_Court_Judgements_RAG


## Introduction
This project aims to create the Indian-Supreme-Court-Judgements-Rag System via a pipeline and the chunked contents from the PDF files from the Supreme Court of India.

## About the Dataset
Dataset published on - https://huggingface.co/datasets/vihaannnn/Indian-Supreme-Court-Judgements-Chunked - as a part of the AIPI 510 Course


## Prerequisites
This project requires the use of a Python virtual environment to manage dependencies and ensure consistent behavior across different systems. This guide provides step-by-step instructions for setting up a virtual environment on both Windows and Mac, as well as installing dependencies via a `requirements.txt` file.
- Python 3.x installed on your system.
- Git installed on your machine
- Basic knowledge of command-line operations.

## Cloning the Project
- Open the Command Shell or Terminal on your machine and execute the following command
   ```sh
   git clone https://github.com/vihaannnn/Supreme_Court_Judgements_RAG.git
   ```


# Time and Space Complexity Analysis

## 1. `validate_environment_variables()`

**Time Complexity:** O(n)
- Where n is the number of required environment variables (constant = 3)
- Effectively O(1) since n is fixed

**Space Complexity:** O(n)
- Creates a list to store missing variables
- Effectively O(1) since maximum size is bounded by number of required variables

## 2. `init_pinecone()`

**Time Complexity:** O(1)
- Network calls to Pinecone API are considered constant time operations
- Index stats retrieval is a single API call

**Space Complexity:** O(1)
- Only stores references to Pinecone client and index objects
- No data structures that grow with input size

## 3. `get_embedding(text)`

**Time Complexity:** O(n)
- Where n is the length of input text
- Text truncation is O(n) for scanning/slicing
- OpenAI API call time depends on text length

**Space Complexity:** O(d)
- Where d is the embedding dimension (1536 for text-embedding-ada-002)
- Returns a fixed-size vector regardless of input text length

## 4. `semantic_search(index, query_vector, top_k, namespace)`

**Time Complexity:** O(log V + k)
- Where V is the total number of vectors in the index
- Pinecone uses approximate nearest neighbor search (typically O(log V))
- k is the top_k parameter for results returned

**Space Complexity:** O(k × M)
- Where k is top_k and M is the average metadata size per match
- Results include metadata which can vary in size

## 5. `group_results_by_source(search_results)`

**Time Complexity:** O(n)
- Where n is the number of search results
- Single pass through all results to group by source

**Space Complexity:** O(n)
- In worst case, each result has a different source file
- Dictionary storage grows linearly with number of unique sources

## 6. `get_openai_response(query, context_texts, source_file)`

**Time Complexity:** O(n + API_time)
- Where n is the total length of context texts (for joining and truncation)
- API call time depends on prompt length and model processing

**Space Complexity:** O(n)
- Where n is the combined length of context texts
- Stores combined context string and prompt string

## 7. `enhance_query(original_query)`

**Time Complexity:** O(m + API_time)
- Where m is the length of original query
- API processing time is generally proportional to input length

**Space Complexity:** O(m)
- Where m is the length of the enhanced query response
- Typically similar to or slightly larger than original query

## 8. `get_additional_context_from_source(index, source_file, query_vector, exclude_ids, top_k)`

**Time Complexity:** O(log V + k + |E|)
- Where V is vectors in index, k is top_k, E is exclude_ids set
- Pinecone filtered query: O(log V + k)
- Filtering excluded IDs: O(k × |E|) worst case, but typically O(k) with set lookup

**Space Complexity:** O(k × M)
- Where k is top_k and M is average metadata size
- Filtered results list grows with number of non-excluded matches

## 9. `main()`

**Time Complexity:** O(Q + E + S + G + A)
- Where:
  - Q = query enhancement time
  - E = embedding generation time  
  - S = semantic search time
  - G = grouping results time
  - A = response generation time for all sources
- Overall: O(n × m) where n is number of sources and m is average processing time per source

**Space Complexity:** O(R × M + C)
- Where:
  - R = number of search results
  - M = average metadata size per result
  - C = total context text size across all sources
- Dominated by storing search results and context texts

## Overall Application Complexity

**Time Complexity:** O(log V + k + n × (context_processing + API_calls))
- Dominated by:
  1. Pinecone search operations: O(log V + k)
  2. Multiple OpenAI API calls: O(number_of_sources × API_time)
  3. Text processing: Linear in text length

**Space Complexity:** O(k × M + total_context_size)
- Dominated by:
  1. Search results with metadata: O(k × M)
  2. Context texts for response generation: O(total_context_size)
  3. Embedding vectors: O(embedding_dimension)

## Performance Bottlenecks

1. **API Calls:** OpenAI API calls (both for embeddings and chat completions) are the primary time bottlenecks
2. **Network I/O:** Pinecone queries and OpenAI requests involve network latency
3. **Text Processing:** Large context texts require memory and processing time
4. **Multiple Source Processing:** Sequential processing of multiple sources multiplies API call overhead

## Optimization Opportunities

1. **Parallel Processing:** Process multiple sources concurrently
2. **Caching:** Cache embeddings and frequent queries
3. **Batch Operations:** Use batch APIs where available
4. **Text Chunking:** Optimize context size to balance relevance and API limits
5. **Connection Pooling:** Reuse HTTP connections for API calls
## Setting Up a Virtual Environment

### Windows

1. **Open Command Prompt or PowerShell**:
   - Search for `cmd` or `PowerShell` in the start menu and open it.

2. **Navigate to your project directory**:
   cd (move) into your specific project path (where you have saved it on your computer), for example - 
   ```sh
   cd /Individual-Dataset
   ```

3. **Create a virtual environment**:
   ```sh
   python -m venv venv
   ```
   This creates a directory named `venv` that contains the virtual environment.

4. **Activate the virtual environment**:
   ```sh
   .\venv\Scripts\activate
   ```
   After activation, your command prompt will show `(venv)` indicating the virtual environment is active.

### Mac

1. **Open Terminal**:
   - You can find Terminal in your Applications > Utilities folder.

2. **Navigate to your project directory**:
   cd (move) into your specific project path (where you have saved it on your computer), for example - 
   ```sh
   cd /Individual-Dataset
   ```

3. **Create a virtual environment**:
   ```sh
   python3 -m venv venv
   ```
   This creates a directory named `venv` that contains the virtual environment.

4. **Activate the virtual environment**:
   ```sh
   source venv/bin/activate
   ```
   After activation, your terminal prompt will show `(venv)` indicating the virtual environment is active.

## Installing Dependencies

1. **Ensure your virtual environment is activated**:
   - Verify that `(venv)` is present in your terminal/command prompt.

2. **Install the dependencies from `requirements.txt`**:
   ```sh
   pip install -r requirements.txt
   ```
   This command installs all the packages listed in the `requirements.txt` file into your virtual environment.

   **Install the dependencies using setup.py**:
   Make sure you have setuptools installed on your machine
   ```sh
   pip install setuptools
   ```
   Next just run - 
   ```sh
   python setup.py install
   ```
   Either of these methods should have all your dependencies downloaded

## Deactivating the Virtual Environment

Once you're done working, you can deactivate the virtual environment by running:
  ```sh
  deactivate
  ```
  After deactivation, the `(venv)` prefix will disappear from your terminal/command prompt.

## Creating the ENV file
Go into the workingDir directory. Create a file name '.env'

Go to openai platform - https://platform.openai.com/docs/overview and create an API Key

Place the API key in your .env file - 
```sh
  PINECONE_API_KEY=<your-API-key>
  PINECONE_ENVIRONMENT="aws"
  PINECONE_INDEX="document-embeddings"
  OPENAI_API_KEY=<your-API-key>
```
## To run the project
Go to the root of the project

The code to run the project is - 
```sh
  cd Individual-Dataset/workingDir
  streamlit run app.py
```

## Credits
- Part of this README.md file was generated using the Artificial Intelligence agent - ChatGPT
- Original data sourced from - https://www.sci.gov.in/judgements-judgement-date/
