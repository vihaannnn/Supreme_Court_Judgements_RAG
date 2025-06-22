import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from collections import defaultdict
import logging
from typing import List, Dict, Any

# Add the main module to the path (adjust the import based on your file structure)
# Assuming the main code is in semantic_search_app.py
try:
    from semantic_search_app import (
        validate_environment_variables,
        init_pinecone,
        get_embedding,
        semantic_search,
        group_results_by_source,
        get_openai_response,
        enhance_query,
        get_additional_context_from_source,
        handle_exceptions
    )
except ImportError:
    # If running as a module, adjust the import path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from semantic_search_app import (
        validate_environment_variables,
        init_pinecone,
        get_embedding,
        semantic_search,
        group_results_by_source,
        get_openai_response,
        enhance_query,
        get_additional_context_from_source,
        handle_exceptions
    )

# Test fixtures
@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing"""
    return {
        'PINECONE_API_KEY': 'test-pinecone-key',
        'PINECONE_INDEX': 'test-index',
        'OPENAI_API_KEY': 'test-openai-key'
    }

@pytest.fixture
def mock_pinecone_index():
    """Mock Pinecone index for testing"""
    mock_index = Mock()
    mock_index.describe_index_stats.return_value = Mock(total_vector_count=1000)
    mock_index.query.return_value = Mock(
        matches=[
            Mock(
                id='test-id-1',
                score=0.95,
                metadata={'text': 'Test document content', 'source_file': 'test.pdf'}
            ),
            Mock(
                id='test-id-2',
                score=0.87,
                metadata={'text': 'Another test document', 'source_file': 'test2.pdf'}
            )
        ]
    )
    return mock_index

@pytest.fixture
def mock_openai_embedding_response():
    """Mock OpenAI embedding response"""
    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3] * 512)]  # 1536 dimensions
    return mock_response

@pytest.fixture
def mock_openai_chat_response():
    """Mock OpenAI chat completion response"""
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test response from OpenAI"))]
    return mock_response

@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    return [
        Mock(
            id='doc1-chunk1',
            score=0.95,
            metadata={'text': 'Legal document content about contracts', 'source_file': 'contract_law.pdf'}
        ),
        Mock(
            id='doc1-chunk2',
            score=0.87,
            metadata={'text': 'More contract information', 'source_file': 'contract_law.pdf'}
        ),
        Mock(
            id='doc2-chunk1',
            score=0.82,
            metadata={'text': 'Employment law content', 'source_file': 'employment_law.pdf'}
        )
    ]

class TestValidateEnvironmentVariables:
    """Test suite for validate_environment_variables function"""
    
    def test_all_variables_present(self, mock_env_vars):
        """Test when all required environment variables are present"""
        with patch.dict(os.environ, mock_env_vars):
            is_valid, missing_vars = validate_environment_variables()
            assert is_valid is True
            assert missing_vars == []
    
    def test_missing_single_variable(self, mock_env_vars):
        """Test when one environment variable is missing"""
        incomplete_vars = mock_env_vars.copy()
        del incomplete_vars['PINECONE_API_KEY']
        
        with patch.dict(os.environ, incomplete_vars, clear=True):
            is_valid, missing_vars = validate_environment_variables()
            assert is_valid is False
            assert 'PINECONE_API_KEY' in missing_vars
    
    def test_missing_multiple_variables(self):
        """Test when multiple environment variables are missing"""
        with patch.dict(os.environ, {}, clear=True):
            is_valid, missing_vars = validate_environment_variables()
            assert is_valid is False
            assert len(missing_vars) == 3
            assert all(var in missing_vars for var in ['PINECONE_API_KEY', 'PINECONE_INDEX', 'OPENAI_API_KEY'])
    
    def test_empty_environment_variables(self, mock_env_vars):
        """Test when environment variables are empty strings"""
        empty_vars = {key: '' for key in mock_env_vars.keys()}
        
        with patch.dict(os.environ, empty_vars, clear=True):
            is_valid, missing_vars = validate_environment_variables()
            assert is_valid is False
            assert len(missing_vars) == 3

class TestInitPinecone:
    """Test suite for init_pinecone function"""
    
    @patch('semantic_search_app.Pinecone')
    def test_successful_initialization(self, mock_pinecone_class, mock_env_vars):
        """Test successful Pinecone initialization"""
        mock_pc = Mock()
        mock_index = Mock()
        mock_index.describe_index_stats.return_value = Mock(total_vector_count=1000)
        mock_pc.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_pc
        
        with patch.dict(os.environ, mock_env_vars):
            result = init_pinecone()
            
            assert result is not None
            mock_pinecone_class.assert_called_once_with(api_key='test-pinecone-key')
            mock_pc.Index.assert_called_once_with('test-index')
    
    @patch('semantic_search_app.Pinecone')
    def test_missing_api_key(self, mock_pinecone_class):
        """Test initialization with missing API key"""
        with patch.dict(os.environ, {}, clear=True):
            result = init_pinecone()
            assert result is None
    
    @patch('semantic_search_app.Pinecone')
    def test_pinecone_connection_error(self, mock_pinecone_class, mock_env_vars):
        """Test Pinecone connection error"""
        mock_pinecone_class.side_effect = Exception("Connection failed")
        
        with patch.dict(os.environ, mock_env_vars):
            result = init_pinecone()
            assert result is None
    
    @patch('semantic_search_app.Pinecone')
    def test_index_stats_error(self, mock_pinecone_class, mock_env_vars):
        """Test when index stats retrieval fails"""
        mock_pc = Mock()
        mock_index = Mock()
        mock_index.describe_index_stats.side_effect = Exception("Stats error")
        mock_pc.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_pc
        
        with patch.dict(os.environ, mock_env_vars):
            result = init_pinecone()
            # Should still return the index even if stats fail
            assert result is not None

class TestGetEmbedding:
    """Test suite for get_embedding function"""
    
    @patch('semantic_search_app.OpenAI')
    def test_successful_embedding(self, mock_openai_class, mock_env_vars, mock_openai_embedding_response):
        """Test successful embedding generation"""
        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_openai_embedding_response
        mock_openai_class.return_value = mock_client
        
        with patch.dict(os.environ, mock_env_vars):
            result = get_embedding("Test text")
            
            assert result is not None
            assert len(result) == 1536  # OpenAI embedding dimension
            mock_client.embeddings.create.assert_called_once()
    
    def test_empty_text(self):
        """Test embedding with empty text"""
        result = get_embedding("")
        assert result is None
    
    def test_whitespace_only_text(self):
        """Test embedding with whitespace-only text"""
        result = get_embedding("   ")
        assert result is None
    
    @patch('semantic_search_app.OpenAI')
    def test_long_text_truncation(self, mock_openai_class, mock_env_vars, mock_openai_embedding_response):
        """Test text truncation for long inputs"""
        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_openai_embedding_response
        mock_openai_class.return_value = mock_client
        
        long_text = "A" * 10000  # Text longer than 8000 characters
        
        with patch.dict(os.environ, mock_env_vars):
            result = get_embedding(long_text)
            
            assert result is not None
            # Check that the text was truncated
            call_args = mock_client.embeddings.create.call_args
            assert len(call_args[1]['input']) == 8000
    
    @patch('semantic_search_app.OpenAI')
    def test_openai_api_error(self, mock_openai_class, mock_env_vars):
        """Test OpenAI API error handling"""
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client
        
        with patch.dict(os.environ, mock_env_vars):
            result = get_embedding("Test text")
            assert result is None
    
    def test_missing_openai_key(self):
        """Test embedding with missing OpenAI API key"""
        with patch.dict(os.environ, {}, clear=True):
            result = get_embedding("Test text")
            assert result is None

class TestSemanticSearch:
    """Test suite for semantic_search function"""
    
    def test_successful_search(self, mock_pinecone_index):
        """Test successful semantic search"""
        query_vector = [0.1, 0.2, 0.3] * 512
        
        result = semantic_search(mock_pinecone_index, query_vector, top_k=5)
        
        assert result is not None
        assert len(result) == 2
        mock_pinecone_index.query.assert_called_once()
    
    def test_empty_query_vector(self, mock_pinecone_index):
        """Test search with empty query vector"""
        result = semantic_search(mock_pinecone_index, [], top_k=5)
        assert result == []
    
    def test_invalid_top_k(self, mock_pinecone_index):
        """Test search with invalid top_k parameter"""
        query_vector = [0.1, 0.2, 0.3] * 512
        
        result = semantic_search(mock_pinecone_index, query_vector, top_k=0)
        assert result == []
        
        result = semantic_search(mock_pinecone_index, query_vector, top_k=-1)
        assert result == []
    
    def test_no_search_results(self, mock_pinecone_index):
        """Test when no search results are returned"""
        mock_pinecone_index.query.return_value = Mock(matches=[])
        query_vector = [0.1, 0.2, 0.3] * 512
        
        result = semantic_search(mock_pinecone_index, query_vector, top_k=5)
        assert result == []
    
    def test_pinecone_query_error(self, mock_pinecone_index):
        """Test Pinecone query error handling"""
        mock_pinecone_index.query.side_effect = Exception("Query failed")
        query_vector = [0.1, 0.2, 0.3] * 512
        
        result = semantic_search(mock_pinecone_index, query_vector, top_k=5)
        assert result == []
    
    def test_invalid_match_format(self, mock_pinecone_index):
        """Test handling of invalid match formats"""
        invalid_match = Mock()
        del invalid_match.score  # Remove required attribute
        
        mock_pinecone_index.query.return_value = Mock(matches=[invalid_match])
        query_vector = [0.1, 0.2, 0.3] * 512
        
        result = semantic_search(mock_pinecone_index, query_vector, top_k=5)
        assert result == []

class TestGroupResultsBySource:
    """Test suite for group_results_by_source function"""
    
    def test_successful_grouping(self, sample_search_results):
        """Test successful grouping of search results"""
        result = group_results_by_source(sample_search_results)
        
        assert len(result) == 2
        assert 'contract_law.pdf' in result
        assert 'employment_law.pdf' in result
        assert len(result['contract_law.pdf']) == 2
        assert len(result['employment_law.pdf']) == 1
    
    def test_empty_search_results(self):
        """Test grouping with empty search results"""
        result = group_results_by_source([])
        assert result == {}
    
    def test_missing_metadata(self):
        """Test grouping with missing metadata"""
        invalid_results = [Mock(metadata=None), Mock()]
        del invalid_results[1].metadata  # Remove metadata attribute
        
        result = group_results_by_source(invalid_results)
        assert result == {}
    
    def test_missing_source_file(self):
        """Test grouping with missing source_file in metadata"""
        results_without_source = [
            Mock(metadata={'text': 'Test content'})  # No source_file
        ]
        
        result = group_results_by_source(results_without_source)
        assert 'unknown' in result
        assert len(result['unknown']) == 1

class TestGetOpenAIResponse:
    """Test suite for get_openai_response function"""
    
    @patch('semantic_search_app.OpenAI')
    def test_successful_response(self, mock_openai_class, mock_env_vars, mock_openai_chat_response):
        """Test successful OpenAI response generation"""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_chat_response
        mock_openai_class.return_value = mock_client
        
        query = "What is contract law?"
        context_texts = ["Contract law is...", "Legal contracts are..."]
        source_file = "contract_law.pdf"
        
        with patch.dict(os.environ, mock_env_vars):
            result = get_openai_response(query, context_texts, source_file)
            
            assert result is not None
            assert result == "Test response from OpenAI"
            mock_client.chat.completions.create.assert_called_once()
    
    def test_empty_query(self):
        """Test response with empty query"""
        result = get_openai_response("", ["context"], "source.pdf")
        assert result is None
    
    def test_empty_context(self):
        """Test response with empty context"""
        result = get_openai_response("query", [], "source.pdf")
        assert result is None
    
    @patch('semantic_search_app.OpenAI')
    def test_long_context_truncation(self, mock_openai_class, mock_env_vars, mock_openai_chat_response):
        """Test context truncation for long inputs"""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_chat_response
        mock_openai_class.return_value = mock_client
        
        long_context = ["A" * 15000]  # Very long context
        
        with patch.dict(os.environ, mock_env_vars):
            result = get_openai_response("query", long_context, "source.pdf")
            
            assert result is not None
            # Verify the context was truncated
            call_args = mock_client.chat.completions.create.call_args
            prompt = call_args[1]['messages'][1]['content']
            assert len(prompt) < 15000
    
    @patch('semantic_search_app.OpenAI')
    def test_openai_api_error(self, mock_openai_class, mock_env_vars):
        """Test OpenAI API error handling"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client
        
        with patch.dict(os.environ, mock_env_vars):
            result = get_openai_response("query", ["context"], "source.pdf")
            assert result is None
    
    @patch('semantic_search_app.OpenAI')
    def test_empty_response_content(self, mock_openai_class, mock_env_vars):
        """Test handling of empty response content"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=None))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        with patch.dict(os.environ, mock_env_vars):
            result = get_openai_response("query", ["context"], "source.pdf")
            assert result is None

class TestEnhanceQuery:
    """Test suite for enhance_query function"""
    
    @patch('semantic_search_app.OpenAI')
    def test_successful_enhancement(self, mock_openai_class, mock_env_vars):
        """Test successful query enhancement"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Enhanced legal query about contracts"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        original_query = "contract law"
        
        with patch.dict(os.environ, mock_env_vars):
            result = enhance_query(original_query)
            
            assert result is not None
            assert result == "Enhanced legal query about contracts"
            mock_client.chat.completions.create.assert_called_once()
    
    def test_empty_query(self):
        """Test enhancement with empty query"""
        result = enhance_query("")
        assert result == ""
    
    def test_whitespace_only_query(self):
        """Test enhancement with whitespace-only query"""
        result = enhance_query("   ")
        assert result == "   "
    
    @patch('semantic_search_app.OpenAI')
    def test_openai_api_error(self, mock_openai_class, mock_env_vars):
        """Test OpenAI API error handling - should return original query"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client
        
        original_query = "contract law"
        
        with patch.dict(os.environ, mock_env_vars):
            result = enhance_query(original_query)
            assert result == original_query
    
    def test_missing_openai_key(self):
        """Test enhancement with missing OpenAI API key"""
        original_query = "contract law"
        
        with patch.dict(os.environ, {}, clear=True):
            result = enhance_query(original_query)
            assert result == original_query

class TestGetAdditionalContextFromSource:
    """Test suite for get_additional_context_from_source function"""
    
    def test_successful_additional_context(self, mock_pinecone_index):
        """Test successful retrieval of additional context"""
        # Mock additional results
        additional_matches = [
            Mock(id='new-id-1', metadata={'text': 'Additional context 1'}),
            Mock(id='new-id-2', metadata={'text': 'Additional context 2'})
        ]
        mock_pinecone_index.query.return_value = Mock(matches=additional_matches)
        
        source_file = "test.pdf"
        query_vector = [0.1, 0.2, 0.3] * 512
        exclude_ids = {'existing-id-1', 'existing-id-2'}
        
        result = get_additional_context_from_source(
            mock_pinecone_index, source_file, query_vector, exclude_ids
        )
        
        assert len(result) == 2
        assert all(match.id not in exclude_ids for match in result)
        
        # Verify the query was called with correct filter
        mock_pinecone_index.query.assert_called_once()
        call_args = mock_pinecone_index.query.call_args
        assert call_args[1]['filter'] == {"source_file": {"$eq": source_file}}
    
    def test_empty_source_file(self, mock_pinecone_index):
        """Test with empty source file"""
        result = get_additional_context_from_source(
            mock_pinecone_index, "", [0.1, 0.2, 0.3], set()
        )
        assert result == []
    
    def test_empty_query_vector(self, mock_pinecone_index):
        """Test with empty query vector"""
        result = get_additional_context_from_source(
            mock_pinecone_index, "test.pdf", [], set()
        )
        assert result == []
    
    def test_invalid_top_k(self, mock_pinecone_index):
        """Test with invalid top_k parameter"""
        result = get_additional_context_from_source(
            mock_pinecone_index, "test.pdf", [0.1, 0.2, 0.3], set(), top_k=0
        )
        assert result == []
    
    def test_no_additional_results(self, mock_pinecone_index):
        """Test when no additional results are found"""
        mock_pinecone_index.query.return_value = Mock(matches=[])
        
        result = get_additional_context_from_source(
            mock_pinecone_index, "test.pdf", [0.1, 0.2, 0.3], set()
        )
        assert result == []
    
    def test_all_results_excluded(self, mock_pinecone_index):
        """Test when all results are in exclude list"""
        excluded_matches = [
            Mock(id='excluded-id-1'),
            Mock(id='excluded-id-2')
        ]
        mock_pinecone_index.query.return_value = Mock(matches=excluded_matches)
        
        exclude_ids = {'excluded-id-1', 'excluded-id-2'}
        
        result = get_additional_context_from_source(
            mock_pinecone_index, "test.pdf", [0.1, 0.2, 0.3], exclude_ids
        )
        assert result == []
    
    def test_pinecone_query_error(self, mock_pinecone_index):
        """Test Pinecone query error handling"""
        mock_pinecone_index.query.side_effect = Exception("Query failed")
        
        result = get_additional_context_from_source(
            mock_pinecone_index, "test.pdf", [0.1, 0.2, 0.3], set()
        )
        assert result == []

class TestHandleExceptionsDecorator:
    """Test suite for handle_exceptions decorator"""
    
    def test_successful_function_execution(self):
        """Test decorator with successful function execution"""
        @handle_exceptions
        def test_function(x, y):
            return x + y
        
        result = test_function(2, 3)
        assert result == 5
    
    def test_function_with_exception(self):
        """Test decorator with function that raises exception"""
        @handle_exceptions
        def test_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            test_function()
    
    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function metadata"""
        @handle_exceptions
        def test_function():
            """Test docstring"""
            pass
        
        assert test_function.__name__ == "test_function"
        assert test_function.__doc__ == "Test docstring"

# Integration tests
class TestIntegration:
    """Integration tests for the complete workflow"""
    
    @patch('semantic_search_app.OpenAI')
    @patch('semantic_search_app.Pinecone')
    def test_complete_search_workflow(self, mock_pinecone_class, mock_openai_class, mock_env_vars):
        """Test the complete search workflow"""
        # Setup mocks
        mock_pc = Mock()
        mock_index = Mock()
        mock_index.describe_index_stats.return_value = Mock(total_vector_count=1000)
        mock_index.query.return_value = Mock(
            matches=[
                Mock(
                    id='doc1',
                    score=0.9,
                    metadata={'text': 'Contract law content', 'source_file': 'contract.pdf'}
                )
            ]
        )
        mock_pc.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_pc
        
        mock_openai_client = Mock()
        mock_openai_client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=[0.1] * 1536)]
        )
        mock_openai_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="This is about contract law"))]
        )
        mock_openai_class.return_value = mock_openai_client
        
        with patch.dict(os.environ, mock_env_vars):
            # Test the workflow
            index = init_pinecone()
            assert index is not None
            
            embedding = get_embedding("contract law")
            assert embedding is not None
            
            results = semantic_search(index, embedding)
            assert len(results) == 1
            
            grouped = group_results_by_source(results)
            assert 'contract.pdf' in grouped
            
            response = get_openai_response(
                "What is contract law?", 
                ["Contract law content"], 
                "contract.pdf"
            )
            assert response == "This is about contract law"

# Performance tests
class TestPerformance:
    """Performance tests for critical functions"""
    
    def test_large_context_handling(self):
        """Test handling of large context texts"""
        large_contexts = ["A" * 1000 for _ in range(100)]  # 100 contexts of 1000 chars each
        
        # This should not raise an exception
        grouped = group_results_by_source([
            Mock(
                id=f'doc-{i}',
                score=0.9,
                metadata={'text': context, 'source_file': 'large_doc.pdf'}
            )
            for i, context in enumerate(large_contexts)
        ])
        
        assert 'large_doc.pdf' in grouped
        assert len(grouped['large_doc.pdf']) == 100

# Fixtures for pytest configuration
@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for tests"""
    logging.getLogger().setLevel(logging.DEBUG)

@pytest.fixture(autouse=True)
def cleanup_environment():
    """Clean up environment after each test"""
    yield
    # Cleanup code here if needed

# Test configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )

# Run specific test categories
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers"""
    if config.getoption("--integration"):
        # Only run integration tests
        items[:] = [item for item in items if "integration" in item.keywords]
    elif config.getoption("--performance"):
        # Only run performance tests
        items[:] = [item for item in items if "performance" in item.keywords]

# Custom pytest command line options
def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="run integration tests only"
    )
    parser.addoption(
        "--performance",
        action="store_true",
        default=False,
        help="run performance tests only"
    )

if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__])