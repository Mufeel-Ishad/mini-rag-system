# Mini-RAG System

A complete Retrieval-Augmented Generation (RAG) system built with Hugging Face models and FAISS for efficient similarity search and question answering.

## Overview

This system implements a full RAG pipeline that:
1. Loads and chunks text documents (300-500 tokens per chunk)
2. Generates embeddings using Hugging Face's sentence-transformers
3. Stores embeddings in FAISS for fast similarity search
4. Answers questions by retrieving relevant chunks and using LLMs (DistilBERT, T5, or BART)
5. Supports optional reranking and different chunking modes

## Features

- **Text Chunking**: Multiple chunking strategies (token-based, sentence-based)
- **Embedding Generation**: Uses sentence-transformers for high-quality embeddings
- **FAISS Indexing**: Fast similarity search using FAISS
- **Question Answering**: Supports multiple models (DistilBERT, T5, BART)
- **Reranking**: Optional cross-encoder reranking for improved accuracy
- **Flexible Configuration**: Toggle between different modes and models

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda

### Setup

1. Clone or navigate to the project directory:
```bash
cd mini-rag-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the sample text files in the `data/` directory:
   - `handbook.txt`
   - `faq.txt`
   - `blog.txt`

## Project Structure

```
mini-rag-system/
├── rag_system.py          # Main RAG system implementation
├── main.py                # Main script to run the system
├── test_rag.py            # Test script with sample questions
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── data/                  # Text files directory
    ├── handbook.txt
    ├── faq.txt
    └── blog.txt
```

## Usage

### Basic Usage

Run the main script to build the index and answer questions:

```bash
python main.py
```

### Testing

Run the test script to evaluate the system with sample questions:

```bash
python test_rag.py
```

### Programmatic Usage

```python
from rag_system import MiniRAGSystem

# Initialize the RAG system
rag = MiniRAGSystem(
    embedding_model="all-MiniLM-L6-v2",
    qa_model_type="distilbert",
    chunk_mode="tokens",
    use_reranking=False
)

# Load and chunk files
chunks = rag.load_and_chunk_files([
    "data/handbook.txt",
    "data/faq.txt",
    "data/blog.txt"
])

# Build the FAISS index
rag.build_index(chunks)

# Answer a question
result = rag.answer_question("What is the main purpose of this system?", top_k=5)
print(result['answer'])
```

## Configuration Options

### Embedding Models

- `all-MiniLM-L6-v2` (default): Fast and efficient, good quality
- `all-mpnet-base-v2`: Higher quality, slower
- `sentence-transformers/all-MiniLM-L12-v2`: Balanced option

### QA Models

- `distilbert` (default): Fast, good for most use cases
- `t5`: Better quality, slower
- `bart`: Best quality, slowest

### Chunking Modes

- `tokens`: Split by token count (default)
- `sentences`: Split by sentences, respecting token limits

### Reranking

Set `use_reranking=True` to enable cross-encoder reranking for improved accuracy.

## Components

### TextChunker

Handles text chunking with different strategies:
- Token-based chunking
- Sentence-based chunking with overlap

### EmbeddingGenerator

Generates embeddings using Hugging Face sentence-transformers:
- Supports various pre-trained models
- Efficient batch processing

### FAISSIndex

Manages FAISS index for similarity search:
- L2 distance for similarity
- Normalized embeddings
- Metadata storage

### Reranker

Optional reranking using cross-encoder models:
- Scores chunks based on query relevance
- Improves retrieval accuracy

### QuestionAnswerer

Answers questions using retrieved context:
- Supports multiple model types
- Handles context length limits

## Performance Analysis

The test script (`test_rag.py`) provides detailed performance metrics:

- Success rate
- Average answer length
- Average chunks used
- Retrieval distances
- Comparison with/without reranking
- Comparison of chunking modes

Results are saved to a JSON file with timestamp for analysis.

## Example Output

```
Q: What is the main purpose of this system?
A: The main purpose of this system is to provide a powerful and flexible 
   platform for managing and processing information. It is designed to help 
   users organize their work, collaborate effectively, and achieve their 
   goals with maximum efficiency.
Retrieved 5 chunks
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce chunk size or use a smaller embedding model
2. **Slow Performance**: Use `distilbert` instead of `t5` or `bart`
3. **Poor Answers**: Enable reranking or increase `top_k` parameter
4. **Model Download**: First run will download models (may take time)

### Performance Tips

- Use `all-MiniLM-L6-v2` for faster embeddings
- Use `distilbert` for faster QA
- Enable reranking only when needed (slower but more accurate)
- Adjust chunk size based on your documents

## Advanced Features

### Saving and Loading Index

```python
# Save index
rag.save_index("faiss_index.index")

# Load index
rag.load_index("faiss_index.index")
```

### Custom Chunking

Modify `TextChunker` parameters:
```python
chunker = TextChunker(chunk_size=500, chunk_overlap=100)
```

### Different Models

```python
# Use T5 for better quality
rag = MiniRAGSystem(qa_model_type="t5")

# Use better embedding model
rag = MiniRAGSystem(embedding_model="all-mpnet-base-v2")
```

## Limitations

- First-time model downloads can be slow
- Large documents may require significant memory
- Real-time performance depends on hardware
- Answer quality depends on retrieval quality

## Future Improvements

- Support for more document formats (PDF, DOCX)
- Web interface for interactive Q&A
- Fine-tuning on domain-specific data
- Multi-query retrieval strategies
- Hybrid search (keyword + semantic)

## License

This project is provided as-is for educational and demonstration purposes.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for improvements.

## Acknowledgments

- Hugging Face for transformers and sentence-transformers
- Facebook AI Research for FAISS
- OpenAI for tiktoken

