# Quick Start Guide

Get up and running with the Mini-RAG System in minutes!

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify data files exist:**
   - `data/handbook.txt`
   - `data/faq.txt`
   - `data/blog.txt`

## Quick Test

Run the test script to see the system in action:

```bash
python test_rag.py
```

This will:
- Load the text files
- Build the FAISS index
- Answer 5 sample questions
- Generate performance analysis
- Test reranking and different chunking modes

## Basic Usage

```python
from rag_system import MiniRAGSystem

# Create RAG system
rag = MiniRAGSystem()

# Load files
chunks = rag.load_and_chunk_files([
    "data/handbook.txt",
    "data/faq.txt",
    "data/blog.txt"
])

# Build index
rag.build_index(chunks)

# Ask a question
result = rag.answer_question("What is the main purpose?", top_k=5)
print(result['answer'])
```

## Common Use Cases

### 1. Fast Answers (Default)
```python
rag = MiniRAGSystem(
    qa_model_type="distilbert",
    use_reranking=False
)
```

### 2. Better Quality (Slower)
```python
rag = MiniRAGSystem(
    qa_model_type="t5",
    use_reranking=True
)
```

### 3. Custom Chunking
```python
rag = MiniRAGSystem(
    chunk_mode="sentences"  # or "tokens"
)
```

## Expected Output

When you run the test, you should see:

```
Question: What is the main purpose of this system?
Answer: The main purpose of this system is to provide a powerful and flexible 
        platform for managing and processing information...
Retrieved 5 chunks
```

## Troubleshooting

**First run is slow?**
- Models are being downloaded (one-time process)
- Subsequent runs will be faster

**Out of memory?**
- Use smaller embedding model: `all-MiniLM-L6-v2`
- Reduce chunk size in `TextChunker`

**Poor answers?**
- Enable reranking: `use_reranking=True`
- Increase `top_k` parameter
- Try different QA model: `qa_model_type="t5"`

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [example_usage.py](example_usage.py) for more examples
- Modify the code to use your own text files

## Support

If you encounter issues:
1. Check that all dependencies are installed
2. Verify data files exist in `data/` directory
3. Check Python version (3.8+ required)
4. Review error messages for specific issues

