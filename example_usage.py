"""
Example usage script demonstrating different features of the Mini-RAG System.

This script shows:
- Basic usage
- Using different models
- Toggling reranking
- Different chunking modes
"""

from rag_system import MiniRAGSystem
from pathlib import Path


def example_basic_usage():
    """Basic usage example."""
    print("\n" + "="*60)
    print("Example 1: Basic Usage")
    print("="*60)
    
    # Initialize RAG system with default settings
    rag = MiniRAGSystem(
        embedding_model="all-MiniLM-L6-v2",
        qa_model_type="distilbert",
        chunk_mode="tokens",
        use_reranking=False
    )
    
    # Load files
    data_dir = Path("data")
    text_files = [
        str(data_dir / "handbook.txt"),
        str(data_dir / "faq.txt"),
        str(data_dir / "blog.txt")
    ]
    
    chunks = rag.load_and_chunk_files(text_files)
    rag.build_index(chunks)
    
    # Ask a question
    question = "What is the main purpose of this system?"
    result = rag.answer_question(question, top_k=3)
    
    print(f"\nQuestion: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Chunks used: {result['num_chunks_used']}")


def example_with_reranking():
    """Example with reranking enabled."""
    print("\n" + "="*60)
    print("Example 2: With Reranking")
    print("="*60)
    
    rag = MiniRAGSystem(
        embedding_model="all-MiniLM-L6-v2",
        qa_model_type="distilbert",
        chunk_mode="tokens",
        use_reranking=True  # Enable reranking
    )
    
    data_dir = Path("data")
    text_files = [
        str(data_dir / "handbook.txt"),
        str(data_dir / "faq.txt"),
        str(data_dir / "blog.txt")
    ]
    
    chunks = rag.load_and_chunk_files(text_files)
    rag.build_index(chunks)
    
    question = "How do I get started?"
    result = rag.answer_question(question, top_k=5)
    
    print(f"\nQuestion: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Chunks used (after reranking): {result['num_chunks_used']}")


def example_different_chunking():
    """Example with different chunking modes."""
    print("\n" + "="*60)
    print("Example 3: Different Chunking Modes")
    print("="*60)
    
    data_dir = Path("data")
    text_files = [str(data_dir / "handbook.txt")]
    
    # Token-based chunking
    rag_tokens = MiniRAGSystem(
        embedding_model="all-MiniLM-L6-v2",
        qa_model_type="distilbert",
        chunk_mode="tokens",
        use_reranking=False
    )
    chunks_tokens = rag_tokens.load_and_chunk_files(text_files)
    rag_tokens.build_index(chunks_tokens)
    
    # Sentence-based chunking
    rag_sentences = MiniRAGSystem(
        embedding_model="all-MiniLM-L6-v2",
        qa_model_type="distilbert",
        chunk_mode="sentences",
        use_reranking=False
    )
    chunks_sentences = rag_sentences.load_and_chunk_files(text_files)
    rag_sentences.build_index(chunks_sentences)
    
    question = "What are the key features?"
    
    result_tokens = rag_tokens.answer_question(question, top_k=3)
    result_sentences = rag_sentences.answer_question(question, top_k=3)
    
    print(f"\nQuestion: {question}")
    print(f"\nToken-based chunking:")
    print(f"  Chunks created: {len(chunks_tokens)}")
    print(f"  Answer: {result_tokens['answer'][:150]}...")
    print(f"\nSentence-based chunking:")
    print(f"  Chunks created: {len(chunks_sentences)}")
    print(f"  Answer: {result_sentences['answer'][:150]}...")


def example_save_and_load():
    """Example of saving and loading the index."""
    print("\n" + "="*60)
    print("Example 4: Save and Load Index")
    print("="*60)
    
    # Build and save index
    rag = MiniRAGSystem()
    data_dir = Path("data")
    text_files = [str(data_dir / "handbook.txt")]
    
    chunks = rag.load_and_chunk_files(text_files)
    rag.build_index(chunks)
    rag.save_index("example_index.index")
    print("Index saved to example_index.index")
    
    # Load index in a new instance
    rag_new = MiniRAGSystem()
    rag_new.load_index("example_index.index")
    print("Index loaded successfully")
    
    question = "What is the main purpose?"
    result = rag_new.answer_question(question, top_k=3)
    print(f"\nQuestion: {question}")
    print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    print("Mini-RAG System - Example Usage")
    print("="*60)
    
    try:
        example_basic_usage()
        example_with_reranking()
        example_different_chunking()
        example_save_and_load()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

