"""
Main script to run the Mini-RAG System.

This script demonstrates:
1. Loading text files
2. Building the FAISS index
3. Answering questions
4. Testing with sample questions
"""

import logging
from pathlib import Path
from rag_system import MiniRAGSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the RAG system."""
    
    # Initialize the RAG system
    # Options:
    # - embedding_model: "all-MiniLM-L6-v2" (fast, good quality)
    # - qa_model_type: "distilbert" (fast), "t5" (slower, better), "bart" (best, slowest)
    # - chunk_mode: "tokens" or "sentences"
    # - use_reranking: True/False (improves accuracy but slower)
    
    rag = MiniRAGSystem(
        embedding_model="all-MiniLM-L6-v2",
        qa_model_type="distilbert",
        chunk_mode="tokens",
        use_reranking=False  # Set to True for better accuracy
    )
    
    # Step 1: Load and chunk text files
    data_dir = Path("data")
    text_files = [
        str(data_dir / "handbook.txt"),
        str(data_dir / "faq.txt"),
        str(data_dir / "blog.txt")
    ]
    
    logger.info("=" * 60)
    logger.info("Step 1: Loading and chunking text files")
    logger.info("=" * 60)
    
    chunks = rag.load_and_chunk_files(text_files)
    
    # Step 2 & 3: Generate embeddings and build FAISS index
    logger.info("=" * 60)
    logger.info("Step 2 & 3: Generating embeddings and building FAISS index")
    logger.info("=" * 60)
    
    rag.build_index(chunks)
    
    # Save index for future use
    rag.save_index("faiss_index.index")
    
    # Step 4: Answer questions
    logger.info("=" * 60)
    logger.info("Step 4: Answering questions")
    logger.info("=" * 60)
    
    # Sample questions for testing
    test_questions = [
        "What is the main purpose of this system?",
        "How do I get started?",
        "What are the key features?",
        "What should I do if I encounter an error?",
        "What are the best practices?"
    ]
    
    results = []
    
    for i, question in enumerate(test_questions, 1):
        logger.info(f"\n--- Question {i}/{len(test_questions)} ---")
        result = rag.answer_question(question, top_k=5)
        results.append(result)
        
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"Retrieved {result['num_chunks_used']} chunks")
        print("-" * 60)
    
    # Step 5: Log results and analysis
    logger.info("=" * 60)
    logger.info("Step 5: Performance Analysis")
    logger.info("=" * 60)
    
    log_results(results)
    
    # Demonstrate reranking toggle
    logger.info("\n" + "=" * 60)
    logger.info("Demonstrating reranking mode")
    logger.info("=" * 60)
    
    rag_rerank = MiniRAGSystem(
        embedding_model="all-MiniLM-L6-v2",
        qa_model_type="distilbert",
        chunk_mode="tokens",
        use_reranking=True
    )
    rag_rerank.build_index(chunks)
    
    sample_question = "What is the main purpose of this system?"
    result_with_rerank = rag_rerank.answer_question(sample_question, top_k=5)
    
    print(f"\nQ: {sample_question}")
    print(f"A (with reranking): {result_with_rerank['answer']}")
    print(f"Retrieved {result_with_rerank['num_chunks_used']} chunks")


def log_results(results):
    """Log and analyze the results."""
    logger.info(f"\nTotal questions answered: {len(results)}")
    
    for i, result in enumerate(results, 1):
        logger.info(f"\nQuestion {i}:")
        logger.info(f"  Question: {result['question']}")
        logger.info(f"  Answer length: {len(result['answer'])} characters")
        logger.info(f"  Chunks used: {result['num_chunks_used']}")
        logger.info(f"  Average distance: {sum(result['distances']) / len(result['distances']) if result['distances'] else 0:.4f}")
        
        # Check if answer is meaningful (not empty, not too short)
        if len(result['answer']) < 10:
            logger.warning(f"  ⚠️  Answer might be too short or empty")
        elif len(result['answer']) > 500:
            logger.info(f"  ✓ Answer is comprehensive")
        else:
            logger.info(f"  ✓ Answer looks reasonable")


if __name__ == "__main__":
    main()

