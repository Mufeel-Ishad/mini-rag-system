"""
Test script for the Mini-RAG System.

This script tests the system with 5 sample questions and logs the results
for performance analysis.
"""

import logging
from pathlib import Path
from rag_system import MiniRAGSystem
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_rag_system():
    """Test the RAG system with sample questions."""
    
    # Initialize RAG system
    logger.info("Initializing RAG System...")
    rag = MiniRAGSystem(
        embedding_model="all-MiniLM-L6-v2",
        qa_model_type="distilbert",
        chunk_mode="tokens",
        use_reranking=False
    )
    
    # Load and chunk files
    data_dir = Path("data")
    text_files = [
        str(data_dir / "handbook.txt"),
        str(data_dir / "faq.txt"),
        str(data_dir / "blog.txt")
    ]
    
    logger.info("Loading and chunking text files...")
    chunks = rag.load_and_chunk_files(text_files)
    logger.info(f"Created {len(chunks)} chunks")
    
    # Build index
    logger.info("Building FAISS index...")
    rag.build_index(chunks)
    
    # Test questions
    test_questions = [
        "What is the main purpose of this system?",
        "How do I get started with the system?",
        "What are the key features available?",
        "What should I do if I encounter an error?",
        "What are the best practices for using the system?"
    ]
    
    # Answer questions and collect results
    results = []
    
    logger.info("\n" + "=" * 80)
    logger.info("TESTING WITH SAMPLE QUESTIONS")
    logger.info("=" * 80)
    
    for i, question in enumerate(test_questions, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Question {i}/{len(test_questions)}")
        logger.info(f"{'='*80}")
        logger.info(f"Q: {question}")
        
        try:
            result = rag.answer_question(question, top_k=5)
            results.append(result)
            
            logger.info(f"\nAnswer: {result['answer']}")
            logger.info(f"Number of chunks retrieved: {result['num_chunks_used']}")
            logger.info(f"Average distance: {sum(result['distances']) / len(result['distances']) if result['distances'] else 0:.4f}")
            
            # Analyze answer quality
            answer_length = len(result['answer'])
            if answer_length < 10:
                logger.warning("⚠️  Answer is very short - may indicate poor retrieval")
            elif answer_length > 1000:
                logger.info("✓ Answer is comprehensive")
            else:
                logger.info("✓ Answer length is reasonable")
            
            # Show top retrieved chunks
            logger.info("\nTop retrieved chunks:")
            for j, (chunk, distance) in enumerate(zip(result['retrieved_chunks'][:3], result['distances'][:3]), 1):
                logger.info(f"  Chunk {j} (distance: {distance:.4f}): {chunk[:100]}...")
        
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            results.append({
                'question': question,
                'answer': f"ERROR: {str(e)}",
                'error': True
            })
    
    # Performance analysis
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE ANALYSIS")
    logger.info("=" * 80)
    
    successful_answers = [r for r in results if not r.get('error', False)]
    failed_answers = [r for r in results if r.get('error', False)]
    
    logger.info(f"\nTotal questions: {len(test_questions)}")
    logger.info(f"Successful answers: {len(successful_answers)}")
    logger.info(f"Failed answers: {len(failed_answers)}")
    logger.info(f"Success rate: {len(successful_answers) / len(test_questions) * 100:.1f}%")
    
    if successful_answers:
        avg_answer_length = sum(len(r['answer']) for r in successful_answers) / len(successful_answers)
        avg_chunks_used = sum(r['num_chunks_used'] for r in successful_answers) / len(successful_answers)
        avg_distance = sum(
            sum(r['distances']) / len(r['distances']) if r['distances'] else 0
            for r in successful_answers
        ) / len(successful_answers)
        
        logger.info(f"\nAverage answer length: {avg_answer_length:.0f} characters")
        logger.info(f"Average chunks used: {avg_chunks_used:.1f}")
        logger.info(f"Average retrieval distance: {avg_distance:.4f}")
    
    # Detailed results
    logger.info("\n" + "=" * 80)
    logger.info("DETAILED RESULTS")
    logger.info("=" * 80)
    
    for i, result in enumerate(results, 1):
        logger.info(f"\n--- Result {i} ---")
        logger.info(f"Question: {result['question']}")
        if result.get('error'):
            logger.error(f"Status: FAILED - {result['answer']}")
        else:
            logger.info(f"Status: SUCCESS")
            logger.info(f"Answer: {result['answer']}")
            logger.info(f"Chunks used: {result['num_chunks_used']}")
            logger.info(f"Distances: {[f'{d:.4f}' for d in result['distances']]}")
    
    # Save results to file
    output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'test_questions': test_questions,
            'results': results,
            'summary': {
                'total_questions': len(test_questions),
                'successful': len(successful_answers),
                'failed': len(failed_answers),
                'success_rate': len(successful_answers) / len(test_questions) * 100 if test_questions else 0
            }
        }, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    
    # Test with reranking
    logger.info("\n" + "=" * 80)
    logger.info("TESTING WITH RERANKING ENABLED")
    logger.info("=" * 80)
    
    rag_rerank = MiniRAGSystem(
        embedding_model="all-MiniLM-L6-v2",
        qa_model_type="distilbert",
        chunk_mode="tokens",
        use_reranking=True
    )
    
    rag_rerank.build_index(chunks)
    
    sample_question = test_questions[0]
    logger.info(f"\nQuestion: {sample_question}")
    
    result_no_rerank = rag.answer_question(sample_question, top_k=5)
    result_with_rerank = rag_rerank.answer_question(sample_question, top_k=5)
    
    logger.info(f"\nAnswer (no reranking): {result_no_rerank['answer']}")
    logger.info(f"Answer (with reranking): {result_with_rerank['answer']}")
    
    # Test different chunking modes
    logger.info("\n" + "=" * 80)
    logger.info("TESTING DIFFERENT CHUNKING MODES")
    logger.info("=" * 80)
    
    for chunk_mode in ["tokens", "sentences"]:
        logger.info(f"\nTesting chunk mode: {chunk_mode}")
        rag_chunk = MiniRAGSystem(
            embedding_model="all-MiniLM-L6-v2",
            qa_model_type="distilbert",
            chunk_mode=chunk_mode,
            use_reranking=False
        )
        
        chunks_chunk = rag_chunk.load_and_chunk_files(text_files)
        rag_chunk.build_index(chunks_chunk)
        
        result_chunk = rag_chunk.answer_question(sample_question, top_k=5)
        logger.info(f"Answer ({chunk_mode} mode): {result_chunk['answer'][:200]}...")
        logger.info(f"Number of chunks created: {len(chunks_chunk)}")


if __name__ == "__main__":
    test_rag_system()

