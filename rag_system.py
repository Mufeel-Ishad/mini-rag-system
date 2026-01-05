"""
Mini-RAG System: Retrieval-Augmented Generation using Hugging Face models and FAISS.

This module implements a complete RAG system with:
- Text chunking (300-500 tokens)
- Embedding generation using sentence-transformers
- FAISS-based similarity search
- Question answering using Hugging Face models
- Optional reranking and different chunking modes
"""

import os
import json
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
import faiss
import tiktoken
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    pipeline
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextChunker:
    """Handles text chunking with different strategies."""
    
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Target chunk size in tokens (default: 400)
            chunk_overlap: Overlap between chunks in tokens (default: 50)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
    
    def chunk_by_tokens(self, text: str) -> List[str]:
        """
        Split text into chunks based on token count.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            if i + self.chunk_size >= len(tokens):
                break
        
        return chunks
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """
        Split text into chunks by sentences, respecting token limits.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.encoding.encode(sentence))
            
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(len(self.encoding.encode(s)) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks


class EmbeddingGenerator:
    """Generates embeddings using Hugging Face sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformer model
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of embeddings
        """
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return np.array(embeddings).astype('float32')


class FAISSIndex:
    """Manages FAISS index for efficient similarity search."""
    
    def __init__(self, dimension: int):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Dimension of the embeddings
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance
        self.chunks = []
        self.metadata = []
    
    def add_embeddings(self, embeddings: np.ndarray, chunks: List[str], metadata: List[Dict] = None):
        """
        Add embeddings to the FAISS index.
        
        Args:
            embeddings: numpy array of embeddings
            chunks: List of text chunks
            metadata: Optional metadata for each chunk
        """
        if metadata is None:
            metadata = [{}] * len(chunks)
        
        # Normalize embeddings for cosine similarity (optional, but often improves results)
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(chunks)} chunks to FAISS index. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[str], List[float], List[Dict]]:
        """
        Search for the k most similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            Tuple of (chunks, distances, metadata)
        """
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        results = []
        result_distances = []
        result_metadata = []
        
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append(self.chunks[idx])
                result_distances.append(float(distances[0][i]))
                result_metadata.append(self.metadata[idx])
        
        return results, result_distances, result_metadata
    
    def save(self, filepath: str):
        """Save the FAISS index and metadata to disk."""
        faiss.write_index(self.index, filepath)
        metadata_file = filepath.replace('.index', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump({
                'chunks': self.chunks,
                'metadata': self.metadata
            }, f, indent=2)
        logger.info(f"Saved FAISS index to {filepath}")
    
    def load(self, filepath: str):
        """Load the FAISS index and metadata from disk."""
        self.index = faiss.read_index(filepath)
        metadata_file = filepath.replace('.index', '_metadata.json')
        with open(metadata_file, 'r') as f:
            data = json.load(f)
            self.chunks = data['chunks']
            self.metadata = data['metadata']
        logger.info(f"Loaded FAISS index from {filepath}")


class Reranker:
    """Reranks retrieved chunks based on relevance to the query."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the reranker.
        
        Args:
            model_name: Name of the cross-encoder model for reranking
        """
        logger.info(f"Loading reranker model: {model_name}")
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, chunks: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Rerank chunks based on relevance to the query.
        
        Args:
            query: Query string
            chunks: List of retrieved chunks
            top_k: Number of top chunks to return
            
        Returns:
            List of (chunk, score) tuples sorted by relevance
        """
        if not chunks:
            return []
        
        pairs = [[query, chunk] for chunk in chunks]
        scores = self.model.predict(pairs)
        
        # Sort by score (higher is better)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        
        return ranked[:top_k]


class QuestionAnswerer:
    """Answers questions using retrieved context and Hugging Face models."""
    
    def __init__(self, model_type: str = "distilbert", model_name: Optional[str] = None):
        """
        Initialize the question answerer.
        
        Args:
            model_type: Type of model ('distilbert', 't5', or 'bart')
            model_name: Specific model name (optional)
        """
        self.model_type = model_type.lower()
        
        if model_name is None:
            if self.model_type == "distilbert":
                model_name = "distilbert-base-uncased-distilled-squad"
            elif self.model_type == "t5":
                model_name = "t5-small"
            elif self.model_type == "bart":
                model_name = "facebook/bart-large"
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Loading {self.model_type} model: {model_name}")
        
        if self.model_type == "distilbert":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.model,
                tokenizer=self.tokenizer
            )
        elif self.model_type in ["t5", "bart"]:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def answer(self, question: str, context: str) -> str:
        """
        Generate an answer to a question given context.
        
        Args:
            question: Question string
            context: Context string (retrieved chunks)
            
        Returns:
            Answer string
        """
        if self.model_type == "distilbert":
            result = self.qa_pipeline(question=question, context=context)
            return result['answer']
        elif self.model_type == "t5":
            input_text = f"question: {question} context: {context}"
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.model.generate(**inputs, max_length=100)
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return answer
        elif self.model_type == "bart":
            input_text = f"question: {question} context: {context}"
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.model.generate(**inputs, max_length=100)
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return answer


class MiniRAGSystem:
    """Main RAG system that orchestrates all components."""
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        qa_model_type: str = "distilbert",
        chunk_mode: str = "tokens",
        use_reranking: bool = False
    ):
        """
        Initialize the RAG system.
        
        Args:
            embedding_model: Name of the embedding model
            qa_model_type: Type of QA model ('distilbert', 't5', or 'bart')
            chunk_mode: Chunking mode ('tokens' or 'sentences')
            use_reranking: Whether to use reranking
        """
        self.chunk_mode = chunk_mode
        self.use_reranking = use_reranking
        
        # Initialize components
        self.chunker = TextChunker()
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.question_answerer = QuestionAnswerer(qa_model_type)
        self.faiss_index = None
        self.reranker = Reranker() if use_reranking else None
        
        logger.info("Mini-RAG System initialized")
    
    def load_and_chunk_files(self, file_paths: List[str]) -> List[str]:
        """
        Load text files and chunk them.
        
        Args:
            file_paths: List of file paths to load
            
        Returns:
            List of text chunks
        """
        all_chunks = []
        
        for file_path in file_paths:
            logger.info(f"Loading file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if self.chunk_mode == "tokens":
                chunks = self.chunker.chunk_by_tokens(text)
            elif self.chunk_mode == "sentences":
                chunks = self.chunker.chunk_by_sentences(text)
            else:
                raise ValueError(f"Unknown chunk mode: {self.chunk_mode}")
            
            # Add metadata
            for i, chunk in enumerate(chunks):
                logger.debug(f"Chunk {i+1} from {file_path}: {len(chunk)} chars")
            
            all_chunks.extend(chunks)
            logger.info(f"Created {len(chunks)} chunks from {file_path}")
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def build_index(self, chunks: List[str], metadata: List[Dict] = None):
        """
        Build the FAISS index from chunks.
        
        Args:
            chunks: List of text chunks
            metadata: Optional metadata for each chunk
        """
        logger.info("Building FAISS index...")
        embeddings = self.embedding_generator.generate_embeddings(chunks)
        
        dimension = embeddings.shape[1]
        self.faiss_index = FAISSIndex(dimension)
        self.faiss_index.add_embeddings(embeddings, chunks, metadata)
        
        logger.info("FAISS index built successfully")
    
    def answer_question(self, question: str, top_k: int = 5) -> Dict:
        """
        Answer a question using the RAG system.
        
        Args:
            question: Question string
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer and metadata
        """
        if self.faiss_index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        logger.info(f"Answering question: {question}")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embeddings([question])[0]
        
        # Retrieve chunks
        chunks, distances, metadata = self.faiss_index.search(query_embedding, top_k)
        
        # Rerank if enabled
        if self.use_reranking and self.reranker and chunks:
            logger.info("Reranking retrieved chunks...")
            ranked_chunks = self.reranker.rerank(question, chunks, top_k=3)
            chunks = [chunk for chunk, score in ranked_chunks]
            distances = [score for chunk, score in ranked_chunks]
        
        # Combine context
        context = "\n\n".join(chunks)
        
        # Generate answer
        answer = self.question_answerer.answer(question, context)
        
        result = {
            'question': question,
            'answer': answer,
            'retrieved_chunks': chunks,
            'distances': distances,
            'num_chunks_used': len(chunks)
        }
        
        logger.info(f"Generated answer: {answer[:100]}...")
        return result
    
    def save_index(self, filepath: str):
        """Save the FAISS index to disk."""
        if self.faiss_index:
            self.faiss_index.save(filepath)
    
    def load_index(self, filepath: str):
        """Load the FAISS index from disk."""
        # We need to know the dimension, so we'll load metadata first
        metadata_file = filepath.replace('.index', '_metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                if data.get('chunks'):
                    # Get dimension from a sample embedding
                    sample_embedding = self.embedding_generator.generate_embeddings([data['chunks'][0]])[0]
                    dimension = len(sample_embedding)
                    self.faiss_index = FAISSIndex(dimension)
                    self.faiss_index.load(filepath)
        else:
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

