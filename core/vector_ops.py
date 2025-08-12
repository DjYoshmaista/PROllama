# core/vector_ops.py
import math
import torch
import numpy as np
import logging
from typing import List, Union, Optional

logger = logging.getLogger(__name__)

class VectorOperations:
    """Centralized vector operations with GPU acceleration support"""
    
    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0

    @staticmethod
    def euclidean_distance(a: List[float], b: List[float]) -> float:
        """Compute Euclidean distance between two vectors"""
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    @staticmethod
    def is_zero_vector(vector: List[float]) -> bool:
        """Check if vector contains only zeros"""
        return all(x == 0 for x in vector)

    @staticmethod
    def normalize_vector(vector: List[float]) -> List[float]:
        """Normalize vector to unit length"""
        norm = math.sqrt(sum(x * x for x in vector))
        return [x / norm for x in vector] if norm > 0 else vector

    @staticmethod
    def gpu_cosine_similarity_batch(
        query_embedding: Union[List[float], np.ndarray], 
        db_embeddings_tensor: torch.Tensor
    ) -> np.ndarray:
        """
        Compute cosine similarity between a query vector and batch of vectors on GPU
        
        Args:
            query_embedding: Query embedding [dim]
            db_embeddings_tensor: Database embeddings [batch_size, dim] on GPU
            
        Returns:
            Cosine similarities [batch_size]
        """
        # Convert query to tensor and move to same device
        if isinstance(query_embedding, list):
            query_tensor = torch.tensor(query_embedding, dtype=torch.float32)
        else:
            query_tensor = torch.from_numpy(query_embedding).float()
            
        if db_embeddings_tensor.is_cuda:
            query_tensor = query_tensor.cuda()
        
        # Ensure query is 2D [1, dim] for matrix multiplication
        if query_tensor.dim() == 1:
            query_tensor = query_tensor.unsqueeze(0)
        
        # Normalize vectors
        query_norm = torch.nn.functional.normalize(query_tensor, p=2, dim=1)
        db_norm = torch.nn.functional.normalize(db_embeddings_tensor, p=2, dim=1)
        
        # Compute cosine similarity via dot product
        cos_sim = torch.mm(query_norm, db_norm.t())
        
        # Return as numpy array on CPU
        return cos_sim.squeeze().cpu().numpy()

    @staticmethod
    def batch_cosine_similarity(
        query_embedding: List[float], 
        embeddings: List[List[float]]
    ) -> List[float]:
        """
        Compute cosine similarity between query and multiple embeddings (CPU)
        
        Args:
            query_embedding: Query vector
            embeddings: List of embedding vectors
            
        Returns:
            List of similarity scores
        """
        return [
            VectorOperations.cosine_similarity(query_embedding, emb) 
            for emb in embeddings
        ]

    @staticmethod
    def validate_embedding_dimension(
        embedding: List[float], 
        expected_dim: int
    ) -> bool:
        """Validate embedding has correct dimensions"""
        return len(embedding) == expected_dim

    @staticmethod
    def create_gpu_tensor(embeddings: List[List[float]]) -> Optional[torch.Tensor]:
        """Create GPU tensor from embeddings if CUDA available"""
        try:
            tensor = torch.tensor(embeddings, dtype=torch.float32)
            if torch.cuda.is_available():
                tensor = tensor.cuda()
            return tensor
        except Exception as e:
            logger.error(f"Failed to create GPU tensor: {e}")
            return None

    @classmethod
    def similarity_search(
        cls,
        query_embedding: List[float],
        embeddings: Union[List[List[float]], torch.Tensor],
        top_k: int = 10,
        threshold: float = 0.0,
        use_gpu: bool = True
    ) -> List[tuple]:
        """
        Perform similarity search with optional GPU acceleration
        
        Args:
            query_embedding: Query vector
            embeddings: Database embeddings (list or tensor)
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            use_gpu: Whether to use GPU acceleration
            
        Returns:
            List of (index, similarity_score) tuples
        """
        if isinstance(embeddings, torch.Tensor) and use_gpu and torch.cuda.is_available():
            # GPU path
            similarities = cls.gpu_cosine_similarity_batch(query_embedding, embeddings)
            
            # Filter by threshold
            valid_indices = np.where(similarities >= threshold)[0]
            if len(valid_indices) == 0:
                return []
            
            # Get top-k
            valid_similarities = similarities[valid_indices]
            sorted_indices = valid_indices[np.argsort(valid_similarities)[::-1]]
            top_indices = sorted_indices[:top_k]
            
            return [(int(idx), float(similarities[idx])) for idx in top_indices]
        else:
            # CPU path
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy().tolist()
            
            similarities = cls.batch_cosine_similarity(query_embedding, embeddings)
            
            # Create (index, similarity) pairs and filter
            indexed_similarities = [
                (i, sim) for i, sim in enumerate(similarities) 
                if sim >= threshold
            ]
            
            # Sort by similarity (descending) and take top-k
            indexed_similarities.sort(key=lambda x: x[1], reverse=True)
            return indexed_similarities[:top_k]