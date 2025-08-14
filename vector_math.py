# vector_math.py - Optimized vector operations
import torch
import numpy as np
from gpu_utils import GPUMemoryManager
from performance_optimizer import vector_optimizer, performance_profiler
import logging
from typing import Union, List

logger = logging.getLogger(__name__)

def batched_cosine_similarity(
    query_embedding: Union[List[float], np.ndarray, torch.Tensor], 
    embeddings: Union[List[List[float]], np.ndarray, torch.Tensor], 
    batch_size: int = 5000, 
    use_gpu: bool = None
) -> np.ndarray:
    """
    Optimized cosine similarity calculation with automatic GPU/CPU selection
    
    Args:
        query_embedding: Query vector (list, numpy array, or tensor)
        embeddings: List/array of document embeddings
        batch_size: Size of batches for GPU processing
        use_gpu: Force GPU usage (None for auto-detection)
    
    Returns:
        numpy array of similarity scores
    """
    with performance_profiler.profile_operation("cosine_similarity"):
        # Convert inputs to numpy arrays for consistency
        if isinstance(query_embedding, (list, tuple)):
            query_array = np.array(query_embedding, dtype=np.float32)
        elif isinstance(query_embedding, torch.Tensor):
            query_array = query_embedding.cpu().numpy().astype(np.float32)
        else:
            query_array = query_embedding.astype(np.float32)
        
        if isinstance(embeddings, (list, tuple)):
            embeddings_array = np.array(embeddings, dtype=np.float32)
        elif isinstance(embeddings, torch.Tensor):
            embeddings_array = embeddings.cpu().numpy().astype(np.float32)
        else:
            embeddings_array = embeddings.astype(np.float32)
        
        # Optimize storage format
        embeddings_array = vector_optimizer.optimize_embeddings_storage(embeddings_array)
        
        # Use optimized similarity calculation
        return vector_optimizer.optimize_batch_cosine_similarity(
            query_array, embeddings_array, batch_size, use_gpu
        )

def _gpu_batched_similarity(query_tensor, embeddings_tensor, batch_size):
    """GPU-accelerated batched similarity calculation"""
    with GPUMemoryManager():
        query_tensor = query_tensor.cuda()
        results = []
        
        for i in range(0, len(embeddings_tensor), batch_size):
            batch = embeddings_tensor[i:i+batch_size].cuda()
            
            # Normalize vectors
            query_norm = torch.nn.functional.normalize(query_tensor.unsqueeze(0), p=2, dim=1)
            batch_norm = torch.nn.functional.normalize(batch, p=2, dim=1)
            
            # Compute similarity
            similarities = torch.mm(query_norm, batch_norm.t()).squeeze()
            results.append(similarities.cpu())
            
            # Clean up batch from GPU
            del batch
        
        return torch.cat(results).numpy()

def _cpu_cosine_similarity(query_tensor, embeddings_tensor):
    """CPU-based cosine similarity calculation"""
    query_np = query_tensor.numpy()
    embeddings_np = embeddings_tensor.numpy()
    
    # Normalize vectors
    query_norm = query_np / np.linalg.norm(query_np)
    embeddings_norm = embeddings_np / np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    
    # Compute dot product (cosine similarity for normalized vectors)
    return np.dot(embeddings_norm, query_norm)

def euclidean_distance(a, b):
    """Compute Euclidean distance between vectors"""
    if isinstance(a, torch.Tensor):
        a = a.cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.cpu().numpy()
    return np.linalg.norm(np.array(a) - np.array(b))