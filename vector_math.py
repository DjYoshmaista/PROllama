# vector_math.py
import torch
import numpy as np
from gpu_utils import GPUMemoryManager
import logging

logger = logging.getLogger(__name__)

def batched_cosine_similarity(query_embedding, embeddings, batch_size=5000, use_gpu=None):
    """
    Unified cosine similarity calculation with automatic GPU/CPU selection
    
    Args:
        query_embedding: Query vector (list, numpy array, or tensor)
        embeddings: List/array of document embeddings
        batch_size: Size of batches for GPU processing
        use_gpu: Force GPU usage (None for auto-detection)
    
    Returns:
        numpy array of similarity scores
    """
    # Convert inputs to tensors
    if not isinstance(query_embedding, torch.Tensor):
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32)
    else:
        query_tensor = query_embedding.clone()
    
    if not isinstance(embeddings, torch.Tensor):
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    else:
        embeddings_tensor = embeddings.clone()
    
    # Auto-detect GPU usage
    if use_gpu is None:
        use_gpu = torch.cuda.is_available() and len(embeddings) > 100
    
    if use_gpu:
        return _gpu_batched_similarity(query_tensor, embeddings_tensor, batch_size)
    else:
        return _cpu_cosine_similarity(query_tensor, embeddings_tensor)

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