# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Global Token Clustering (GTC) Processor

Cross-frame token clustering using Soft K-Means.
Clusters all history tokens into a fixed number of output tokens.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Literal

from .base import HistoryProcessor


def soft_kmeans_step(
    tokens: torch.Tensor,
    centroids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Update centroids with one cosine-similarity soft k-means step."""
    dtype = tokens.dtype
    tokens_norm = F.normalize(tokens.float(), dim=-1)
    centroids_norm = F.normalize(centroids.float(), dim=-1)
    similarities = torch.mm(tokens_norm, centroids_norm.T)
    assignments = F.softmax(similarities / temperature, dim=-1)

    numerator = torch.mm(assignments.T.to(dtype), tokens)
    denominator = (
        assignments.sum(dim=0, keepdim=True)
        .T.to(dtype)
        .clamp(min=1e-6)
    )
    return numerator / denominator


class GlobalTokenClustering(HistoryProcessor):
    """
    Global Token Clustering (GTC) processor.
    
    Treats all history frame tokens as a single unordered set and
    clusters them into a fixed number of output tokens using Soft K-Means.
    
    Key features:
    - Cross-frame redundancy removal
    - Fixed output token count regardless of input
    - Best for high-redundancy scenarios (corridors, repeated scenes)
    
    Trade-offs:
    - Loses temporal order information
    - Best when task cares about "what" not "when"
    
    Algorithm:
    1. Concatenate all tokens: X ∈ R^{M×d}
    2. Initialize K centroids
    3. Soft Assignment: A = softmax(X·C^T / τ)
    4. Weighted Aggregation: C_new = normalize(A^T) · X
    5. Optional: Iterate steps 3-4
    """
    
    def __init__(
        self,
        output_tokens: int = 512,
        temperature: float = 0.1,
        num_iterations: int = 1,
        init_method: Literal['uniform', 'pooled'] = 'uniform',
    ):
        """
        Initialize GTC processor.
        
        Args:
            output_tokens: Fixed number of output tokens (K)
            temperature: Softmax temperature (lower = sharper assignment)
            num_iterations: Number of soft k-means iterations
            init_method: 'uniform' (evenly spaced) or 'pooled' (avg pooling)
        """
        self.output_tokens = output_tokens
        self.temperature = temperature
        self.num_iterations = num_iterations
        self.init_method = init_method
    
    @property
    def name(self) -> str:
        return f"GTC(K={self.output_tokens}, τ={self.temperature}, iter={self.num_iterations})"
    
    def get_output_token_count(
        self,
        num_frames: int,
        frame_infos: List[Tuple[int, int, int]],
    ) -> int:
        """GTC outputs fixed tokens, capped by total input."""
        total_input = sum(t * h * w for t, h, w in frame_infos)
        return min(self.output_tokens, total_input)
    
    def process(
        self,
        frame_embeds_list: List[torch.Tensor],
        frame_grid_thws: List[torch.Tensor],
    ) -> torch.Tensor:
        """Cluster all history tokens into fixed output tokens."""
        if not frame_embeds_list:
            return torch.empty(0, 0)
        
        # Concatenate all frame tokens
        X = torch.cat(frame_embeds_list, dim=0)
        M, d = X.shape
        
        K = min(self.output_tokens, M)
        if K >= M:
            return X  # No compression needed
        
        # Initialize centroids
        C = self._initialize_centroids(X, K)
        
        # Soft K-Means iteration(s)
        for _ in range(self.num_iterations):
            C = soft_kmeans_step(X, C, self.temperature)
        
        return C
    
    def _initialize_centroids(self, X: torch.Tensor, K: int) -> torch.Tensor:
        """Initialize K centroids from input tokens."""
        M, d = X.shape
        
        if self.init_method == 'uniform':
            indices = torch.linspace(0, M - 1, K, dtype=torch.long, device=X.device)
            return X[indices].clone()
        
        elif self.init_method == 'pooled':
            X_pool = X.T.unsqueeze(0)  # [1, d, M]
            pool_size = M // K
            if pool_size > 1:
                pooled = F.avg_pool1d(X_pool.float(), kernel_size=pool_size, stride=pool_size)
                pooled = pooled.squeeze(0).T  # [K', d]
                if pooled.shape[0] >= K:
                    return pooled[:K].to(X.dtype)
                else:
                    extra_needed = K - pooled.shape[0]
                    extra_indices = torch.linspace(0, M - 1, extra_needed, dtype=torch.long, device=X.device)
                    return torch.cat([pooled.to(X.dtype), X[extra_indices]], dim=0)
            else:
                indices = torch.linspace(0, M - 1, K, dtype=torch.long, device=X.device)
                return X[indices].clone()
        
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")
