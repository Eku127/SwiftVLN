# Copyright (c) Alibaba, Inc. and its affiliates.
"""
History Token Compressor for VLN

This module implements compression methods for history frames in Visual Language
Navigation tasks. Supports multiple compression strategies:
- 2D Average Pooling (default): Fast, simple spatial downsampling
- ToMe (Grid-based Soft K-Means): Better semantic preservation for small objects

Each method has its own token count calculation, allowing flexible compression ratios.

Used by: MonoVLN, CompressVLN, SwiftVLN
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Literal


class HistoryTokenCompressor:
    """
    Compressor for history frame visual tokens.
    
    Supports multiple compression methods:
    - 'pooling': 2D Average Pooling (fast, default)
    - 'tome': Grid-based Soft K-Means Token Merging (better semantic preservation)
    
    Each method has its own token count calculation via `get_compressed_token_count()`.
    This allows the template's `_encode` phase to pre-allocate the correct number
    of placeholder tokens for each compression method.
    
    Args:
        stride: Compression stride (default: 2). A stride of 2 results in 4x compression
                for pooling method.
        method: Compression method ('pooling' or 'tome'). Default: 'pooling'
        grid_size: Grid size for ToMe (default: 2, splits into 2x2 regions)
    """
    
    def __init__(
        self, 
        stride: int = 2, 
        method: Literal['pooling', 'tome'] = 'pooling',
        grid_size: int = 2
    ):
        self.stride = stride
        self.method = method
        self.grid_size = grid_size
    
    def get_compressed_token_count(
        self,
        t: int,
        h: int,
        w: int,
        stride: int = None
    ) -> int:
        """
        Calculate the number of tokens after compression.
        
        This is the KEY function that allows different compression methods to have
        different output token counts. Called by template's `_encode` to determine
        how many placeholder tokens to create.
        
        Args:
            t: Temporal dimension (usually 1 for images)
            h: Height after ViT spatial merge
            w: Width after ViT spatial merge
            stride: Compression stride (if None, uses self.stride)
            
        Returns:
            Number of tokens after compression
        """
        if stride is None:
            stride = self.stride
        
        t, h, w = int(t), int(h), int(w)
        
        if self.method == 'tome':
            return self._get_tome_token_count(t, h, w, stride)
        else:
            return self._get_pool_token_count(t, h, w, stride)
    
    def _get_pool_token_count(self, t: int, h: int, w: int, stride: int) -> int:
        """Calculate token count for pooling method."""
        new_h = h // stride
        new_w = w // stride
        return t * new_h * new_w
    
    def _get_tome_token_count(self, t: int, h: int, w: int, stride: int) -> int:
        """
        Calculate token count for ToMe (grid-based soft k-means) method.
        
        The image is split into grid_size x grid_size regions, and each region
        is compressed independently using soft k-means.
        """
        total_tokens = 0
        
        # Base cell dimensions
        base_cell_h = h // self.grid_size
        base_cell_w = w // self.grid_size
        
        for gi in range(self.grid_size):
            for gj in range(self.grid_size):
                # Calculate actual cell dimensions (last cells may be larger)
                if gi < self.grid_size - 1:
                    cell_h = base_cell_h
                else:
                    cell_h = h - gi * base_cell_h
                
                if gj < self.grid_size - 1:
                    cell_w = base_cell_w
                else:
                    cell_w = w - gj * base_cell_w
                
                # Tokens after compression in this cell
                out_h = cell_h // stride
                out_w = cell_w // stride
                
                # Ensure at least 1 token per cell
                out_h = max(1, out_h)
                out_w = max(1, out_w)
                
                total_tokens += t * out_h * out_w
        
        return total_tokens
    
    def compress(
        self,
        image_embeds: torch.Tensor,
        grid_thw: torch.Tensor,
        stride: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress image embeddings using the configured method.
        
        Args:
            image_embeds: Image embeddings [num_tokens, hidden_size]
            grid_thw: Grid dimensions [t, h, w] - temporal, height, width
            stride: Compression stride (if None, uses self.stride)
            
        Returns:
            compressed_embeds: Compressed embeddings
            new_grid_thw: New grid dimensions after compression
        """
        if self.method == 'tome':
            return self.compress_grid_tome(image_embeds, grid_thw, stride)
        else:
            return self.compress_2d_pool(image_embeds, grid_thw, stride)
    
    def compress_2d_pool(
        self, 
        image_embeds: torch.Tensor, 
        grid_thw: torch.Tensor,
        stride: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress image embeddings using 2D average pooling.
        
        Args:
            image_embeds: Image embeddings [num_tokens, hidden_size]
            grid_thw: Grid dimensions [t, h, w] - temporal, height, width
            stride: Pooling stride (if None, uses self.stride)
        
        Returns:
            compressed_embeds: Compressed embeddings [compressed_tokens, hidden_size]
            new_grid_thw: New grid dimensions after compression [t, h//stride, w//stride]
        """
        if stride is None:
            stride = self.stride
        
        t, h, w = grid_thw[0].item(), grid_thw[1].item(), grid_thw[2].item()
        hidden_size = image_embeds.shape[-1]
        
        # Reshape from 1D sequence to 2D spatial layout
        # [num_tokens, hidden_size] -> [t, h, w, hidden_size]
        embeds_2d = image_embeds.view(t, h, w, hidden_size)
        
        # Permute to [t, hidden_size, h, w] for avg_pool2d
        embeds_2d = embeds_2d.permute(0, 3, 1, 2).contiguous()
        
        # Apply 2D average pooling
        # [t, hidden_size, h, w] -> [t, hidden_size, h//stride, w//stride]
        compressed = F.avg_pool2d(
            embeds_2d, 
            kernel_size=stride, 
            stride=stride
        )
        
        # Permute back and flatten
        # [t, hidden_size, h//stride, w//stride] -> [t, h//stride, w//stride, hidden_size]
        compressed = compressed.permute(0, 2, 3, 1).contiguous()
        
        # Flatten to 1D sequence
        # [t, h//stride, w//stride, hidden_size] -> [t * h//stride * w//stride, hidden_size]
        compressed = compressed.reshape(-1, hidden_size)
        
        # Calculate new grid dimensions
        new_h = h // stride
        new_w = w // stride
        new_grid_thw = torch.tensor([t, new_h, new_w], dtype=grid_thw.dtype, device=grid_thw.device)
        
        return compressed, new_grid_thw
    
    def compress_grid_tome(
        self,
        image_embeds: torch.Tensor,
        grid_thw: torch.Tensor,
        stride: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress image embeddings using Grid-based Soft K-Means Token Merging.
        
        This method:
        1. Splits the image into grid_size x grid_size regions
        2. Applies Soft K-Means within each region
        3. Concatenates results to preserve spatial structure
        
        Benefits over pooling:
        - Better semantic preservation for small objects
        - Maintains spatial structure (left/right/top/bottom)
        - Uses attention-weighted aggregation instead of simple averaging
        
        Args:
            image_embeds: Image embeddings [num_tokens, hidden_size]
            grid_thw: Grid dimensions [t, h, w] - temporal, height, width
            stride: Compression stride (if None, uses self.stride)
            
        Returns:
            compressed_embeds: Compressed embeddings
            new_grid_thw: New grid dimensions (approximate, for reference)
        """
        if stride is None:
            stride = self.stride
        
        t, h, w = grid_thw[0].item(), grid_thw[1].item(), grid_thw[2].item()
        t, h, w = int(t), int(h), int(w)
        hidden_size = image_embeds.shape[-1]
        device = image_embeds.device
        dtype = image_embeds.dtype
        
        # Reshape to spatial layout: [num_tokens, C] -> [t, h, w, C]
        x = image_embeds.view(t, h, w, hidden_size)
        
        # Base cell dimensions
        base_cell_h = h // self.grid_size
        base_cell_w = w // self.grid_size
        
        # Check if grid cells are too small
        if base_cell_h // stride < 1 or base_cell_w // stride < 1:
            # Fall back to pooling if grid cells too small
            return self.compress_2d_pool(image_embeds, grid_thw, stride)
        
        compressed_cells = []
        total_out_h = 0
        total_out_w = 0
        
        for gi in range(self.grid_size):
            row_cells = []
            for gj in range(self.grid_size):
                # Calculate cell boundaries (last cells may be larger)
                h_start = gi * base_cell_h
                h_end = h if gi == self.grid_size - 1 else (gi + 1) * base_cell_h
                w_start = gj * base_cell_w
                w_end = w if gj == self.grid_size - 1 else (gj + 1) * base_cell_w
                
                cell_h = h_end - h_start
                cell_w = w_end - w_start
                
                # Extract grid cell: [t, cell_h, cell_w, C]
                cell = x[:, h_start:h_end, w_start:w_end, :]
                cell_flat = cell.reshape(-1, hidden_size)  # [t*cell_h*cell_w, C]
                
                # Apply soft k-means within cell
                cell_compressed = self._soft_kmeans_compress(
                    cell_flat, 
                    (t, cell_h, cell_w), 
                    stride
                )
                compressed_cells.append(cell_compressed)
                
                # Track output dimensions for first row/column
                if gi == 0:
                    total_out_w += cell_w // stride
            if gj == 0:
                total_out_h += cell_h // stride
        
        # Concatenate all cells
        compressed = torch.cat(compressed_cells, dim=0)
        
        # Create approximate grid dimensions (for reference, not exact)
        new_h = h // stride
        new_w = w // stride
        new_grid_thw = torch.tensor([t, new_h, new_w], dtype=grid_thw.dtype, device=device)
        
        return compressed, new_grid_thw
    
    def _soft_kmeans_compress(
        self,
        features: torch.Tensor,
        grid_thw: Tuple[int, int, int],
        stride: int
    ) -> torch.Tensor:
        """
        One-Step Soft K-Means compression for a single grid cell.
        
        Uses AvgPooling result as initial centroids, then applies soft attention
        for weighted aggregation.
        
        Args:
            features: [num_tokens, hidden_size]
            grid_thw: (t, h, w) grid dimensions of the cell
            stride: Compression stride
            
        Returns:
            Compressed features [compressed_tokens, hidden_size]
        """
        t, h, w = grid_thw
        hidden_size = features.shape[-1]
        dtype = features.dtype
        
        # Ensure minimum dimensions
        out_h = max(1, h // stride)
        out_w = max(1, w // stride)
        
        # Step 1: Get initial centroids using AvgPooling
        x_img = features.view(t, h, w, hidden_size)
        x_img = x_img.permute(0, 3, 1, 2).contiguous()  # [t, C, h, w]
        
        # Use adaptive pooling to ensure exact output size
        centers = F.adaptive_avg_pool2d(x_img.float(), (out_h, out_w))
        centers = centers.to(dtype)
        centers = centers.permute(0, 2, 3, 1).contiguous()  # [t, out_h, out_w, C]
        centers = centers.view(-1, hidden_size)  # [num_centers, C]
        
        num_centers = centers.shape[0]
        
        # Step 2: Compute similarity (attention scores)
        # Normalize for cosine similarity
        features_norm = F.normalize(features.float(), dim=-1)
        centers_norm = F.normalize(centers.float(), dim=-1)
        
        # features: [N, C], centers: [K, C] -> sim: [N, K]
        sim = torch.mm(features_norm, centers_norm.T)
        
        # Apply temperature scaling for sharper attention
        temperature = 0.1
        sim = sim / temperature
        
        # Step 3: Soft assignment using softmax
        # Each centroid aggregates from tokens (column-wise softmax)
        attn = F.softmax(sim, dim=0)  # [N, K]
        
        # Step 4: Weighted aggregation
        # out = attn.T @ features -> [K, C]
        out = torch.mm(attn.T.to(dtype), features)
        
        return out
