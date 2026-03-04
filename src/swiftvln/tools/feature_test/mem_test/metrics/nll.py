# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Chunk-level NLL/CE Computation

This module implements teacher-forcing NLL (Negative Log-Likelihood) and
CE (Cross-Entropy) computation for evaluating VLN action prediction quality.

Key concepts:
- Teacher forcing: Feed ground-truth prefix to compute probability of next token
- Chunk: A sequence of K=4 actions predicted in one turn
- NLL: Sum of negative log probabilities for all tokens
- CE: NLL normalized by number of tokens
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Add parent directories to path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_mem_test_dir = os.path.dirname(_current_dir)
_vln_dir = os.path.dirname(os.path.dirname(_mem_test_dir))
if _vln_dir not in sys.path:
    sys.path.insert(0, _vln_dir)


@dataclass
class NLLResult:
    """Result of NLL computation for a single chunk."""
    nll: float  # Total negative log-likelihood
    ce: float   # Cross-entropy (NLL / num_tokens)
    num_tokens: int  # Number of action tokens
    per_token_nll: List[float]  # NLL for each token
    target_text: str  # Target action text
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'nll': self.nll,
            'ce': self.ce,
            'num_tokens': self.num_tokens,
            'per_token_nll': self.per_token_nll,
            'target_text': self.target_text,
        }


class NLLComputer:
    """
    Compute Chunk-level NLL/CE using teacher forcing.
    
    This class handles the computation of negative log-likelihood for
    action prediction in a teacher-forcing setup.
    """
    
    # Action vocabulary (must match dataset.py)
    IDX2ACTIONS = {
        0: 'STOP',
        1: "↑",  # MOVE_FORWARD
        2: "←",  # TURN_LEFT
        3: "→",  # TURN_RIGHT
    }
    
    def __init__(
        self,
        model,
        processor,
        device: str = 'cuda',
        num_future_steps: int = 4,
    ):
        """
        Initialize NLL computer.
        
        Args:
            model: The VLN model
            processor: The tokenizer/processor
            device: Device for computation
            num_future_steps: Number of actions per chunk (K)
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.num_future_steps = num_future_steps
        
        # Ensure model is in eval mode
        self.model.eval()
    
    def actions_to_text(self, actions: List[int]) -> str:
        """Convert action indices to text (no spaces, matching training format)."""
        if len(actions) == 0:
            return "STOP"
        return ''.join([self.IDX2ACTIONS.get(a, 'STOP') for a in actions])
    
    @torch.no_grad()
    def compute_nll(
        self,
        inputs_embeds: torch.Tensor,
        target_actions: List[int],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> NLLResult:
        """
        Compute NLL for a chunk using teacher forcing.
        
        Args:
            inputs_embeds: Input embeddings [1, seq_len, hidden_size]
            target_actions: Ground-truth actions for this chunk
            attention_mask: Optional attention mask [1, seq_len]
            
        Returns:
            NLLResult with NLL and CE values
        """
        # Convert actions to text
        target_text = self.actions_to_text(target_actions)
        
        # Tokenize target
        target_tokens = self.processor.tokenizer.encode(
            target_text,
            add_special_tokens=False,
            return_tensors='pt'
        ).to(self.device)
        
        num_target_tokens = target_tokens.shape[1]
        
        if num_target_tokens == 0:
            return NLLResult(
                nll=0.0,
                ce=0.0,
                num_tokens=0,
                per_token_nll=[],
                target_text=target_text,
            )
        
        # Get target embeddings
        target_embeds = self._get_text_embeddings(target_tokens)
        
        # Concatenate input and target (except last target token)
        # For teacher forcing, we feed [input, target[:-1]] and predict target
        full_embeds = torch.cat([
            inputs_embeds,
            target_embeds[:, :-1, :]  # All but last target token
        ], dim=1)
        
        # Create attention mask
        seq_len = full_embeds.shape[1]
        if attention_mask is None:
            full_attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=self.device)
        else:
            # Extend attention mask for target tokens
            target_mask = torch.ones(1, num_target_tokens - 1, dtype=torch.long, device=self.device)
            full_attention_mask = torch.cat([attention_mask, target_mask], dim=1)
        
        # Forward pass
        outputs = self.model(
            inputs_embeds=full_embeds,
            attention_mask=full_attention_mask,
            use_cache=False,
        )
        
        # Get logits for target positions
        # Logits shape: [1, seq_len, vocab_size]
        logits = outputs.logits
        
        # We want logits at positions that should predict target tokens
        # Position i predicts token i+1
        # So for target tokens at positions [input_len, input_len+1, ..., input_len+num_target-1]
        # We need logits at positions [input_len-1, input_len, ..., input_len+num_target-2]
        input_len = inputs_embeds.shape[1]
        target_logits = logits[:, input_len-1:input_len+num_target_tokens-1, :]  # [1, num_target, vocab]
        
        # Compute per-token NLL
        log_probs = F.log_softmax(target_logits, dim=-1)
        
        per_token_nll = []
        total_nll = 0.0
        
        for i in range(num_target_tokens):
            token_id = target_tokens[0, i].item()
            token_log_prob = log_probs[0, i, token_id].item()
            token_nll = -token_log_prob
            per_token_nll.append(token_nll)
            total_nll += token_nll
        
        # Compute CE (average NLL)
        ce = total_nll / num_target_tokens if num_target_tokens > 0 else 0.0
        
        return NLLResult(
            nll=total_nll,
            ce=ce,
            num_tokens=num_target_tokens,
            per_token_nll=per_token_nll,
            target_text=target_text,
        )
    
    def _get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get text embeddings from input_ids."""
        base_model = self.model
        if hasattr(base_model, 'model') and hasattr(base_model.model, 'embed_tokens'):
            return base_model.model.embed_tokens(input_ids)
        elif hasattr(base_model, 'model') and hasattr(base_model.model, 'language_model'):
            return base_model.model.language_model.embed_tokens(input_ids)
        else:
            raise ValueError("Cannot find embed_tokens in model")


def compute_chunk_nll(
    model,
    processor,
    inputs_embeds: torch.Tensor,
    target_actions: List[int],
    attention_mask: Optional[torch.Tensor] = None,
    device: str = 'cuda',
) -> NLLResult:
    """
    Convenience function to compute NLL for a single chunk.
    
    Args:
        model: The VLN model
        processor: The tokenizer/processor
        inputs_embeds: Input embeddings [1, seq_len, hidden_size]
        target_actions: Ground-truth actions
        attention_mask: Optional attention mask
        device: Device for computation
        
    Returns:
        NLLResult with computed metrics
    """
    computer = NLLComputer(model, processor, device)
    return computer.compute_nll(inputs_embeds, target_actions, attention_mask)
