"""
Loss functions for Sound Event Detection

Available loss functions:
- UnifiedLoss (ULF): Handles both inter-class and intra-class imbalance
- UnifiedLossWithClassWeights: ULF with additional per-class weights
- FocalLoss: Focal Loss for addressing class imbalance (simpler than ULF)
"""

from .unified_loss import (
    UnifiedLoss,
    UnifiedLossWithClassWeights,
    create_ulf_loss,
    FocalLoss
)

__all__ = [
    'UnifiedLoss',
    'UnifiedLossWithClassWeights',
    'create_ulf_loss',
    'FocalLoss'
]
