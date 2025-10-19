"""
Unified Loss Function (ULF) for Sound Event Detection

Reference:
    Zhang, Y., Togneri, R., & Huang, D. (2024).
    A Unified Loss Function to Tackle Inter-Class and Intra-Class Data Imbalance
    in Sound Event Detection. ICASSP 2024.

ULF addresses both inter-class and intra-class data imbalance simultaneously by:
1. Inter-class balancing: Adjusting weights based on class frequency (ρ, τ parameters)
2. Intra-class balancing: Adjusting weights between active/inactive frames (α, β parameters)
3. Hard example mining: Focal loss components (γ, ξ parameters)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedLoss(nn.Module):
    """
    Unified Loss Function (ULF) for handling both inter-class and intra-class imbalance

    Args:
        num_classes (int): Number of sound event classes
        alpha (float): Global weight for active frames (default: 0.5, paper setting)
        beta (float): Global weight for inactive frames (default: 1.0)
        rho (float): Exponent for active frame class balancing (default: 1.0)
        tau (float): Exponent for inactive frame class balancing (default: 1.0)
        gamma (float): Focal loss focusing parameter for active frames (default: 4.0, paper setting)
        xi (float): Focal loss focusing parameter for inactive frames (default: 4.0, set equal to gamma)
        epsilon (float): Small constant to prevent division by zero (default: 1e-5)
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
    """

    def __init__(self,
                 num_classes,
                 alpha=0.5,
                 beta=1.0,
                 rho=1.0,
                 tau=1.0,
                 gamma=4.0,
                 xi=4.0,
                 epsilon=1e-5,
                 reduction='mean'):
        super(UnifiedLoss, self).__init__()

        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.tau = tau
        self.gamma = gamma
        self.xi = xi
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: Model predictions (before sigmoid), shape [batch, time, num_classes] or [batch, num_classes]
            targets: Ground truth labels, shape [batch, time, num_classes] or [batch, num_classes]

        Returns:
            loss: Computed ULF loss
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Reshape if needed for batch processing
        original_shape = logits.shape
        if len(original_shape) == 3:
            # [batch, time, num_classes] -> [batch*time, num_classes]
            batch_size, time_dim, num_classes = original_shape
            logits = logits.view(-1, num_classes)
            probs = probs.view(-1, num_classes)
            targets = targets.view(-1, num_classes)
        else:
            batch_size = original_shape[0]
            time_dim = 1

        # Calculate C = T × batch_size (total frames in batch)
        C = batch_size * time_dim

        # Count active and inactive frames per class in the batch
        # N_m+ : number of active frames for class m
        # N_m- : number of inactive frames for class m
        N_positive = targets.sum(dim=0) + self.epsilon  # [num_classes]
        N_negative = (1 - targets).sum(dim=0) + self.epsilon  # [num_classes]

        # Compute weights for active frames: w_m^+ = α * (C / (N_m+ + ε))^ρ * (1 - y_t,m)^γ
        freq_weight_pos = (C / N_positive) ** self.rho  # [num_classes]
        focal_weight_pos = (1 - probs) ** self.gamma  # [batch*time, num_classes]
        w_pos = self.alpha * freq_weight_pos.unsqueeze(0) * focal_weight_pos  # [batch*time, num_classes]

        # Compute weights for inactive frames: w_m^- = β * (C / (N_m- + ε))^τ * (y_t,m)^ξ
        freq_weight_neg = (C / N_negative) ** self.tau  # [num_classes]
        focal_weight_neg = probs ** self.xi  # [batch*time, num_classes]
        w_neg = self.beta * freq_weight_neg.unsqueeze(0) * focal_weight_neg  # [batch*time, num_classes]

        # Compute binary cross-entropy loss with computed weights
        # L = -Σ{w_m^+ * z_t,m * log(y_t,m) + w_m^- * (1 - z_t,m) * log(1 - y_t,m)}

        # Numerical stability: use log_sigmoid
        pos_loss = w_pos * targets * F.logsigmoid(logits)
        neg_loss = w_neg * (1 - targets) * F.logsigmoid(-logits)

        loss = -(pos_loss + neg_loss)

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            # Reshape back to original shape
            if len(original_shape) == 3:
                return loss.view(batch_size, time_dim, self.num_classes)
            return loss


class UnifiedLossWithClassWeights(nn.Module):
    """
    Enhanced ULF with additional per-class weights for extreme imbalance

    This variant allows manual specification of per-class weights on top of ULF's automatic balancing.
    Useful when certain classes are known to be particularly important or problematic.

    Args:
        num_classes (int): Number of sound event classes
        class_weights (torch.Tensor, optional): Per-class weights, shape [num_classes]
        **kwargs: Other ULF parameters (alpha, beta, rho, tau, gamma, xi, epsilon, reduction)
    """

    def __init__(self, num_classes, class_weights=None, **kwargs):
        super(UnifiedLossWithClassWeights, self).__init__()

        self.ulf = UnifiedLoss(num_classes, **kwargs)

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.register_buffer('class_weights', torch.ones(num_classes))

    def forward(self, logits, targets):
        loss = self.ulf(logits, targets)

        # Apply per-class weights
        if loss.dim() == 3:  # [batch, time, num_classes]
            weighted_loss = loss * self.class_weights.view(1, 1, -1)
        elif loss.dim() == 2:  # [batch, num_classes]
            weighted_loss = loss * self.class_weights.view(1, -1)
        else:
            weighted_loss = loss

        # Apply reduction
        if self.ulf.reduction == 'mean':
            return weighted_loss.mean()
        elif self.ulf.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)

    Args:
        alpha (float): Weighting factor for positive examples (default: 0.25)
        gamma (float): Focusing parameter (default: 2.0)
        reduction (str): 'none' | 'mean' | 'sum'
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: Model predictions (before sigmoid), shape [batch, time, num_classes] or [batch, num_classes]
            targets: Ground truth labels, shape [batch, time, num_classes] or [batch, num_classes]
        """
        # Get probabilities
        probs = torch.sigmoid(logits)

        # Binary cross entropy loss
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Focal term: (1 - p_t)^gamma
        # For positive examples: p_t = probs
        # For negative examples: p_t = 1 - probs
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Final focal loss
        focal_loss = alpha_weight * focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def create_ulf_loss(config_type='whale', custom_params=None):
    """
    Factory function to create ULF loss with preset configurations

    Args:
        config_type (str): 'whale', 'dcase', or 'paper'
            - 'whale': For Whale dataset (4 classes)
            - 'dcase': For DCASE dataset (10 classes)
            - 'paper': Original paper settings
        custom_params (dict, optional): Override default parameters

    Returns:
        UnifiedLoss instance
    """

    # Default configurations
    configs = {
        'paper': {
            'num_classes': 10,  # DCASE 2020
            'alpha': 0.5,
            'beta': 1.0,
            'rho': 1.0,
            'tau': 1.0,
            'gamma': 4.0,
            'xi': 4.0,
            'epsilon': 1e-5,
            'reduction': 'mean'
        },
        'whale': {
            'num_classes': 4,
            'alpha': 0.5,
            'beta': 1.0,
            'rho': 1.0,
            'tau': 1.0,
            'gamma': 3.0,  # Slightly less focusing for fewer classes
            'xi': 3.0,
            'epsilon': 1e-5,
            'reduction': 'mean'
        },
        'dcase': {
            'num_classes': 10,
            'alpha': 0.5,
            'beta': 1.0,
            'rho': 1.0,
            'tau': 1.0,
            'gamma': 4.0,
            'xi': 4.0,
            'epsilon': 1e-5,
            'reduction': 'mean'
        }
    }

    # Get base config
    base_config = configs.get(config_type, configs['paper'])

    # Override with custom parameters
    if custom_params:
        base_config.update(custom_params)

    return UnifiedLoss(**base_config)


# Convenience functions for quick testing
def test_ulf():
    """Test ULF with dummy data"""
    print("Testing Unified Loss Function...")

    # Test case 1: DCASE-like data
    batch_size = 4
    time_frames = 309
    num_classes = 10

    logits = torch.randn(batch_size, time_frames, num_classes)
    targets = torch.randint(0, 2, (batch_size, time_frames, num_classes)).float()

    # Create loss
    ulf_loss = create_ulf_loss('dcase')

    # Compute loss
    loss = ulf_loss(logits, targets)
    print(f"DCASE ULF Loss: {loss.item():.4f}")

    # Test case 2: Whale-like data
    num_classes = 4
    logits = torch.randn(batch_size, time_frames, num_classes)
    targets = torch.randint(0, 2, (batch_size, time_frames, num_classes)).float()

    ulf_loss = create_ulf_loss('whale')
    loss = ulf_loss(logits, targets)
    print(f"Whale ULF Loss: {loss.item():.4f}")

    # Test gradient flow
    logits.requires_grad = True
    loss = ulf_loss(logits, targets)
    loss.backward()
    print(f"Gradient computed successfully: {logits.grad is not None}")

    print("ULF test passed!")


if __name__ == '__main__':
    test_ulf()
