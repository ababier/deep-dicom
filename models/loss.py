import torch
import torch.nn.functional as F
from torch import nn


class FocalL1Loss(torch.nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred, target):
        abs_error = (pred - target).abs()
        focal_weight = torch.exp(-self.gamma * target)  # Higher weight where values are high
        return (focal_weight * abs_error).mean()


class HybridGradientLoss(nn.Module):
    """
    Hybrid loss combining Huber loss and Gradient loss (Sobel-based).

    Args:
        alpha (float): Weight for Huber loss.
        beta (float): Weight for Gradient loss.
    """

    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.huber = nn.MSELoss()

    def forward(self, pred, target):
        huber_loss = self.huber(pred, target)
        grad_loss = gradient_loss_3d(pred, target)
        return self.alpha * huber_loss + self.beta * grad_loss


def gradient_loss_3d(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes gradient loss using 3D Sobel filters.

    Args:
        pred (torch.Tensor): Predicted 3D image tensor [B, C, D, H, W].
        target (torch.Tensor): Target 3D image tensor [B, C, D, H, W].

    Returns:
        torch.Tensor: Scalar loss.
    """
    device = pred.device

    # 3D Sobel kernels (approximating derivatives)
    sobel_x = torch.tensor(
        [[[[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]], [[-3, 0, 3], [-6, 0, 6], [-3, 0, 3]], [[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]]],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(
        0
    )  # Shape: [1, 1, 3, 3, 3]

    sobel_y = torch.tensor(
        [[[[-1, -3, -1], [0, 0, 0], [1, 3, 1]], [[-3, -6, -3], [0, 0, 0], [3, 6, 3]], [[-1, -3, -1], [0, 0, 0], [1, 3, 1]]]],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    sobel_z = torch.tensor(
        [[[[-1, -3, -1], [-3, -6, -3], [-1, -3, -1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[1, 3, 1], [3, 6, 3], [1, 3, 1]]]],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    def sobel_filter(image, kernel):
        return F.conv3d(image, kernel, padding=1)

    pred_grad_x, pred_grad_y, pred_grad_z = sobel_filter(pred, sobel_x), sobel_filter(pred, sobel_y), sobel_filter(pred, sobel_z)
    target_grad_x, target_grad_y, target_grad_z = sobel_filter(target, sobel_x), sobel_filter(target, sobel_y), sobel_filter(target, sobel_z)

    loss_x = F.l1_loss(pred_grad_x, target_grad_x)
    loss_y = F.l1_loss(pred_grad_y, target_grad_y)
    loss_z = F.l1_loss(pred_grad_z, target_grad_z)

    return loss_x + loss_y + loss_z
