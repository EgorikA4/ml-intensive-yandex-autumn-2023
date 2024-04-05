"""A module for calculating dice loss."""
from torch import Tensor, where


def dice_coeff(
    pred: Tensor,
    target: Tensor,
    epsilon: float = 1e-6,
) -> float:
    """Calcuate dice coefficient.

    The formula for finding DiceLoss:
    DL = 1 - (2 * pred * target + eps) / (target + pred + eps).

    Args:
        pred (Tensor): predicted value.
        target (Tensor): target value.
        epsilon (float): deviation hyperparameter. By default, 1e-6.

    Returns:
        float: DiceLoss.
    """
    sum_dim = (-1, -2, -3)

    inter = 2 * (pred * target).sum(dim=sum_dim)
    sets_sum = pred.sum(dim=sum_dim) + target.sum(dim=sum_dim)

    # getting rid of division by zero
    sets_sum = where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return 1 - dice.mean()
