import torch.nn as nn
import torch
import torch.nn.functional as F


def sparse_ce_ssc_loss(logits, target, class_weights):
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=255,
        reduction="mean",
    )
    return criterion(logits, target.long())


def sparse_geo_scal_loss(logits, target):
    # Sparse logits are shaped as [N, C], so we filter unknown points before scoring.
    probs, target = valid_probs(logits, target)
    if target.numel() == 0:
        return logits.sum() * 0

    empty_probs = probs[:, 0]
    nonempty_probs = 1 - empty_probs
    nonempty_target = (target != 0).float()

    # This is the dense geometric scaling loss on occupied vs. empty points.
    intersection = (nonempty_target * nonempty_probs).sum()
    loss = logits.sum() * 0
    loss = loss + ratio_bce(intersection, nonempty_probs.sum())
    loss = loss + ratio_bce(intersection, nonempty_target.sum())
    return loss + ratio_bce(
        ((1 - nonempty_target) * empty_probs).sum(),
        (1 - nonempty_target).sum(),
    )


def sparse_sem_scal_loss(logits, target):
    # The semantic term repeats precision/recall/specificity for each present class.
    probs, target = valid_probs(logits, target)
    if target.numel() == 0:
        return logits.sum() * 0

    loss = logits.sum() * 0
    classes = target.unique()
    for class_idx in classes:
        class_prob = probs[:, class_idx]
        class_target = (target == class_idx).float()
        intersection = (class_prob * class_target).sum()

        loss = loss + ratio_bce(intersection, class_prob.sum())
        loss = loss + ratio_bce(intersection, class_target.sum())
        loss = loss + ratio_bce(
            ((1 - class_prob) * (1 - class_target)).sum(),
            (1 - class_target).sum(),
        )
    return loss / classes.numel()


def valid_probs(logits, target):
    valid = target != 255
    return F.softmax(logits[valid].float(), dim=1), target[valid].long()


def ratio_bce(numerator, denominator):
    if denominator <= 1e-6:
        return numerator * 0
    return unit_bce(numerator / denominator)


def unit_bce(value):
    if not torch.isfinite(value):
        raise RuntimeError(f"Non-finite sparse scaling metric: {value.item()}")
    if (value < -1e-4) or (value > 1 + 1e-4):
        raise RuntimeError(f"Sparse scaling metric outside [0, 1]: {value.item()}")
    value = value.clamp(1e-6, 1 - 1e-6)
    return F.binary_cross_entropy(value, value.new_ones(()))
