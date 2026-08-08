"""Post-processing for the released model's brain left/right channel mapping."""

from __future__ import annotations

from categories import prediction


def _split_laterality(name):
    if name.startswith("L-"):
        return "left", name[2:]
    if name.startswith("R-"):
        return "right", name[2:]
    return None, None


def brain_left_right_pairs():
    grouped = {}
    for label in range(132, 215):
        side, region = _split_laterality(prediction[label])
        if side is not None:
            grouped.setdefault(region, {})[side] = label
    pairs = []
    for region, sides in grouped.items():
        if set(sides) == {"left", "right"}:
            pairs.append((sides["left"], sides["right"]))
    return tuple(sorted(pairs))


BRAIN_LEFT_RIGHT_PAIRS = brain_left_right_pairs()


def correct_brain_laterality_scores(scores):
    """Swap bilateral brain channels while preserving canonical label IDs.

    The released checkpoints place each bilateral brain channel on the opposite
    physical side. Swapping score channels before argmax corrects the output
    labels without changing their documented IDs or names.
    """
    if scores.ndim < 2 or scores.shape[1] < 215:
        raise ValueError("Expected scores shaped [batch, >=215 channels, ...]")
    indices = list(range(scores.shape[1]))
    for left_label, right_label in BRAIN_LEFT_RIGHT_PAIRS:
        indices[left_label] = right_label
        indices[right_label] = left_label
    return scores[:, indices, ...]
