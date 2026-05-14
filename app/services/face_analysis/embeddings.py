from math import sqrt


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embeddings must have the same dimension")

    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)


def running_average(
    current: list[float],
    incoming: list[float],
    current_count: int,
) -> list[float]:
    if len(current) != len(incoming):
        raise ValueError("Embeddings must have the same dimension")

    next_count = current_count + 1
    return [
        ((value * current_count) + new_value) / next_count
        for value, new_value in zip(current, incoming)
    ]
