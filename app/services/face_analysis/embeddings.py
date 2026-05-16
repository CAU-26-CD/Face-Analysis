import numpy as np


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embeddings must have the same dimension")

    left_vec = np.asarray(left, dtype=np.float32)
    right_vec = np.asarray(right, dtype=np.float32)

    left_norm = float(np.linalg.norm(left_vec))
    right_norm = float(np.linalg.norm(right_vec))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return float(np.dot(left_vec, right_vec) / (left_norm * right_norm))


def running_average(
    current: list[float],
    incoming: list[float],
    current_count: int,
) -> list[float]:
    if len(current) != len(incoming):
        raise ValueError("Embeddings must have the same dimension")

    next_count = current_count + 1
    current_vec = np.asarray(current, dtype=np.float64)
    incoming_vec = np.asarray(incoming, dtype=np.float64)
    return ((current_vec * current_count + incoming_vec) / next_count).tolist()
