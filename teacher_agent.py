from typing import Any

def calculate_manhattan_distance_in_blocks(
    guessed_x_in_blocks: int,
    guessed_y_in_blocks: int,
    true_x_in_blocks: int,
    true_y_in_blocks: int
) -> int:
    x_distance_in_blocks = abs(true_x_in_blocks - guessed_x_in_blocks)
    y_distance_in_blocks = abs(true_y_in_blocks - guessed_y_in_blocks)
    return x_distance_in_blocks + y_distance_in_blocks


def update_model_state_with_true_observation(
    model_state: dict[str, Any],
    true_x_in_blocks: int,
    true_y_in_blocks: int
) -> dict[str, Any]:
    cell_counts_by_row = model_state["cell_counts_by_row"]

    cell_counts_by_row[true_y_in_blocks][true_x_in_blocks] += 1
    model_state["total_observations_count"] += 1
    model_state["sum_of_true_x_in_blocks"] += true_x_in_blocks
    model_state["sum_of_true_y_in_blocks"] += true_y_in_blocks

    return model_state
    
def update_scoreboard_with_distance(
    model_state: dict[str, Any],
    manhattan_distance_in_blocks: int
) -> dict[str, Any]:
    model_state["total_rounds_count"] += 1
    model_state["total_manhattan_distance_sum"] += manhattan_distance_in_blocks

    best_distance = model_state["best_manhattan_distance_in_blocks"]
    if best_distance is None or manhattan_distance_in_blocks < best_distance:
        model_state["best_manhattan_distance_in_blocks"] = manhattan_distance_in_blocks

    worst_distance = model_state["worst_manhattan_distance_in_blocks"]
    if worst_distance is None or manhattan_distance_in_blocks > worst_distance:
        model_state["worst_manhattan_distance_in_blocks"] = manhattan_distance_in_blocks

    return model_state