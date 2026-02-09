import json
import os
from typing import Any

def create_new_model_state(
    grid_width_in_blocks: int,
    grid_height_in_blocks: int
) -> dict[str, Any]:
    cell_counts_by_row = []
    for _y in range(grid_height_in_blocks):
        row_counts = []
        for _x in range(grid_width_in_blocks):
            row_counts.append(0)
        cell_counts_by_row.append(row_counts)

    model_state = {
        "grid_width_in_blocks": grid_width_in_blocks,
        "grid_height_in_blocks": grid_height_in_blocks,
        "total_observations_count": 0,
        "sum_of_true_x_in_blocks": 0,
        "sum_of_true_y_in_blocks": 0,
        "cell_counts_by_row": cell_counts_by_row,
        "total_rounds_count": 0,
        "total_manhattan_distance_sum": 0,
        "best_manhattan_distance_in_blocks": None,
        "worst_manhattan_distance_in_blocks": None
    }
    return model_state

def load_model_state_or_create_new(
    json_file_path: str,
    grid_width_in_blocks: int,
    grid_height_in_blocks: int
) -> dict[str, Any]:
    if os.path.exists(json_file_path) is False:
        model_state = create_new_model_state(
            grid_width_in_blocks=grid_width_in_blocks,
            grid_height_in_blocks=grid_height_in_blocks
        )
        save_model_state(json_file_path=json_file_path, model_state=model_state)
        return model_state

    with open(json_file_path, "r", encoding="utf-8") as file_handle:
        model_state = json.load(file_handle)
        return model_state


def save_model_state(json_file_path: str, model_state: dict[str, Any]) -> None:
    with open(json_file_path, "w", encoding="utf-8") as file_handle:
        json.dump(model_state, file_handle, indent=2)