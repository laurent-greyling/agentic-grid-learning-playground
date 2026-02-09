import json
from datetime import datetime

def append_episode_to_json_file(
    json_file_path: str,
    round_index: int,
    guessed_x_in_blocks: int,
    guessed_y_in_blocks: int,
    true_x_in_blocks: int,
    true_y_in_blocks: int,
    manhattan_distance_in_blocks: int
) -> None:
    episode_record = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "round_index": round_index,
        "guessed_location": {"x_in_blocks": guessed_x_in_blocks, "y_in_blocks": guessed_y_in_blocks},
        "true_location": {"x_in_blocks": true_x_in_blocks, "y_in_blocks": true_y_in_blocks},
        "manhattan_distance_in_blocks": manhattan_distance_in_blocks
    }

    with open(json_file_path, "a", encoding="utf-8") as file_handle:
        file_handle.write(json.dumps(episode_record))
        file_handle.write("\n")