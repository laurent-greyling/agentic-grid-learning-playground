import random
from typing import Any

from model_state_store import load_model_state_or_create_new
from policy import EpsilonGreedyPolicy


class EpsilonGreedyModelBasedGuesserAgent:
    def __init__(
        self,
        grid_width_in_blocks: int,
        grid_height_in_blocks: int,
        model_state_json_file_path: str,
        policy: EpsilonGreedyPolicy,
        exploration_radius_in_blocks: int
    ):
        self.grid_width_in_blocks = grid_width_in_blocks
        self.grid_height_in_blocks = grid_height_in_blocks
        self.model_state_json_file_path = model_state_json_file_path
        self.policy = policy
        self.exploration_radius_in_blocks = exploration_radius_in_blocks

    def guess_flag_location(self, round_index: int) -> tuple[int, int]:
        belief_center_x_in_blocks, belief_center_y_in_blocks = self.get_belief_center_in_blocks()

        should_explore_this_round = self.policy.should_explore(round_index=round_index)
        if should_explore_this_round:
            guessed_x_in_blocks, guessed_y_in_blocks = self._create_local_exploration_guess_in_blocks(
                belief_center_x_in_blocks=belief_center_x_in_blocks,
                belief_center_y_in_blocks=belief_center_y_in_blocks
            )
            return guessed_x_in_blocks, guessed_y_in_blocks

        # Exploit: use belief center
        return belief_center_x_in_blocks, belief_center_y_in_blocks

    def get_belief_center_in_blocks(self) -> tuple[int, int]:
        """
        Median belief center:
        - handle outliers better than mean
        - Uses the learned cell_counts_by_row grid
        """
        model_state = self._load_model_state()
        cell_counts_by_row: list[list[int]] = model_state["cell_counts_by_row"]
        total_observations: int = model_state["total_observations_count"]

        # If we have no data yet, default to the center of the grid.
        if total_observations <= 0:
            return (self.grid_width_in_blocks // 2, self.grid_height_in_blocks // 2)

        median_x = self._median_x_from_cell_counts(cell_counts_by_row, total_observations)
        median_y = self._median_y_from_cell_counts(cell_counts_by_row, total_observations)

        return (median_x, median_y)

    def _median_x_from_cell_counts(
        self,
        cell_counts_by_row: list[list[int]],
        total_observations: int
    ) -> int:
        target_rank = (total_observations - 1) // 2  # 0-based median rank
        cumulative = 0

        for x in range(self.grid_width_in_blocks):
            column_sum = 0
            for y in range(self.grid_height_in_blocks):
                column_sum += cell_counts_by_row[y][x]

            cumulative += column_sum
            if cumulative > target_rank:
                return x

        return self.grid_width_in_blocks - 1

    def _median_y_from_cell_counts(
        self,
        cell_counts_by_row: list[list[int]],
        total_observations: int
    ) -> int:
        target_rank = (total_observations - 1) // 2  # 0-based median rank
        cumulative = 0

        for y in range(self.grid_height_in_blocks):
            row_sum = 0
            for x in range(self.grid_width_in_blocks):
                row_sum += cell_counts_by_row[y][x]

            cumulative += row_sum
            if cumulative > target_rank:
                return y

        return self.grid_height_in_blocks - 1

    def _load_model_state(self) -> dict[str, Any]:
        return load_model_state_or_create_new(
            json_file_path=self.model_state_json_file_path,
            grid_width_in_blocks=self.grid_width_in_blocks,
            grid_height_in_blocks=self.grid_height_in_blocks,
        )

    def _create_local_exploration_guess_in_blocks(
        self,
        belief_center_x_in_blocks: int,
        belief_center_y_in_blocks: int
    ) -> tuple[int, int]:
        x_offset_in_blocks = random.randint(-self.exploration_radius_in_blocks, self.exploration_radius_in_blocks)
        y_offset_in_blocks = random.randint(-self.exploration_radius_in_blocks, self.exploration_radius_in_blocks)

        guessed_x_in_blocks = belief_center_x_in_blocks + x_offset_in_blocks
        guessed_y_in_blocks = belief_center_y_in_blocks + y_offset_in_blocks

        guessed_x_in_blocks = self._clamp_to_grid_x_in_blocks(guessed_x_in_blocks)
        guessed_y_in_blocks = self._clamp_to_grid_y_in_blocks(guessed_y_in_blocks)

        return guessed_x_in_blocks, guessed_y_in_blocks

    def _clamp_to_grid_x_in_blocks(self, x_in_blocks: int) -> int:
        return max(0, min(self.grid_width_in_blocks - 1, x_in_blocks))

    def _clamp_to_grid_y_in_blocks(self, y_in_blocks: int) -> int:
        return max(0, min(self.grid_height_in_blocks - 1, y_in_blocks))