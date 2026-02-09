# python3 -m venv .venv
# source .venv/bin/activate
# pip install matplotlib

import random

from environment import create_flag_location_with_hotspot_bias
from teacher_agent import (
    calculate_manhattan_distance_in_blocks,
    update_model_state_with_true_observation,
    update_scoreboard_with_distance
)
from model_state_store import (
    load_model_state_or_create_new,
    save_model_state
)
from policy import EpsilonGreedyPolicy
from guesser_agent import EpsilonGreedyModelBasedGuesserAgent
from visualization import plot_learning_results

APPLICATION_NAME = "Agentic Grid Learning"
APPLICATION_VERSION = "0.1.0"

GRID_WIDTH_IN_BLOCKS = 100
GRID_HEIGHT_IN_BLOCKS = 100

HOTSPOT_CENTER_X_IN_BLOCKS = 70
HOTSPOT_CENTER_Y_IN_BLOCKS = 25
HOTSPOT_STANDARD_DEVIATION_IN_BLOCKS = 8.0
HOTSPOT_PROBABILITY = 0.8

MODEL_STATE_JSON_FILE_PATH = "model_state.json"


def main():
    print("Application name:", APPLICATION_NAME)
    print("Application version:", APPLICATION_VERSION)
    print()

    # Ensure model_state.json exists before the loop starts.
    _ = load_model_state_or_create_new(
        json_file_path=MODEL_STATE_JSON_FILE_PATH,
        grid_width_in_blocks=GRID_WIDTH_IN_BLOCKS,
        grid_height_in_blocks=GRID_HEIGHT_IN_BLOCKS
    )

    #probablility of exploitation figures, enforce learning
    #early we explore more to learn faster, later we will exploit more so we try and perform better
    policy = EpsilonGreedyPolicy(
        starting_epsilon=0.30,
        minimum_epsilon=0.05,
        epsilon_decay_per_round=0.002
    )

    guesser_agent = EpsilonGreedyModelBasedGuesserAgent(
        grid_width_in_blocks=GRID_WIDTH_IN_BLOCKS,
        grid_height_in_blocks=GRID_HEIGHT_IN_BLOCKS,
        model_state_json_file_path=MODEL_STATE_JSON_FILE_PATH,
        policy=policy,
        exploration_radius_in_blocks=10
    )

    session_manhattan_distance_sum_in_blocks = 0
    session_rounds_count = 0
    number_of_rounds = 200

    round_indices = []
    belief_x_history = []
    belief_y_history = []
    true_x_history = []
    true_y_history = []
    distance_history = []
    session_avg_history = []

    for round_index in range(number_of_rounds):
        guessed_x_in_blocks, guessed_y_in_blocks = guesser_agent.guess_flag_location(
            round_index=round_index
        )

        true_x_in_blocks, true_y_in_blocks = create_flag_location_with_hotspot_bias(
            grid_width_in_blocks=GRID_WIDTH_IN_BLOCKS,
            grid_height_in_blocks=GRID_HEIGHT_IN_BLOCKS,
            hotspot_center_x_in_blocks=HOTSPOT_CENTER_X_IN_BLOCKS,
            hotspot_center_y_in_blocks=HOTSPOT_CENTER_Y_IN_BLOCKS,
            hotspot_standard_deviation_in_blocks=HOTSPOT_STANDARD_DEVIATION_IN_BLOCKS,
            hotspot_probability=HOTSPOT_PROBABILITY
        )

        manhattan_distance_in_blocks = calculate_manhattan_distance_in_blocks(
            guessed_x_in_blocks=guessed_x_in_blocks,
            guessed_y_in_blocks=guessed_y_in_blocks,
            true_x_in_blocks=true_x_in_blocks,
            true_y_in_blocks=true_y_in_blocks
        )
        
        # capture belief before the teacher updates the model for this round
        belief_x_in_blocks, belief_y_in_blocks = guesser_agent.get_belief_center_in_blocks()

        session_manhattan_distance_sum_in_blocks += manhattan_distance_in_blocks
        session_rounds_count += 1
        session_average_manhattan_distance_in_blocks = (
            session_manhattan_distance_sum_in_blocks / session_rounds_count
        )

        # Teacher updates model state with the true observation
        model_state = load_model_state_or_create_new(
            json_file_path=MODEL_STATE_JSON_FILE_PATH,
            grid_width_in_blocks=GRID_WIDTH_IN_BLOCKS,
            grid_height_in_blocks=GRID_HEIGHT_IN_BLOCKS
        )

        model_state = update_model_state_with_true_observation(
            model_state=model_state,
            true_x_in_blocks=true_x_in_blocks,
            true_y_in_blocks=true_y_in_blocks
        )

        model_state = update_scoreboard_with_distance(
            model_state=model_state,
            manhattan_distance_in_blocks=manhattan_distance_in_blocks
        )

        lifetime_average_manhattan_distance_in_blocks = (
            model_state["total_manhattan_distance_sum"] / model_state["total_rounds_count"]
        )

        save_model_state(
            json_file_path=MODEL_STATE_JSON_FILE_PATH,
            model_state=model_state
        )

        should_print_this_round = (round_index % 10 == 0)
        if should_print_this_round:
            epsilon = policy.get_epsilon_for_round(round_index=round_index)

            round_indices.append(round_index)
            belief_x_history.append(belief_x_in_blocks)
            belief_y_history.append(belief_y_in_blocks)
            true_x_history.append(true_x_in_blocks)
            true_y_history.append(true_y_in_blocks)
            distance_history.append(manhattan_distance_in_blocks)
            session_avg_history.append(session_average_manhattan_distance_in_blocks)

            print(
                "Round", round_index,
                "| Epsilon:", round(epsilon, 3),
                "| Guess (x, y):", guessed_x_in_blocks, guessed_y_in_blocks,
                "| Belief (x, y):", belief_x_in_blocks, belief_y_in_blocks,
                "| True (x, y):", true_x_in_blocks, true_y_in_blocks,
                "| Distance:", manhattan_distance_in_blocks,
                "| Session avg:", round(session_average_manhattan_distance_in_blocks, 2),
                "| Lifetime avg:", round(lifetime_average_manhattan_distance_in_blocks, 2),
                "| Lifetime rounds:", model_state["total_rounds_count"],
                "| Total observations:", model_state["total_observations_count"],
            )

    plot_learning_results(
        round_indices=round_indices,
        belief_x_history=belief_x_history,
        belief_y_history=belief_y_history,
        true_x_history=true_x_history,
        true_y_history=true_y_history,
        distance_history=distance_history,
        session_avg_history=session_avg_history,
    )

if __name__ == "__main__":
    main()