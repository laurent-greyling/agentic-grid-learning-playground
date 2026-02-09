import matplotlib.pyplot as plt

def plot_learning_results(
    round_indices,
    belief_x_history,
    belief_y_history,
    true_x_history,
    true_y_history,
    distance_history,
    session_avg_history
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Agentic Grid Learning – Learning Behaviour", fontsize=16)

    # --- Belief vs True (X) ---
    axes[0, 0].plot(round_indices, belief_x_history, label="Belief X", linewidth=2)
    axes[0, 0].plot(round_indices, true_x_history, label="True X", alpha=0.6)
    axes[0, 0].set_title("X Coordinate Learning")
    axes[0, 0].set_xlabel("Round")
    axes[0, 0].set_ylabel("X")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # --- Belief vs True (Y) ---
    axes[0, 1].plot(round_indices, belief_y_history, label="Belief Y", linewidth=2)
    axes[0, 1].plot(round_indices, true_y_history, label="True Y", alpha=0.6)
    axes[0, 1].set_title("Y Coordinate Learning")
    axes[0, 1].set_xlabel("Round")
    axes[0, 1].set_ylabel("Y")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # --- Distance per round ---
    axes[1, 0].plot(round_indices, distance_history, color="orange")
    axes[1, 0].set_title("Manhattan Distance per Round")
    axes[1, 0].set_xlabel("Round")
    axes[1, 0].set_ylabel("Distance")
    axes[1, 0].grid(True)

    # --- Learning curve ---
    axes[1, 1].plot(round_indices, session_avg_history, color="green", linewidth=2)
    axes[1, 1].set_title("Session Average Distance (Learning Curve)")
    axes[1, 1].set_xlabel("Round")
    axes[1, 1].set_ylabel("Average Distance")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()