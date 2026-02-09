import random

class EpsilonGreedyPolicy:
    def __init__(self, starting_epsilon: float, minimum_epsilon: float, epsilon_decay_per_round: float):
        self.starting_epsilon = starting_epsilon
        self.minimum_epsilon = minimum_epsilon
        self.epsilon_decay_per_round = epsilon_decay_per_round

    def get_epsilon_for_round(self, round_index: int) -> float:
        decayed_epsilon = self.starting_epsilon - (round_index * self.epsilon_decay_per_round)
        return max(self.minimum_epsilon, decayed_epsilon)

    def should_explore(self, round_index: int) -> bool:
        epsilon = self.get_epsilon_for_round(round_index)
        random_value_between_0_and_1 = random.random()
        return random_value_between_0_and_1 < epsilon
