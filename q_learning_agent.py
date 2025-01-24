import numpy as np
from numpy import random
from scipy.special import softmax


class QLearningAgent:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.3,
        learning_rate_decay_parameter=0.999,
        discount_factor=0.99,
        exploration_rate=1.0,
        exploration_decay_parameter=0.95,
        min_exploration_rate=0.0,
    ):
        """
        Initialize the Q-learning agent.

        :param state_size: Size of the state space
        :param action_size: Size of the action space
        :param learning_rate: Learning rate for Q-learning
        :param discount_factor: Discount factor for future rewards
        :param exploration_rate: Initial exploration rate for epsilon-greedy policy
        :param exploration_decay: Decay rate for exploration rate
        :param min_exploration_rate: Minimum exploration rate
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.learning_rate_decay_parameter = learning_rate_decay_parameter
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_parameter = exploration_decay_parameter
        self.min_exploration_rate = min_exploration_rate
        self.q_table = np.zeros((state_size, action_size))
        self.v_table = np.zeros(state_size)
        self.rng = random.default_rng()

    def choose_action(self, state):
        """
        Choose an action based on the current state using epsilon-greedy policy.

        :param state: Current state
        :return: Action to take
        """
        if random.uniform(0, 1) < self.exploration_rate:
            return self.rng.integers(0, self.action_size)
        else:   
            probs = softmax(self.q_table[state])
            draw = self.rng.choice(self.action_size, p=probs)
            return draw

    def update_q_table(self, state, action, reward, next_state):
        """
        Update the Q-table based on the action taken and the reward received.

        :param state: Previous state
        :param action: Action taken
        :param reward: Reward received
        :param next_state: Next state
        """
        max_value = np.max(self.q_table[next_state])
        td_target = reward + self.discount_factor * max_value
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def update_v_table(self, state, reward, next_state):
        """
        Update the V-table based on the reward received.

        :param state: Current state
        :param reward: Reward received
        """
        self.v_table[state] = (1 - self.learning_rate) * self.v_table[
            state
        ] + self.learning_rate * (
            reward + self.discount_factor * self.v_table[next_state]
        )

    def decay(self, param, delta, time):
        """
        Decays the parameter according to the Kara-Dogan rule
        """
        y = time**2 / (delta + time)
        new_param = param / (1 + y)
        return new_param

    def decay_exploration_rate(self, time):
        """
        Decay the exploration rate.
        """
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.decay(self.exploration_rate, self.exploration_decay_parameter, time),
        )

    def decay_learning_rate(self, time):
        """
        Decay the learning rate.
        """
        self.learning_rate = self.decay(
            self.learning_rate, self.learning_rate_decay_parameter, time
        )
