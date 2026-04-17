import random

class QAgent:
    def __init__(self, actions=[0, 1], alpha=0.1, gamma=0.9, epsilon=1.0):
        self.Q = {}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995

    def get_q(self, state, action):
        return self.Q.get((state, action), 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q0 = self.get_q(state, 0)
        q1 = self.get_q(state, 1)
        if q0 > q1: return 0
        if q1 > q0: return 1
        return random.choice([0, 1])

    def update(self, state, action, reward, next_state):
        current_q = self.get_q(state, action)
        max_next_q = max(self.get_q(next_state, a) for a in self.actions)
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.Q[(state, action)] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)