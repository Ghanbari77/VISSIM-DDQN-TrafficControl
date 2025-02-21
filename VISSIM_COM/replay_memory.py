from collections import deque

class ReplayMemory:
    def __init__(self, max_size):
        self.memory = deque(maxlen=max_size)

    def store_experience(self, state, action, reward, next_state):
        """Stores experiences for training."""
        self.memory.append((state, action, reward, next_state))

    def sample_batch(self, batch_size):
        """Samples a batch of experiences."""
        return random.sample(self.memory, min(len(self.memory), batch_size))
