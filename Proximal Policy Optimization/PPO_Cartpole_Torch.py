import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import scipy.signal
import matplotlib.pyplot as plt

def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, -discount], x[::-1], axis=0)[::-1]


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        self.observation_buffer = np.zeros((size, observation_dimensions), dtype=np.float32)
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data from the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / (advantage_std + 1e-8)
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )


class MLP(nn.Module):
    def __init__(self, sizes, activation=nn.Tanh, output_activation=None):
        super(MLP, self).__init__()
        layers = []
        for j in range(len(sizes) - 1):
            act = activation if j < len(sizes) - 2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j + 1])]
            if act is not None:
                layers += [act()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def logprobabilities(logits, a):
    # Compute log-probabilities of actions using the logits from the actor
    logp_all = nn.functional.log_softmax(logits, dim=1)
    logp = logp_all.gather(1, a.unsqueeze(1)).squeeze(1)
    return logp


# Sample action from the actor
def sample_action(observation):
    logits = actor(observation)
    action_distribution = torch.distributions.Categorical(logits=logits)
    action = action_distribution.sample()
    return logits, action


# Train the policy by maximizing the PPO-Clip objective
def train_policy(observation_buffer, action_buffer, logprobability_buffer, advantage_buffer):
    observation_buffer = torch.tensor(observation_buffer, dtype=torch.float32)
    action_buffer = torch.tensor(action_buffer, dtype=torch.int64)
    logprobability_buffer = torch.tensor(logprobability_buffer, dtype=torch.float32)
    advantage_buffer = torch.tensor(advantage_buffer, dtype=torch.float32)

    for _ in range(train_policy_iterations):
        logits = actor(observation_buffer)
        new_logprobabilities = logprobabilities(logits, action_buffer)
        ratio = torch.exp(new_logprobabilities - logprobability_buffer)
        surrogate1 = ratio * advantage_buffer
        surrogate2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage_buffer
        loss = -torch.mean(torch.min(surrogate1, surrogate2))

        # Compute gradients and update parameters
        policy_optimizer.zero_grad()
        loss.backward()
        policy_optimizer.step()

        # Approximate KL divergence
        with torch.no_grad():
            kl = torch.mean(logprobability_buffer - new_logprobabilities).item()
        if kl > 1.5 * target_kl:
            break


# Train the value function by regression on mean-squared error
def train_value_function(observation_buffer, return_buffer):
    observation_buffer = torch.tensor(observation_buffer, dtype=torch.float32)
    return_buffer = torch.tensor(return_buffer, dtype=torch.float32)

    for _ in range(train_value_iterations):
        value = critic(observation_buffer).squeeze(1)
        loss = nn.functional.mse_loss(value, return_buffer)

        # Compute gradients and update parameters
        value_optimizer.zero_grad()
        loss.backward()
        value_optimizer.step()


# Hyperparameters
steps_per_epoch = 4000
epochs = 30
gamma = 0.99
clip_ratio = 0.2
policy_learning_rate = 3e-4
value_function_learning_rate = 1e-3
train_policy_iterations = 80
train_value_iterations = 80
lam = 0.97
target_kl = 0.01
hidden_sizes = (64, 64)
render = False  # Set to True if you want to render the environment

# Initialize environment and get dimensions
env = gym.make("CartPole-v1")
observation_dimensions = env.observation_space.shape[0]
num_actions = env.action_space.n

# Initialize buffer
buffer = Buffer(observation_dimensions, steps_per_epoch, gamma, lam)

# Initialize actor and critic models
actor = MLP([observation_dimensions] + list(hidden_sizes) + [num_actions], activation=nn.Tanh)
critic = MLP([observation_dimensions] + list(hidden_sizes) + [1], activation=nn.Tanh)

# Initialize optimizers
policy_optimizer = optim.Adam(actor.parameters(), lr=policy_learning_rate)
value_optimizer = optim.Adam(critic.parameters(), lr=value_function_learning_rate)

# Initialize environment variables
observation, info = env.reset()
episode_return, episode_length = 0, 0

mean_returns = []
mean_lengths = []

# Main training loop
for epoch in range(epochs):
    sum_return = 0
    sum_length = 0
    num_episodes = 0

    for t in range(steps_per_epoch):
        if render:
            env.render()

        # Prepare observation
        observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        logits, action = sample_action(observation_tensor)
        value_t = critic(observation_tensor).squeeze(1).item()
        logprobability_t = logprobabilities(logits, action).item()

        # Take action in the environment
        action_np = action.item()
        observation_new, reward, done, truncated, info = env.step(action_np)
        episode_return += reward
        episode_length += 1

        # Store data in buffer
        buffer.store(observation, action_np, reward, value_t, logprobability_t)

        # Update observation
        observation = observation_new

        # Finish trajectory if terminal
        if done or (t == steps_per_epoch - 1):
            last_value = 0 if done else critic(
                torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            ).item()
            buffer.finish_trajectory(last_value)
            sum_return += episode_return
            sum_length += episode_length
            num_episodes += 1
            observation, info = env.reset()
            episode_return, episode_length = 0, 0

    # Get data from buffer
    (
        observation_buffer,
        action_buffer,
        advantage_buffer,
        return_buffer,
        logprobability_buffer,
    ) = buffer.get()

    # Update policy
    train_policy(
        observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
    )

    # Update value function
    train_value_function(observation_buffer, return_buffer)

    # Record metrics
    mean_return = sum_return / num_episodes
    mean_length = sum_length / num_episodes
    mean_returns.append(mean_return)
    mean_lengths.append(mean_length)

    # Print training progress
    print(
        f"Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
    )

# Plot Mean Return and Mean Length over Epochs
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(mean_returns)
plt.xlabel('Epoch')
plt.ylabel('Mean Return')
plt.title('Mean Return over Epochs')

plt.subplot(1, 2, 2)
plt.plot(mean_lengths)
plt.xlabel('Epoch')
plt.ylabel('Mean Episode Length')
plt.title('Mean Episode Length over Epochs')

plt.tight_layout()
plt.show()

# Visualize trained agent
render = True  # Set to True to render the environment

test_episodes = 5

for episode in range(test_episodes):
    observation, info = env.reset()
    done = False
    episode_return = 0

    while not done:
        if render:
            env.render()
        observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = actor(observation_tensor)
            action_distribution = torch.distributions.Categorical(logits=logits)
            action = action_distribution.sample().item()
        observation, reward, done, truncated, info = env.step(action)
        episode_return += reward

    print(f"Test Episode {episode + 1}: Return = {episode_return}")

env.close()
