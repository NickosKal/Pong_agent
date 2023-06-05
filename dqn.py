import random
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward, terminate):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward, terminate)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward)
        """
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))


class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.env_name = env_config["env_name"]
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.epsilon_reduction_step = (self.eps_start - self.eps_end) / self.anneal_length
        self.n_actions = env_config["n_actions"]

        self.obs_stack_size = env_config["obs_stack_size"]

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)

        self.fc1 = nn.Linear(in_features=3136, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def reduce_epsilon(self):
        """
        Method that reduces self.eps_start linearly with self.epsilon_reduction
        length while it's greater that self.eps_end
        """
        if self.eps_start > self.eps_end:
            self.eps_start -= self.epsilon_reduction_step

    def act(self, observation, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        # TODO: Implement action selection using the Deep Q-network. This function
        #       takes an observation tensor and should return a tensor of actions.
        #       For example, if the state dimension is 4 and the batch size is 32,
        #       the input would be a [32, 4] tensor and the output a [32, 1] tensor.
        # TODO: Implement epsilon-greedy exploration.

        n_observations = observation.size(0)
        predictions = self.forward(observation)  # predicted action values for each observation

        random_number = random.random()
        if exploit or random_number > self.eps_start:
            actions = torch.argmax(predictions, dim=1).long() + 2  # returns a tensor with indices of max values
        else:
            actions = torch.randint(low=2, high=4, size=(n_observations, 1))  # return random actions for each obs
        return actions


def optimize(dqn, target_dqn, memory, optimizer):
    """
    This function samples a batch from the replay buffer and optimizes the
    Q-network.
    """

    # If we don't have enough transitions stored yet, we don't train
    if len(memory) < dqn.batch_size:
        return

    # If enough transitions, sample batch from memory
    batch = memory.sample(dqn.batch_size)

    # TODO: Create 4 separate tensors for observations, actions, next observations,
    #       rewards and move to GPU if available
    observations = torch.cat(batch[0]).to(device)
    actions = torch.cat([torch.tensor(action).unsqueeze(0) for action in batch[1]], dim=0).to(device)
    rewards = torch.cat([torch.tensor(reward).unsqueeze(0) for reward in batch[3]], dim=0).to(device)

    # For next observations, need to handle terminal states as special case
    non_terminal_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch[2])), device=device, dtype=torch.bool
    )
    non_terminal_next_obs = torch.cat([s for s in batch[2] if s is not None])

    # TODO: Compute Q-values for observations using the policy network
    #       This is Q(s, a; theta_i), and uses the most updated weights
    #       Need to convert back the action values in order for selection to work
    q_values = dqn.forward(observations).gather(1, torch.sub(actions, 2).unsqueeze(1),)

    # TODO: Compute the Q-value targets for next obs, using target network
    #       This is y_i = E[r + gamma * max_a Q(s', a'; theta_i-1], and uses "old" weights
    #       from the policy network
    # For terminal states, the action value is 0
    target_action_val = torch.zeros(target_dqn.batch_size, device=device)
    with torch.no_grad():  # Context manager to speed up computation
        target_action_val[non_terminal_mask] = target_dqn.forward(non_terminal_next_obs).max(1)[0]
    q_value_targets = rewards + target_dqn.gamma * target_action_val

    # Compute the loss with current weights
    loss = F.mse_loss(q_values.squeeze(), q_value_targets)

    # Perform gradient descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()




