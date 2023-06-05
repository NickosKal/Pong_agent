import argparse
import copy

import gymnasium as gym
import matplotlib.pyplot as plt
import torch

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v1', 'ALE/Pong-v5'], default='CartPole-v1')
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'ALE/Pong-v5': config.Pong
}


def plotting(env_name, n_episodes, return_value, show=False):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set(xlabel="Episode", ylabel="Mean Return", title=f"Mean return during training for {args.env}")
    x, y = zip(*return_value)
    ax.plot(x, y, color="magenta")
    path = 'plots'
    plt.savefig(f"{path}/{args.env}.png")
    if show:
        plt.show()


if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    env_config = ENV_CONFIGS[args.env]
    env_name = env_config['env_name']

    env = gym.wrappers.AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    # TODO: Create and initialize target Q-network.
    target_dqn = copy.deepcopy(dqn).to(device)
    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of the best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")

    n_episodes = env_config['n_episodes']
    steps = 0  # number of steps taken during the entire training
    evaluate_freq = 25  # How often to run evaluation
    evaluation_episodes = 4  # Number of evaluation episodes
    mean_return_train = []  # list that will contain mean return for each evaluation phase
    k = 4  # take a new action in every kth frame instead of every frame

    for episode in range(env_config['n_episodes']):
        terminated = False
        obs, info = env.reset()

        obs = preprocess(obs, env=args.env).unsqueeze(0)

        obs_stack = torch.cat(dqn.obs_stack_size * [obs]).unsqueeze(0).to(device)
        next_obs_stack = torch.cat(dqn.obs_stack_size * [obs]).unsqueeze(0).to(device)
        while not terminated:
            dqn.reduce_epsilon()

            action = dqn.act(obs_stack).item()
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = preprocess(next_obs, env=env_name).unsqueeze(0).to(device)

            next_obs_stack = torch.cat((next_obs_stack[:, 1:, ...], next_obs.unsqueeze(1)), dim=1).to(device)

            action = torch.tensor(action)
            reward = torch.tensor(reward)

            memory.push(obs_stack, action, next_obs_stack, reward, 1 - int(terminated))

            if not terminated:
                obs_stack = next_obs_stack

            steps += 1

            # TODO: Run DQN.optimize() every env_config["train_frequency"] steps.
            train_frequency = env_config['train_frequency']
            if steps % train_frequency == 0:
                optimize(dqn, target_dqn, memory, optimizer)

            # TODO: Update the target network every env_config["target_update_frequency"] steps.
            target_update_frequency = env_config["target_update_frequency"]
            if steps % target_update_frequency == 0:
                target_dqn = copy.deepcopy(dqn)

        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
            mean_return_train.append((episode, mean_return))
            print(f'Episode {episode + 1}/{env_config["n_episodes"]}: {mean_return}')

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving models.')
                torch.save(dqn, f'models/{args.env}_best.pt')

                plotting(env_name=env, n_episodes=env_config["n_episodes"], return_value=mean_return_train, show=False)

    # Close environment after training is completed.
    env.close()
