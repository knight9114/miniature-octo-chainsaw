# -------------------------------------------------------------------------
#   REINFORCE Example
# -------------------------------------------------------------------------

# Standard Library Imports
from argparse import ArgumentParser, Namespace

# Special Imports
import numpy as np
import gymnasium as gym
import torch
from torch import nn, optim, Tensor
from torch.distributions import Categorical


# -------------------------------------------------------------------------
#   Policy Network
# -------------------------------------------------------------------------

class PolicyNetwork(nn.Sequential):
    def __init__(
        self,
        d_inputs: int,
        d_hidden: int,
        n_actions: int,
    ) -> None:
        super().__init__(
            nn.Linear(d_inputs, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_hidden * 2),
            nn.Tanh(),
            nn.Linear(d_hidden * 2, n_actions),
        )

    def act(self, state: Tensor) -> tuple[Tensor, Tensor]:
        policy = self(state)
        distribution = Categorical(logits=policy)
        action = distribution.sample()
        logprob = distribution.log_prob(action)

        return action, logprob



# -------------------------------------------------------------------------
#   REINFORCE Trainer
# -------------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        n_episodes: int = 5000,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon: float = 1e-6,
        device: torch.device | str = "cpu",
    ) -> None:
        self.n_episodes = n_episodes
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.ctx = device

    def fit(
        self,
        agent: PolicyNetwork,
        env: gym.Env,
        seed: float | int | None = None,
    ) -> None:
        seed = torch.randint(1 << 32, [1]).item()
        torch.manual_seed(seed)

        env = gym.wrappers.RecordEpisodeStatistics(env, 50)
        agent = agent.to(self.ctx)
        optimizer = optim.AdamW(
            params=agent.parameters(),
            lr=self.lr,
        )

        global_rewards = []
        for episode in range(self.n_episodes):
            rewards = []
            logprobs = []

            obs, info = env.reset(seed=seed + episode)
            done = False

            steps = 0
            while not done:
                state = torch.from_numpy(obs.reshape([1, *obs.shape])).to(self.ctx)
                action, logprob = agent.act(state)
                steps += 1

                obs, reward, terminated, truncated, info = env.step(action.item())
                rewards.append(reward)
                logprobs.append(logprob)

                done = terminated or truncated
            
            global_rewards.append(env.return_queue[-1])

            optimizer.zero_grad()
            loss = self.compute_update(rewards, logprobs, self.gamma)
            loss.backward()
            optimizer.step()

            print(f"Episode #{episode}")
            print(f"    Reward: {np.mean(env.return_queue):.4f}")
            print(f"    Steps:  {steps}")
            print()

    @staticmethod
    def compute_update(
        rewards: list[float],
        logprobs: list[float],
        gamma: float,
    ) -> Tensor:
        running_g = 0
        gs = []

        for reward in rewards[::-1]:
            running_g = reward + gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0
        for logprob, delta in zip(logprobs, deltas):
            loss += logprob.mean() * delta * -1

        return loss


def main(argv: tuple[str] | None = None):
    args = parse_cli_args(argv)
    env = gym.make(args.name)
    agent = PolicyNetwork(
        np.prod(env.observation_space.shape),
        args.d_hidden,
        env.action_space.n,
    )
    trainer = Trainer(
        args.n_episodes,
        args.lr,
        args.gamma,
        args.epsilon,
        args.device,
    )
    trainer.fit(agent, env, args.seed)


def parse_cli_args(argv: tuple[str] | None = None) -> Namespace:
    parser = ArgumentParser()
    
    env = parser.add_argument_group("Environment Arguments")
    env.add_argument("--name", default="CartPole-v1")

    agent = parser.add_argument_group("Agent Arguments")
    agent.add_argument("--d-hidden", type=int, default=16)
    agent.add_argument("--lr", type=float, default=1e-4)

    train = parser.add_argument_group("Trainer Arguments")
    train.add_argument("--n-episodes", type=int, default=5000)
    train.add_argument("--gamma", type=float, default=0.99)
    train.add_argument("--epsilon", type=float, default=1e-6)
    train.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    train.add_argument("--seed", type=int)

    return parser.parse_args(argv)


if __name__ == "__main__":
    main()