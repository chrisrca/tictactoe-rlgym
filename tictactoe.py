from typing import Any, Dict, List, Optional
import numpy as np
from dataclasses import dataclass
from rlgym.api import RLGym, TransitionEngine, StateMutator, ObsBuilder, ActionParser, RewardFunction, DoneCondition
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium.spaces import Discrete
import argparse
import wandb

# Callback for periodic evaluation
class EvalCallback(BaseCallback):
    def __init__(self, eval_freq_rollouts, model, env, wandb_run, verbose=1):
        super(EvalCallback, self).__init__(verbose)
        self.eval_freq_rollouts = eval_freq_rollouts
        self.model = model
        self.env = env
        self.wandb_run = wandb_run
        self.rollout_count = 0

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        self.rollout_count += 1
        if self.rollout_count % self.eval_freq_rollouts == 0:
            print(f"Evaluating model after {self.rollout_count} rollouts...")
            evaluate_model(self.model, self.env, wandb_run=self.wandb_run)

@dataclass
class TicTacToeState:
    board: np.ndarray  # shape (9,) 0=empty, 1=X (agent), -1=O (opponent)
    steps: int = 0

class TicTacToeEngine(TransitionEngine[int, TicTacToeState, int]):
    def __init__(self, play_mode: bool = False):
        self._state = None
        self._config = {}
        self.play_mode = play_mode  # Flag to disable random opponent moves in play mode

    @property
    def agents(self) -> List[int]:
        return [0]

    @property
    def max_num_agents(self) -> int:
        return 1

    @property
    def state(self) -> TicTacToeState:
        return self._state

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @config.setter
    def config(self, value: Dict[str, Any]):
        self._config = value

    def step(self, actions: Dict[int, int], shared_info: Dict[str, Any]) -> TicTacToeState:
        action = actions[0]
        if self._state.board[action] != 0:  # Invalid move
            shared_info['invalid_move'] = True
            return self._state
        shared_info['invalid_move'] = False
        self._state.board[action] = 1  # Agent places X
        self._state.steps += 1
        # Only apply random opponent move if not in play mode
        if not self.play_mode and not self.check_win(1) and not self.is_draw():
            available = np.where(self._state.board == 0)[0]
            if len(available) > 0:
                opp_action = np.random.choice(available)
                self._state.board[opp_action] = -1
                self._state.steps += 1
        return self._state

    def check_win(self, player: int) -> bool:
        wins = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
        for w in wins:
            if all(self._state.board[i] == player for i in w):
                return True
        return False

    def is_draw(self) -> bool:
        return np.all(self._state.board != 0)

    def create_base_state(self) -> TicTacToeState:
        return TicTacToeState(np.zeros(9, dtype=int), 0)

    def reset(self, initial_state: Optional[TicTacToeState] = None) -> None:
        self._state = initial_state if initial_state is not None else self.create_base_state()

    def set_state(self, desired_state: Optional[TicTacToeState], shared_info: Dict[str, Any]) -> TicTacToeState:
        if desired_state is None:
            self._state = self.create_base_state()
        else:
            self._state = TicTacToeState(
                board=desired_state.board.copy(),
                steps=desired_state.steps
            )
        return self._state

class TicTacToeMutator(StateMutator[TicTacToeState]):
    def apply(self, state: TicTacToeState, shared_info: Dict[str, Any]) -> None:
        state.board = np.zeros(9, dtype=int)
        state.steps = 0

class TicTacToeObs(ObsBuilder[int, np.ndarray, TicTacToeState, np.ndarray]):
    def get_obs_space(self, agent: int) -> np.ndarray:
        return np.zeros(9, dtype=np.float32)

    def reset(self, agents: List[int], initial_state: TicTacToeState, shared_info: Dict[str, Any]) -> None:
        pass

    def build_obs(self, agents: List[int], state: TicTacToeState, shared_info: Dict[str, Any]) -> Dict[int, np.ndarray]:
        observations = {}
        for agent in agents:
            observations[agent] = state.board.astype(np.float32)
        return observations

class TicTacToeActions(ActionParser[int, int, int, TicTacToeState, int]):
    def get_action_space(self, agent: int) -> int:
        return 9  # Discrete 0-8

    def reset(self, agents: List[int], initial_state: TicTacToeState, shared_info: Dict[str, Any]) -> None:
        pass

    def parse_actions(self, actions: Dict[int, int], state: TicTacToeState, shared_info: Dict[str, Any]) -> Dict[int, int]:
        return actions

class TicTacToeReward(RewardFunction[int, TicTacToeState, float]):
    def __init__(self, invalid_move_penalty: float = -0.5):
        self.invalid_move_penalty = invalid_move_penalty

    def reset(self, agents: List[int], initial_state: TicTacToeState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[int], state: TicTacToeState, 
                   is_terminated: Dict[int, bool], is_truncated: Dict[int, bool],
                   shared_info: Dict[str, Any]) -> Dict[int, float]:
        rewards = {}
        for agent in agents:
            if shared_info.get('invalid_move', False):
                rewards[agent] = self.invalid_move_penalty
            elif self.check_win(state.board, 1):
                rewards[agent] = 1.0
            elif self.check_win(state.board, -1):
                rewards[agent] = -1.0
            elif self.is_draw(state.board):
                rewards[agent] = 0.0
            else:
                rewards[agent] = 0.0
        return rewards

    def check_win(self, board: np.ndarray, player: int) -> bool:
        wins = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
        for w in wins:
            if all(board[i] == player for i in w):
                return True
        return False

    def is_draw(self, board: np.ndarray) -> bool:
        return bool(np.all(board != 0))

class TicTacToeTerminalCondition(DoneCondition[int, TicTacToeState]):
    def reset(self, agents: List[int], initial_state: TicTacToeState, shared_info: Dict[str, Any]) -> None:
        pass

    def is_done(self, agents: List[int], state: TicTacToeState, shared_info: Dict[str, Any]) -> Dict[int, bool]:
        done = self.check_win(state.board, 1) or self.check_win(state.board, -1) or self.is_draw(state.board)
        return {agent: bool(done) for agent in agents}

    def check_win(self, board: np.ndarray, player: int) -> bool:
        wins = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
        for w in wins:
            if all(board[i] == player for i in w):
                return True
        return False

    def is_draw(self, board: np.ndarray) -> bool:
        return bool(np.all(board != 0))

class TicTacToeTruncatedCondition(DoneCondition[int, TicTacToeState]):
    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps

    def reset(self, agents: List[int], initial_state: TicTacToeState, shared_info: Dict[str, Any]) -> None:
        pass

    def is_done(self, agents: List[int], state: TicTacToeState, shared_info: Dict[str, Any]) -> Dict[int, bool]:
        done = state.steps >= self.max_steps
        return {agent: bool(done) for agent in agents}

class RLGymWrapper(gym.Env):
    def __init__(self, rlgym_env):
        super().__init__()
        self.env = rlgym_env
        self.truncation_cond = TicTacToeTruncatedCondition(max_steps=10)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)
        self.action_space = Discrete(9)
        self.metadata = {'render.modes': []}
        self.reward_range = (-float('inf'), float('inf'))

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs = self.env.reset()
        return obs[0], {}

    def step(self, action):
        action_dict = {0: action}
        obs, reward, terminated, info = self.env.step(action_dict)
        state = self.env.transition_engine.state
        shared_info = info.get('shared_info', {})
        truncated = self.truncation_cond.is_done([0], state, shared_info)[0]
        terminated_bool = bool(terminated[0])
        return obs[0], reward[0], terminated_bool, truncated, info

def evaluate_model(model, env, num_episodes=100, wandb_run=None):
    wins = 0
    draws = 0
    losses = 0
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                if reward == 1.0:
                    wins += 1
                elif reward == -1.0:
                    losses += 1
                elif reward == 0.0 and not info.get('invalid_move', False):
                    draws += 1
    total = num_episodes
    win_rate = wins / total
    print(f"Evaluation: Wins={wins}, Draws={draws}, Losses={losses}, Win Rate={win_rate:.3f}")
    
    # Log to W&B if available
    if wandb_run:
        wandb_run.log({
            "eval/wins": wins,
            "eval/draws": draws,
            "eval/losses": losses,
        })
    
    return wins, draws, losses

def print_board(board: np.ndarray):
    symbols = {0: ' ', 1: 'X', -1: 'O'}
    print("\nCurrent board:")
    for i in range(3):
        row = ' | '.join(symbols[board[3*i + j]] for j in range(3))
        print(row)
        if i < 2:
            print('---------')
    print()

def check_winner(board: np.ndarray) -> str:
    wins = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
    for w in wins:
        if all(board[i] == 1 for i in w):
            return "Model (X) wins!"
        elif all(board[i] == -1 for i in w):
            return "You (O) win!"
    if np.all(board != 0):
        return "It's a draw!"
    return "Game continues..."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tic-Tac-Toe: Train or play against the model")
    parser.add_argument('--train', action='store_true', help="Train the model (default: play mode)")
    args = parser.parse_args()

    # Build RLGym environment
    env = RLGym(
        state_mutator=TicTacToeMutator(),
        obs_builder=TicTacToeObs(),
        action_parser=TicTacToeActions(),
        reward_fn=TicTacToeReward(),
        transition_engine=TicTacToeEngine(play_mode=not args.train),
        termination_cond=TicTacToeTerminalCondition(),
        truncation_cond=TicTacToeTruncatedCondition(),
    )
    wrapped_env = RLGymWrapper(env)

    if args.train:
        # W&B
        config = {
            "policy_type": "MlpPolicy",
            "total_timesteps": 300000,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "env": "TicTacToe",
            "max_steps_per_episode": 10,
            "eval_episodes": 100,
        }
        run = wandb.init(
            project="tictactoe-rl",
            config=config,
            sync_tensorboard=True,
            save_code=True,
        )

        # Verify environment
        check_env(wrapped_env)

        # Initialize PPO model
        model = PPO(
            config["policy_type"],
            wrapped_env,
            verbose=1,
            tensorboard_log=f"runs/{run.id}",
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            clip_range=config["clip_range"],
            ent_coef=config["ent_coef"],
        )

        # Evaluate before training
        print("Evaluating untrained model...")
        evaluate_model(model, wrapped_env, wandb_run=run)

        # Train the model
        print("Starting training...")
        eval_callback = EvalCallback(
            eval_freq_rollouts=20,
            model=model,
            env=wrapped_env,
            wandb_run=run,
        )
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=eval_callback,
        )

        # Evaluate after training
        print("Evaluating trained model...")
        evaluate_model(model, wrapped_env, wandb_run=run)

        # Save the final model
        model.save("tictactoe_ppo_model")

        run.finish()
    else:
        # Load the trained model
        print("Loading trained model...")
        try:
            model = PPO.load("tictactoe_ppo_model")
        except FileNotFoundError:
            print("Error: Trained model 'tictactoe_ppo_model' not found. Please run with --train first.")
            exit(1)

        # Start game
        obs, _ = wrapped_env.reset()
        state = wrapped_env.env.transition_engine.state
        done = False
        print("Tic-Tac-Toe: You are O, model is X (goes first). Enter moves 0-8.")
        print("\nBoard positions:")
        print("0 | 1 | 2")
        print("---------")
        print("3 | 4 | 5")
        print("---------")
        print("6 | 7 | 8")

        while not done:
            # Model's turn (X)
            print("\nModel's turn...")
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = wrapped_env.step(action)
            state = wrapped_env.env.transition_engine.state
            obs = state.board.astype(np.float32)
            print_board(state.board)
            winner = check_winner(state.board)
            print(winner)
            done = terminated or truncated
            if done:
                break

            # Human's turn (O)
            print("\nYour turn (O):")
            while True:
                try:
                    human_action = int(input("Enter your move (0-8): "))
                    if 0 <= human_action < 9 and state.board[human_action] == 0:
                        state.board[human_action] = -1
                        state.steps += 1
                        obs = state.board.astype(np.float32)
                        break
                    else:
                        print("Invalid move! Choose an empty spot (0-8).")
                except ValueError:
                    print("Invalid input! Enter a number 0-8.")
            print_board(state.board)
            print(f"After human move: steps={state.steps}")
            winner = check_winner(state.board)
            print(winner)
            done = TicTacToeTerminalCondition().is_done([0], state, {})[0] or \
                wrapped_env.truncation_cond.is_done([0], state, {})[0]
        
        print("\nGame over!")