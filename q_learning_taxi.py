import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import time

BASE_TOTAL_EPISODES = 10000 # Graph plateau at around 5000, so 10000 is more than enough
EVAL_EPISODES = 5
RENDER_EVALUATION = False # Set true if you want to see vroom vroom
RESULTS_DIR = "experiment_results"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def moving_average(data, window_size):
    if len(data) < window_size:
        return np.array(data)
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def run_q_learning_experiment(
    learning_rate, gamma, initial_epsilon,
    min_epsilon, max_epsilon, decay_rate,
    q_table_init_strategy="zeros",
    total_episodes=BASE_TOTAL_EPISODES,
    experiment_name="default_experiment"
):
    print(f"\n--- Experiment: {experiment_name} | Params: LR={learning_rate}, Gamma={gamma}, Decay={decay_rate}, Q_Init={q_table_init_strategy} ---")

    env = gym.make("Taxi-v3")
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    if q_table_init_strategy == "random":
        q_table = np.random.uniform(low=-0.01, high=0.01, size=(num_states, num_actions))
    else:
        q_table = np.zeros((num_states, num_actions))

    epsilon = initial_epsilon
    episode_rewards = []
    episode_lengths = []

    for episode in range(total_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        current_episode_reward = 0
        current_episode_length = 0

        while not terminated and not truncated:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)

            # Q-Learning update rule
            q_table[state, action] = q_table[state, action] + learning_rate * \
                                     (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])
            state = new_state
            current_episode_reward += reward
            current_episode_length += 1

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        episode_rewards.append(current_episode_reward)
        episode_lengths.append(current_episode_length)

        if (episode + 1) % (total_episodes // 10) == 0 and total_episodes >=10:
             avg_reward_last_100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
             if not episode_rewards: avg_reward_last_100 = 0
             print(f"  Ep: {episode + 1}/{total_episodes} | AvgRew(100): {avg_reward_last_100:.2f} | Eps: {epsilon:.4f}")
    env.close()

    window = 100
    smoothed_rewards = moving_average(episode_rewards, window)
    smoothed_lengths = moving_average(episode_lengths, window)
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    reward_x_axis = range(window - 1, len(episode_rewards)) if len(episode_rewards) >= window else range(len(episode_rewards))
    axs[0].plot(reward_x_axis, smoothed_rewards); axs[0].set_title(f'Rewards - {experiment_name}'); axs[0].set_ylabel('Avg Reward'); axs[0].grid(True)
    length_x_axis = range(window - 1, len(episode_lengths)) if len(episode_lengths) >= window else range(len(episode_lengths))
    axs[1].plot(length_x_axis, smoothed_lengths, color='green'); axs[1].set_title(f'Lengths - {experiment_name}'); axs[1].set_ylabel('Avg Length'); axs[1].set_xlabel('Episode'); axs[1].grid(True)
    plt.tight_layout()
    safe_exp_name = experiment_name.replace('=', '_').replace(',', '_').replace(':', '_')
    plot_filename = os.path.join(RESULTS_DIR, f"plot_{safe_exp_name}.png")
    plt.savefig(plot_filename); plt.close(fig)
    print(f"Plot saved: {plot_filename}")

    eval_env = gym.make("Taxi-v3", render_mode="human" if RENDER_EVALUATION else None)
    total_eval_rewards, total_eval_steps = 0, 0
    for _ in range(EVAL_EPISODES):
        state, _ = eval_env.reset(); terminated, truncated = False, False
        current_eval_episode_reward, current_eval_episode_steps = 0, 0
        for _ in range(200): # Max steps per eval episode
            action = np.argmax(q_table[state, :])
            new_state, reward, terminated, truncated, _ = eval_env.step(action)
            state = new_state; current_eval_episode_reward += reward; current_eval_episode_steps += 1
            if terminated or truncated: break
        total_eval_rewards += current_eval_episode_reward; total_eval_steps += current_eval_episode_steps
    eval_env.close()
    avg_eval_reward = total_eval_rewards / EVAL_EPISODES
    avg_eval_steps = total_eval_steps / EVAL_EPISODES
    print(f"Evaluation: AvgRew={avg_eval_reward:.2f}, AvgSteps={avg_eval_steps:.2f}")
    
    final_avg_reward_training = np.mean(episode_rewards[-(total_episodes//10):]) if total_episodes >=10 else np.mean(episode_rewards)
    if not episode_rewards: final_avg_reward_training = 0

    return {"experiment_name": experiment_name, "learning_rate": learning_rate, "gamma": gamma,
            "decay_rate": decay_rate, "q_table_init_strategy": q_table_init_strategy,
            "final_avg_training_reward": final_avg_reward_training, "avg_eval_reward": avg_eval_reward,
            "avg_eval_steps": avg_eval_steps, "plot_filename": plot_filename}

if __name__ == "__main__":
    all_results = []
    experiment_timestamp = time.strftime("%Y%m%d-%H%M%S")
    current_experiment_set_dir = os.path.join(RESULTS_DIR, experiment_timestamp)
    if not os.path.exists(current_experiment_set_dir): os.makedirs(current_experiment_set_dir)
    RESULTS_DIR = current_experiment_set_dir

    # --- Baseline Hyperparameters ---
    baseline_lr = 0.1; baseline_gamma = 0.99; baseline_initial_epsilon = 1.0
    baseline_min_epsilon = 0.01; baseline_max_epsilon = 1.0
    baseline_decay_rate = 0.0005; baseline_q_init = "zeros"

    # --- Experiment Set 1: Varying Learning Rates (α) ---
    print("\n\n=== EXPERIMENT SET 1: VARYING LEARNING RATES ===")
    learning_rates_to_test = [0.01, 0.05, 0.1, 0.3, 0.5]
    for lr_val in learning_rates_to_test:
        all_results.append(run_q_learning_experiment(
            learning_rate=lr_val, gamma=baseline_gamma, initial_epsilon=baseline_initial_epsilon,
            min_epsilon=baseline_min_epsilon, max_epsilon=baseline_max_epsilon,
            decay_rate=baseline_decay_rate, q_table_init_strategy=baseline_q_init,
            experiment_name=f"lr_{lr_val}"
        ))

    # --- Experiment Set 2: Varying Discount Factors (γ) ---
    print("\n\n=== EXPERIMENT SET 2: VARYING DISCOUNT FACTORS ===")
    gammas_to_test = [0.9, 0.95, 0.99, 0.999]
    for gamma_val in gammas_to_test:
        all_results.append(run_q_learning_experiment(
            learning_rate=baseline_lr, gamma=gamma_val, initial_epsilon=baseline_initial_epsilon,
            min_epsilon=baseline_min_epsilon, max_epsilon=baseline_max_epsilon,
            decay_rate=baseline_decay_rate, q_table_init_strategy=baseline_q_init,
            experiment_name=f"gamma_{gamma_val}"
        ))

    # --- Experiment Set 3: Varying Epsilon Decay Rates ---
    print("\n\n=== EXPERIMENT SET 3: VARYING EPSILON DECAY RATES ===")
    decay_rates_to_test = [0.0001, 0.0005, 0.001, 0.005]
    for decay_val in decay_rates_to_test:
        all_results.append(run_q_learning_experiment(
            learning_rate=baseline_lr, gamma=baseline_gamma, initial_epsilon=baseline_initial_epsilon,
            min_epsilon=baseline_min_epsilon, max_epsilon=baseline_max_epsilon,
            decay_rate=decay_val, q_table_init_strategy=baseline_q_init,
            experiment_name=f"decay_{decay_val}"
        ))

    # --- Experiment Set 4: Varying Q-Table Initialization ---
    print("\n\n=== EXPERIMENT SET 4: VARYING Q-TABLE INITIALIZATION ===")
    q_init_strategies_to_test = ["zeros", "random"]
    for q_init_val in q_init_strategies_to_test:
        all_results.append(run_q_learning_experiment(
            learning_rate=baseline_lr, gamma=baseline_gamma, initial_epsilon=baseline_initial_epsilon,
            min_epsilon=baseline_min_epsilon, max_epsilon=baseline_max_epsilon,
            decay_rate=baseline_decay_rate, q_table_init_strategy=q_init_val,
            experiment_name=f"q_init_{q_init_val}"
        ))

    print("\n\n=== SUMMARY OF ALL EXPERIMENTS ===")
    print(f"{'Experiment Name':<25} | {'LR':<5} | {'Gamma':<5} | {'Decay':<7} | {'Q-Init':<7} | {'Train Rew (Final Avg)':<20} | {'Eval Rew':<10} | {'Eval Steps':<10}")
    print("-" * 120)
    for res_item in all_results:
        print(f"{res_item['experiment_name']:<25} | {res_item['learning_rate']:<5} | {res_item['gamma']:<5} | {res_item['decay_rate']:<7.4f} | {res_item['q_table_init_strategy']:<7} | {res_item['final_avg_training_reward']:<20.2f} | {res_item['avg_eval_reward']:<10.2f} | {res_item['avg_eval_steps']:<10.2f}")

    print(f"\nResults saved in: {RESULTS_DIR}")
    print("Experimentation complete.")
