import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt # For plotting training progress

# --- 1. Initialize the Environment ---
# render_mode="human" to watch the agent, or None for faster training
# env = gym.make("Taxi-v3", render_mode="human")
env = gym.make("Taxi-v3") # No rendering for faster training

print(f"Action Space: {env.action_space}")
print(f"Observation Space: {env.observation_space}")

num_states = env.observation_space.n
num_actions = env.action_space.n
print(f"Number of States: {num_states}") # Should be 500 for Taxi-v3
print(f"Number of Actions: {num_actions}") # Should be 6 for Taxi-v3

# --- 2. Initialize the Q-table ---
# The Q-table will have dimensions (number of states, number of actions)
# Initialize with zeros, or small random values. Zeros is common.
q_table = np.zeros((num_states, num_actions))

print("Q-table initialized with shape:", q_table.shape)
# print(q_table) # Uncomment to see the initial Q-table

# --- 3. Define Hyperparameters ---
total_episodes = 20000      # Total episodes for training
learning_rate = 0.1         # Alpha: How much new information overrides old information
gamma = 0.99                # Discount factor: Importance of future rewards
epsilon = 1.0               # Exploration rate: Initial probability of taking a random action
max_epsilon = 1.0           # Maximum exploration probability
min_epsilon = 0.01          # Minimum exploration probability
decay_rate = 0.0005         # Exponential decay rate for exploration prob after each episode

# For tracking rewards and episode lengths
episode_rewards = []
episode_lengths = []

# --- 4. Q-Learning Training Loop ---
print("\n--- Starting Training ---")
for episode in range(total_episodes):
    state, info = env.reset() # Reset the environment at the start of each episode
    terminated = False
    truncated = False
    current_episode_reward = 0
    current_episode_length = 0

    while not terminated and not truncated:
        # Exploration vs. Exploitation (Epsilon-greedy strategy)
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore: take a random action
        else:
            action = np.argmax(q_table[state, :]) # Exploit: take the best known action

        # Take the action and observe the new state and reward
        new_state, reward, terminated, truncated, info = env.step(action)

        # Q-learning formula to update Q-value
        # Q(s,a) = Q(s,a) + lr * [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        q_table[state, action] = q_table[state, action] + learning_rate * \
                                 (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])

        state = new_state                 # Move to the new state
        current_episode_reward += reward
        current_episode_length += 1

    # Decay epsilon after each episode (exploration rate decreases over time)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    episode_rewards.append(current_episode_reward)
    episode_lengths.append(current_episode_length)

    # Print training progress (e.g., every 1000 episodes)
    if (episode + 1) % 1000 == 0:
        print(f"Episode: {episode + 1}/{total_episodes} | Avg Reward (last 100): {np.mean(episode_rewards[-100:]):.2f} | Epsilon: {epsilon:.4f}")

env.close() # Close the environment once training is done
print("--- Training Finished ---")

# --- 5. Save the Q-table (Optional) ---
# np.save("q_table_taxi.npy", q_table)
# print("Q-table saved to q_table_taxi.npy")

# --- 6. Plotting Training Metrics ---
# Calculate moving average for rewards for smoother plot
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

window = 100 # Moving average window
smoothed_rewards = moving_average(episode_rewards, window)
smoothed_lengths = moving_average(episode_lengths, window)

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot Rewards
axs[0].plot(range(window -1, len(episode_rewards)), smoothed_rewards)
axs[0].set_title(f'Episode Rewards (Smoothed over {window} episodes)')
axs[0].set_ylabel('Average Reward')
axs[0].grid(True)

# Plot Episode Lengths
axs[1].plot(range(window-1, len(episode_lengths)), smoothed_lengths)
axs[1].set_title(f'Episode Lengths (Smoothed over {window} episodes)')
axs[1].set_ylabel('Average Length')
axs[1].set_xlabel('Episode')
axs[1].grid(True)

plt.tight_layout()
plt.show()

# --- 7. Evaluate the Trained Agent (Example) ---
print("\n--- Evaluating Trained Agent ---")
# Load the Q-table if you saved it and are running evaluation separately
# q_table = np.load("q_table_taxi.npy")

eval_env = gym.make("Taxi-v3", render_mode="human") # Use "human" to watch
# eval_env = gym.make("Taxi-v3") # No rendering

total_eval_episodes = 5
total_eval_rewards = 0
total_eval_steps = 0

for episode in range(total_eval_episodes):
    state, info = eval_env.reset()
    terminated = False
    truncated = False
    episode_reward = 0
    steps_in_episode = 0
    print(f"\n--- Starting Evaluation Episode {episode + 1} ---")

    while not terminated and not truncated:
        action = np.argmax(q_table[state, :]) # Always exploit during evaluation
        new_state, reward, terminated, truncated, info = eval_env.step(action)
        state = new_state
        episode_reward += reward
        steps_in_episode += 1
        if eval_env.render_mode == "human":
            eval_env.render() # Render the environment
            # You might want to add a small delay here if it runs too fast
            # import time
            # time.sleep(0.1)

    total_eval_rewards += episode_reward
    total_eval_steps += steps_in_episode
    print(f"Episode {episode + 1}: Reward = {episode_reward}, Steps = {steps_in_episode}")

eval_env.close()
print(f"\nAverage Reward over {total_eval_episodes} evaluation episodes: {total_eval_rewards / total_eval_episodes}")
print(f"Average Steps over {total_eval_episodes} evaluation episodes: {total_eval_steps / total_eval_episodes}")