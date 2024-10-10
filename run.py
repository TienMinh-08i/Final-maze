import numpy as np
import random
import matplotlib.pyplot as plt
from env import MazeEnv
from generate_maze import Maze
import pygame
import pickle

model_Q = r"C:\Users\admin\Desktop\mazee\policy\q_table_maze_8x8.pkl"
model_Monte = r"C:\Users\admin\Desktop\mazee\q_table_maze_16x16.pkl"
model_Sarsa = r"C:\Users\admin\Desktop\mazee\policy\q_table_sarsa_maze_32x32.pkl"
model_VI = r"C:\Users\admin\Desktop\src\policy\VI_policy_maze_4x4.pkl"
model_PI = r"C:\Users\admin\Desktop\src\policy\PI_policy_maze_4x4.pkl"

maze = r"C:\Users\admin\Desktop\src\maze\maze_4x4.pkl"

def run_agent_with_q_table(env, q_table):
    """Chạy agent trong môi trường bằng cách sử dụng Q-table."""
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        env.render()

        # Chọn hành động có giá trị Q cao nhất
        possible_actions = env.get_possible_actions()
        q_values = [q_table[state, a] for a in possible_actions]
        action = possible_actions[np.argmax(q_values)]

        # Thực hiện hành động và cập nhật
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
        steps += 1

        print(f"Agent moved to {next_state}, Reward: {reward}, Total steps: {steps}")
        pygame.time.delay(500)

    print(f"Episode finished. Total reward: {total_reward}, Total steps: {steps}")
    pygame.time.delay(5000)
def run_agent_with_optimal_policy(env, optimal_policy):
    """Chạy agent trong môi trường bằng cách sử dụng optimal policy."""
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        env.render()

        # Chọn hành động dựa trên optimal policy
        action = optimal_policy.get(state)
        if action is None:
            print(f"Không tìm thấy hành động cho trạng thái {state}. Chọn hành động ngẫu nhiên.")
            possible_actions = env.get_possible_actions()
            if not possible_actions:
                print("Không có hành động nào có thể thực hiện được.")
                break
            action = random.choice(possible_actions)

        # Thực hiện hành động và cập nhật
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
        steps += 1

        print(f"Agent moved to {next_state}, Reward: {reward}, Total steps: {steps}")
        pygame.time.delay(500)

    print(f"Episode finished. Total reward: {total_reward}, Total steps: {steps}")
    pygame.time.delay(5000)

model = model_VI
# Lấy q_table từ file
with open(model, 'rb') as f:
    loaded_q_table = pickle.load(f)


myMaze = Maze().load(maze)
env = MazeEnv(myMaze)

# Chạy agent với Q-table đã học
run_agent_with_q_table(env, loaded_q_table)
# Chạy agent với optimal policy
#run_agent_with_optimal_policy(env, loaded_q_table)
