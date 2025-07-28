import gymnasium as gym
from stable_baselines3 import PPO

# 1. 创建环境
# "CartPole-v1" 是一个经典控制任务：控制一个小车左右移动，让它头上的杆子保持竖直不倒。
# render_mode="human" 会创建一个窗口来实时显示游戏画面。
env = gym.make("CartPole-v1", render_mode="human")

# 2. 定义模型
# 我们使用PPO算法，这是目前最流行、效果最好的算法之一。
# "MlpPolicy" 表示我们使用一个标准的多层感知机（神经网络）作为策略网络。
# verbose=1 会在训练时打印出详细的日志，如奖励、损失等。
model = PPO("MlpPolicy", env, verbose=1)

# 3. 训练模型
# 让模型在环境中进行10000个时间步（timesteps）的“学习”。
model.learn(total_timesteps=10000)
print("\n训练完成！\n")

# 4. 评估和展示训练好的模型
obs, info = env.reset()  # 重置环境，获取初始观测（obs）和其他信息（info）
for i in range(1000):
    # a. 模型根据当前观测（obs）来预测下一步的动作（action）
    action, _states = model.predict(obs, deterministic=True)

    # b. 环境执行动作，并返回新的状态
    obs, reward, terminated, truncated, info = env.step(action)

    # c. 如果游戏结束（杆子倒了或小车出界），则重置环境
    if terminated or truncated:
        print(f"回合在第 {i + 1} 步结束。重置环境。")
        obs, info = env.reset()

# 5. 关闭环境
env.close()