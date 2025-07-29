import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time


# Actor-Critic 网络
class ActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(ActorCritic, self).__init__()
        # 定义共享的神经网络层
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)

        # Actor-specific layer (策略输出)
        self.actor = nn.Linear(128, n_actions)

        # Critic-specific layer (价值输出)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        """前向传播，返回动作概率和状态价值"""
        # 通过共享层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Actor 输出: 动作的概率分布
        actor_output = F.softmax(self.actor(x), dim=-1)

        # Critic 输出: 状态的价值
        critic_output = self.critic(x)

        return actor_output, critic_output

def setup_a3c_and_run():
    # 注意：多进程代码通常需要放在 `if __name__ == '__main__':` 块中以避免问题

    # 创建环境以获取参数
    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n
    input_shape = env.observation_space.shape[0]
    env.close()

    # 1. 创建全局模型
    global_model = ActorCritic(input_shape, n_actions)
    global_model.share_memory()

    # (新增) 使用 Manager 创建一个所有进程共享的列表
    with mp.Manager() as manager:
        rewards_list = manager.list()  # 用于存储所有 worker 的奖励

        # 2. 创建全局优化器
        optimizer = optim.Adam(global_model.parameters(), lr=0.0005)

        # 3. 创建并启动工作进程
        processes = []
        num_processes = mp.cpu_count()
        print(f"Starting {num_processes} worker processes...")

        for i in range(num_processes):
            # (修改) 将共享列表传递给 worker
            p = mp.Process(target=worker, args=(global_model, optimizer, input_shape, n_actions, i, rewards_list))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print("--- A3C Training Finished ---")

        # (修改) 训练结束后，调用评估函数
        # 此时的 global_model 包含了所有 worker 的训练成果
        evaluate(global_model)

        # (新增) 训练结束后，绘制奖励曲线
        plt.figure(figsize=(10, 5))
        plt.plot(list(rewards_list))
        plt.title('A3C Training Rewards')
        plt.xlabel('Episodes (across all workers)')
        plt.ylabel('Total Reward')
        # (可选) 计算并绘制奖励的移动平均线，让趋势更清晰
        rewards_np = np.array(list(rewards_list))
        moving_avg = np.convolve(rewards_np, np.ones(100) / 100, mode='valid')
        plt.plot(moving_avg, linewidth=3, label='Moving Average (100 episodes)')
        plt.legend()
        plt.grid(True)
        plt.show()


def worker(global_model, optimizer, input_shape, n_actions, worker_id, rewards_list):
    """每个 Worker 进程执行的函数"""
    # 1. 创建一个新的环境实例和本地模型
    env = gym.make('CartPole-v1')
    local_model = ActorCritic(input_shape, n_actions)

    # 对每一个 Episode 进行迭代
    for i_episode in range(200):  # 每个worker跑200个episode
        # 从全局模型同步最新的参数到本地模型
        local_model.load_state_dict(global_model.state_dict())

        state, _ = env.reset()
        done = False
        total_reward = 0

        # 在一个 episode 内部进行循环
        while not done:
            # 使用本地模型进行采样
            state_tensor = torch.FloatTensor(state)
            probs, _ = local_model(state_tensor)
            action = probs.multinomial(num_samples=1).item()

            next_state, reward, done, _, info = env.step(action)
            total_reward += reward

            # 使用收集到的数据更新全局模型
            update_a3c(global_model, optimizer, local_model, state, action, reward, next_state, done)

            state = next_state

        # 在 episode 结束后，将总奖励添加到共享列表中
        rewards_list.append(total_reward)

        print(f"Worker {worker_id}, Episode: {i_episode}, Total Reward: {total_reward}")

    env.close()

# A3C 更新函数
def update_a3c(global_model, optimizer, local_model, state, action, reward, next_state, done, gamma=0.99):
    # 将输入数据转换为 PyTorch Tensors
    state = torch.FloatTensor(state).unsqueeze(0)
    next_state = torch.FloatTensor(next_state).unsqueeze(0)
    action = torch.LongTensor([action])
    reward = torch.FloatTensor([reward])
    done = torch.FloatTensor([int(done)])

    # 计算 Advantage (优势)
    # 注意：价值的计算是基于全局模型的预测，因为它被认为是更准确的
    _, next_state_value = global_model(next_state)
    _, state_value = global_model(state)
    advantage = reward + gamma * next_state_value * (1 - done) - state_value

    # 计算 Actor 和 Critic 的 Loss
    # 使用本地模型计算动作概率，因为这是产生该动作的模型
    probs, _ = local_model(state)
    log_prob = torch.log(probs.squeeze(0)[action])
    actor_loss = -(log_prob * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()
    loss = actor_loss + critic_loss

    # 关键：计算梯度并更新全局模型的参数
    optimizer.zero_grad()
    loss.backward()
    # 将本地模型的梯度复制到全局模型
    for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
        if global_param.grad is not None:
            break
        global_param._grad = local_param.grad
    optimizer.step()

    # 更新后，将全局模型的参数复制回本地模型
    local_model.load_state_dict(global_model.state_dict())

def evaluate(model, n_episodes=5):
    """用训练好的模型进行评估并渲染游戏画面"""
    print("\n--- Starting Evaluation ---")
    # (重要) 创建一个可以渲染画面的新环境
    env = gym.make('CartPole-v1', render_mode='human')
    input_shape = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # 加载模型的最终状态
    final_model = ActorCritic(input_shape, n_actions)
    final_model.load_state_dict(model.state_dict())
    final_model.eval() # 设置为评估模式

    for i in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            state_tensor = torch.FloatTensor(state)
            with torch.no_grad(): # 评估时不需要计算梯度
                probs, _ = final_model(state_tensor)
                # 选择概率最高的动作，而不是随机采样
                action = torch.argmax(probs).item()

            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            total_reward += reward
            time.sleep(0.02) # 短暂暂停，方便肉眼观察

        print(f"Evaluation Episode {i+1}, Total Reward: {total_reward}")

    env.close()
    print("--- Evaluation Finished ---")

if __name__ == '__main__':
    # 设置多进程启动方法，'spawn' 在多平台下更稳定
    try:
        mp.set_start_method('spawn')
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        pass

    # 你可以在这里选择运行哪个模型的训练
    # print("--- Training AC Model ---")
    # train_ac()

    print("\n--- Training A3C Model ---")
    setup_a3c_and_run()