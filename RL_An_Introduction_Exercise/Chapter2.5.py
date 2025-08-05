import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class NonStationaryBandit:
    """
    一个非平稳的k臂赌博机环境。

    参数:
    k (int): 臂的数量。
    walk_mean (float): 真实奖励值随机游走的均值。
    walk_std (float): 真实奖励值随机游走的标准差。
    """

    def __init__(self, k=10, walk_mean=0.0, walk_std=0.01):
        self.k = k
        self.walk_mean = walk_mean
        self.walk_std = walk_std
        # 初始时所有臂的真实奖励值为0
        self.true_action_values = np.zeros(self.k)

    def step(self, action):
        """
        选择一个动作，获取奖励，并让环境进行一步随机游走。

        参数:
        action (int): 被选择的臂的索引。

        返回:
        float: 获得的奖励。
        """
        # 奖励就是当前的真实动作值
        reward = self.true_action_values[action]

        # 所有臂的真实值进行一次独立的随机游走
        random_walk = np.random.normal(self.walk_mean, self.walk_std, self.k)
        self.true_action_values += random_walk

        return reward

    def get_optimal_action(self):
        """返回当前最优的动作"""
        return np.argmax(self.true_action_values)


class EpsilonGreedyAgent:
    """
    一个使用ε-greedy策略和指定动作值更新方法的智能体。

    参数:
    k (int): 臂的数量。
    epsilon (float): ε-greedy策略中的探索率。
    alpha (float): 步长参数。如果alpha=0，则使用样本平均法。
    """

    def __init__(self, k=10, epsilon=0.1, alpha=0.1):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha

        # 动作计数值，用于样本平均法
        self.action_counts = np.zeros(k)
        # 动作的估计值 Q(a)
        self.q_estimates = np.zeros(k)

    def choose_action(self):
        """使用ε-greedy策略选择一个动作。"""
        if np.random.rand() < self.epsilon:
            # 探索：随机选择一个动作
            return np.random.randint(self.k)
        else:
            # 利用：选择当前估计值最高的动作
            # 如果有多个最大值，从中随机选择一个
            max_q = np.max(self.q_estimates)
            best_actions = np.where(self.q_estimates == max_q)[0]
            return np.random.choice(best_actions)

    def update(self, action, reward):
        """根据给定的动作和奖励更新动作值估计。"""
        self.action_counts[action] += 1

        if self.alpha == 0:
            # 使用样本平均法，alpha = 1/N(a)
            step_size = 1.0 / self.action_counts[action]
        else:
            # 使用常量步长法
            step_size = self.alpha

        # 更新规则
        self.q_estimates[action] += step_size * (reward - self.q_estimates[action])


def run_experiment(k, num_steps, num_runs, epsilon, alpha):
    """
    运行完整的实验。

    返回:
    avg_rewards (np.array): 每一步的平均奖励。
    optimal_action_pct (np.array): 每一步选择最优动作的百分比。
    """
    # 存储每次运行的结果
    rewards_history = np.zeros((num_runs, num_steps))
    optimal_action_history = np.zeros((num_runs, num_steps))

    # 使用tqdm显示进度条
    for i in tqdm(range(num_runs), desc=f"Running alpha={alpha}"):
        bandit = NonStationaryBandit(k)
        agent = EpsilonGreedyAgent(k, epsilon, alpha)

        for step in range(num_steps):
            optimal_action = bandit.get_optimal_action()
            action = agent.choose_action()

            if action == optimal_action:
                optimal_action_history[i, step] = 1

            reward = bandit.step(action)
            rewards_history[i, step] = reward
            agent.update(action, reward)

    # 计算所有运行的平均值
    avg_rewards = rewards_history.mean(axis=0)
    optimal_action_pct = optimal_action_history.mean(axis=0) * 100

    return avg_rewards, optimal_action_pct


if __name__ == '__main__':
    # --- 实验参数 ---
    K = 10
    NUM_STEPS = 10000
    NUM_RUNS = 2000
    EPSILON = 0.1

    # --- 运行两种方法的实验 ---

    # 1. 样本平均法 (alpha = 0 是我们用来标识这种方法的标志)
    sample_avg_rewards, sample_avg_optimal_pct = run_experiment(
        k=K, num_steps=NUM_STEPS, num_runs=NUM_RUNS, epsilon=EPSILON, alpha=0
    )

    # 2. 常量步长法 (alpha = 0.1)
    constant_alpha_rewards, constant_alpha_optimal_pct = run_experiment(
        k=K, num_steps=NUM_STEPS, num_runs=NUM_RUNS, epsilon=EPSILON, alpha=0.1
    )

    # --- 绘图 ---
    plt.figure(figsize=(12, 10))

    # 图1：平均奖励
    plt.subplot(2, 1, 1)
    plt.plot(sample_avg_rewards, label=r'Sample Average ($ \alpha = 1/N(a) $)')
    plt.plot(constant_alpha_rewards, label=r'Constant Step-Size ($ \alpha = 0.1 $)')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Average Reward in a Non-Stationary Environment')
    plt.legend()
    plt.grid(True)

    # 图2：最优动作选择百分比
    plt.subplot(2, 1, 2)
    plt.plot(sample_avg_optimal_pct, label=r'Sample Average ($ \alpha = 1/N(a) $)')
    plt.plot(constant_alpha_optimal_pct, label=r'Constant Step-Size ($ \alpha = 0.1 $)')
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action')
    plt.title('Percentage of Optimal Actions Taken')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()