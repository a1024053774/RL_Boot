import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# 环境类与练习2.5相同
class NonStationaryBandit:
    """一个非平稳的k臂赌博机环境。"""

    def __init__(self, k=10, walk_mean=0.0, walk_std=0.01):
        self.k = k
        self.walk_mean = walk_mean
        self.walk_std = walk_std
        self.true_action_values = np.zeros(self.k)

    def step(self, action):
        """
        选择一个动作，获取奖励，并让环境随机游走。
        """
        # 所有臂的真实值进行一次独立的随机游走
        random_walk = np.random.normal(self.walk_mean, self.walk_std, self.k)
        self.true_action_values += random_walk # 更新所有臂的真实值
        # 奖励是随机游走之后的新值
        reward = self.true_action_values[action]
        return reward


# --- 智能体类 ---

class EpsilonGreedyAgent:
    """ε-Greedy 智能体，支持乐观初始化。"""

    def __init__(self, k=10, epsilon=0.1, alpha=0.1, initial_q=0.0):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha
        self.q_estimates = np.full(k, initial_q, dtype=np.float64)

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        else:
            max_q = np.max(self.q_estimates)
            best_actions = np.where(self.q_estimates == max_q)[0]
            return np.random.choice(best_actions)

    def update(self, action, reward): # 更新Q值
        self.q_estimates[action] += self.alpha * (reward - self.q_estimates[action])


class UCBAgent:
    """UCB 智能体。"""

    def __init__(self, k=10, c=2.0, alpha=0.1):
        self.k = k
        self.c = c
        self.alpha = alpha
        self.q_estimates = np.zeros(k, dtype=np.float64)
        self.action_counts = np.zeros(k, dtype=int)
        self.time_step = 0

    def choose_action(self):
        self.time_step += 1
        # 优先选择没有被选过的臂
        untried_actions = np.where(self.action_counts == 0)[0]
        if len(untried_actions) > 0:
            return untried_actions[0]

        # UCB计算
        ucb_values = self.q_estimates + self.c * np.sqrt(np.log(self.time_step) / self.action_counts)
        max_ucb = np.max(ucb_values)
        best_actions = np.where(ucb_values == max_ucb)[0]
        return np.random.choice(best_actions)

    def update(self, action, reward):
        self.action_counts[action] += 1
        # 在非平稳环境下，UCB也使用常量步长效果更好
        self.q_estimates[action] += self.alpha * (reward - self.q_estimates[action])


class GradientBanditAgent:
    """梯度赌博机智能体。"""

    def __init__(self, k=10, alpha=0.1):
        self.k = k
        self.alpha = alpha
        self.preferences = np.zeros(k, dtype=np.float64)
        self.avg_reward = 0
        self.time_step = 0

    def choose_action(self):
        # Softmax分布
        exp_prefs = np.exp(self.preferences - np.max(self.preferences))  # 数值稳定
        self.action_probs = exp_prefs / np.sum(exp_prefs)
        return np.random.choice(self.k, p=self.action_probs)

    def update(self, action, reward):
        self.time_step += 1
        # 更新平均奖励（作为基线）
        self.avg_reward += (reward - self.avg_reward) / self.time_step

        baseline = self.avg_reward

        # 更新偏好
        one_hot = np.zeros(self.k)
        one_hot[action] = 1
        self.preferences += self.alpha * (reward - baseline) * (one_hot - self.action_probs)


# --- 实验运行框架 ---

def run_experiment(agent_class, k, num_steps, num_runs, agent_params):
    """为单个算法和参数设置运行实验。"""
    all_rewards = np.zeros(num_steps)

    for _ in range(num_runs):
        bandit = NonStationaryBandit(k) # 创建非平稳赌博机环境
        agent = agent_class(k=k, **agent_params) # 创建智能体实例
        run_rewards = [] # 存储每次运行的奖励
        for _ in range(num_steps):
            action = agent.choose_action()
            reward = bandit.step(action)
            agent.update(action, reward)
            run_rewards.append(reward)
        all_rewards += np.array(run_rewards)

    # 计算最后100,000步的平均奖励
    avg_reward_metric = (all_rewards / num_runs)[-100000:].mean()
    return avg_reward_metric


if __name__ == '__main__':
    # --- 实验参数 ---
    K = 10
    NUM_STEPS = 200000
    NUM_RUNS = 200  # 注意：减少运行次数以加快速度，可能会增加结果的噪声

    # 参数范围
    param_values = [1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4]

    # --- 运行所有实验 ---
    results = {
        "ε-greedy": [],
        "Gradient Bandit": [],
        "UCB": [],
        "Optimistic Greedy": []
    }

    # 1. ε-greedy
    print("Running Epsilon-Greedy...")
    for eps in tqdm(param_values):
        params = {'epsilon': eps, 'alpha': 0.1}
        results["ε-greedy"].append(run_experiment(EpsilonGreedyAgent, K, NUM_STEPS, NUM_RUNS, params))

    # 2. Gradient Bandit
    print("\nRunning Gradient Bandit...")
    for alpha in tqdm(param_values):
        params = {'alpha': alpha}
        results["Gradient Bandit"].append(run_experiment(GradientBanditAgent, K, NUM_STEPS, NUM_RUNS, params))

    # 3. UCB
    print("\nRunning UCB...")
    for c in tqdm(param_values):
        params = {'c': c, 'alpha': 0.1}
        results["UCB"].append(run_experiment(UCBAgent, K, NUM_STEPS, NUM_RUNS, params))

    # 4. Optimistic Greedy
    print("\nRunning Optimistic Greedy...")
    for q0 in tqdm(param_values):
        params = {'epsilon': 0, 'alpha': 0.1, 'initial_q': q0}
        results["Optimistic Greedy"].append(run_experiment(EpsilonGreedyAgent, K, NUM_STEPS, NUM_RUNS, params))

    # --- 绘图 ---
    plt.figure(figsize=(12, 8))

    plt.plot(param_values, results["ε-greedy"], 'r-o', label='ε-greedy (α=0.1)')
    plt.plot(param_values, results["Gradient Bandit"], 'g-o', label='Gradient Bandit')
    plt.plot(param_values, results["UCB"], 'b-o', label='UCB (α=0.1)')
    plt.plot(param_values, results["Optimistic Greedy"], 'k-o', label='Greedy with Optimistic Initialization (α=0.1)')

    plt.xlabel("Parameter (ε, α, c, Q₀)")
    plt.ylabel(f"Average reward over last {NUM_STEPS - 100000} steps")
    plt.title("Parameter study on a non-stationary 10-armed bandit")
    plt.xscale('log', base=2)
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()