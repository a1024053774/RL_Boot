import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from huggingface_hub import HfApi, upload_folder
from huggingface_hub.repocard import metadata_eval_result, metadata_save

from pathlib import Path
import datetime
import tempfile
import json
import shutil
import imageio

from wasabi import Printer
msg = Printer()

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="本次实验的名称")
    parser.add_argument("--seed", type=int, default=1,
        help="实验的种子")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="如果开启, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="如果开启, 默认将启用 cuda")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="如果开启，本次实验将使用 Weights and Biases 进行跟踪")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="wandb 的项目名称")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="wandb 项目的实体（团队）")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="是否捕获智能体表现的视频（请查看 `videos` 文件夹）")

    # 算法特定参数
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="环境的 ID")
    parser.add_argument("--total-timesteps", type=int, default=50000,
        help="实验的总时间步数")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
        help="优化器的学习率")
    parser.add_argument("--num-envs", type=int, default=4,
        help="并行游戏环境的数量")
    parser.add_argument("--num-steps", type=int, default=1024,
        help="每个策略 rollout 在每个环境中运行的步数")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="为策略和价值网络切换学习率退火")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="使用 GAE 进行优势计算")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="折扣因子 gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="广义优势估计的 lambda")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="mini-batch 的数量")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="更新策略的 K 个 epoch")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="切换优势标准化")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="代理裁剪系数")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="根据论文，切换是否为价值函数使用裁剪损失。")
    parser.add_argument("--ent-coef", type=float, default=0.02,
        help="熵的系数")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="价值函数的系数")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="梯度裁剪的最大范数")
    parser.add_argument("--target-kl", type=float, default=None,
        help="目标 KL 散度阈值")

    # 添加 HuggingFace 参数
    parser.add_argument("--repo-id", type=str, default="a1024053774/ppo-CartPole-v1", help="Hugging Face Hub 上的模型仓库 ID {username/repo_name}")
    parser.add_argument("--load-model-from", type=str, default="", help="要加载的预训练模型的路径。如果为 None，则不加载任何模型。")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

def package_to_hub(repo_id,
                model,
                hyperparameters,
                eval_env,
                video_fps=30,
                commit_message="Push agent to the Hub",
                token= None,
                logs=None
                ):
  """
  评估、生成视频并将模型上传到 Hugging Face Hub。
  该方法完成了整个流程：
  - 评估模型
  - 生成模型卡片
  - 生成智能体的回放视频
  - 将所有内容推送到 Hub
  :param repo_id: Hugging Face Hub 上的模型仓库 ID
  :param model: 训练好的模型
  :param eval_env: 用于评估智能体的环境
  :param fps: 渲染视频的帧率
  :param commit_message: 提交信息
  :param logs: 你想上传的本地 TensorBoard 日志目录
  """
  msg.info(
        "该函数将保存、评估、生成你的智能体视频，"
        "创建模型卡片并将所有内容推送到 Hub。"
        "这可能需要长达 1 分钟。\n "
        "这项工作仍在进行中：如果遇到错误，请提出 issue。"
    )
  # 步骤 1：克隆或创建仓库
  repo_url = HfApi().create_repo(
        repo_id=repo_id,
        token=token,
        private=False,
        exist_ok=True,
    )

  with tempfile.TemporaryDirectory() as tmpdirname:
    tmpdirname = Path(tmpdirname)

    # 步骤 2：保存模型
    torch.save(model.state_dict(), tmpdirname / "model.pt")

    # 步骤 3：评估模型并构建 JSON 文件
    mean_reward, std_reward = _evaluate_agent(eval_env,
                                           10,
                                           model)

    # 首先获取日期时间
    eval_datetime = datetime.datetime.now()
    eval_form_datetime = eval_datetime.isoformat()

    evaluate_data = {
        "env_id": hyperparameters.env_id,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "n_evaluation_episodes": 10,
        "eval_datetime": eval_form_datetime,
    }

    # 写入 JSON 文件
    with open(tmpdirname / "results.json", "w") as outfile:
      json.dump(evaluate_data, outfile)

    # 步骤 4：生成视频
    video_path =  tmpdirname / "replay.mp4"
    record_video(eval_env, model, video_path, video_fps)

    # 步骤 5：生成模型卡片
    generated_model_card, metadata = _generate_model_card("PPO", hyperparameters.env_id, mean_reward, std_reward, hyperparameters)
    _save_model_card(tmpdirname, generated_model_card, metadata)

    # 步骤 6：如果需要，添加日志
    if logs:
      _add_logdir(tmpdirname, Path(logs))

    msg.info(f"正在将仓库 {repo_id} 推送到 Hugging Face Hub")

    repo_url = upload_folder(
            repo_id=repo_id,
            folder_path=tmpdirname,
            path_in_repo="",
            commit_message=commit_message,
            token=token,
        )

    msg.info(f"你的模型已推送到 Hub。你可以在这里查看你的模型：{repo_url}")
  return repo_url

def _evaluate_agent(env, n_eval_episodes, policy):
  """
  评估智能体 n_eval_episodes 个回合，并返回平均奖励和奖励的标准差。
  :param env: 评估环境
  :param n_eval_episodes: 评估智能体的回合数
  :param policy: 智能体
  """
  episode_rewards = []
  for episode in range(n_eval_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards_ep = 0

    while done is False:
      state = torch.Tensor(state).to(device)
      action, _, _, _ = policy.get_action_and_value(state)
      new_state, reward, done, info = env.step(action.cpu().numpy())
      total_rewards_ep += reward
      if done:
        break
      state = new_state
    episode_rewards.append(total_rewards_ep)
  mean_reward = np.mean(episode_rewards)
  std_reward = np.std(episode_rewards)

  return mean_reward, std_reward


def record_video(env, policy, out_directory, fps=30):
  images = []
  done = False
  state = env.reset()
  img = env.render(mode='rgb_array')
  images.append(img)
  while not done:
    state = torch.Tensor(state).to(device)
    # 在给定状态下，采取具有最大预期未来奖励的动作（索引）
    action, _, _, _  = policy.get_action_and_value(state)
    state, reward, done, info = env.step(action.cpu().numpy()) # 为了录制逻辑，我们直接将 next_state 设置为 state
    img = env.render(mode='rgb_array')
    images.append(img)
  imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)


def _generate_model_card(model_name, env_id, mean_reward, std_reward, hyperparameters):
  """
  为 Hub 生成模型卡片
  :param model_name: 模型名称
  :env_id: 环境名称
  :mean_reward: 智能体的平均奖励
  :std_reward: 智能体平均奖励的标准差
  :hyperparameters: 训练参数
  """
  # 步骤 1：选择标签
  metadata = generate_metadata(model_name, env_id, mean_reward, std_reward)

  # 将超参数命名空间转换为字符串
  converted_dict = vars(hyperparameters)
  converted_str = str(converted_dict)
  converted_str = converted_str.split(", ")
  converted_str = '\n'.join(converted_str)

  # 步骤 2：生成模型卡片
  model_card = f"""
  # PPO Agent Playing {env_id}

  This is a trained model of a PPO agent playing {env_id}.

  # Hyperparameters
  ```python
  {converted_str}
  ```
  """
  return model_card, metadata

def generate_metadata(model_name, env_id, mean_reward, std_reward):
  """
  为模型卡片定义标签
  :param model_name: 模型名称
  :param env_id: 环境名称
  :mean_reward: 智能体的平均奖励
  :std_reward: 智能体平均奖励的标准差
  """
  metadata = {}
  metadata["tags"] = [
        env_id,
        "ppo",
        "deep-reinforcement-learning",
        "reinforcement-learning",
        "custom-implementation",
        "deep-rl-course"
  ]

  # 添加指标
  eval = metadata_eval_result(
      model_pretty_name=model_name,
      task_pretty_name="reinforcement-learning",
      task_id="reinforcement-learning",
      metrics_pretty_name="mean_reward",
      metrics_id="mean_reward",
      metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
      dataset_pretty_name=env_id,
      dataset_id=env_id,
  )

  # 合并两个字典
  metadata = {**metadata, **eval}

  return metadata

def _save_model_card(local_path, generated_model_card, metadata):
    """为仓库保存模型卡片。
    :param local_path: 仓库目录
    :param generated_model_card: 由 _generate_model_card() 生成的模型卡片
    :param metadata: 元数据
    """
    readme_path = local_path / "README.md"
    readme = ""
    if readme_path.exists():
        with readme_path.open("r", encoding="utf8") as f:
            readme = f.read()
    else:
        readme = generated_model_card

    with readme_path.open("w", encoding="utf-8") as f:
        f.write(readme)

    # 将我们的指标保存到 Readme 元数据中
    metadata_save(readme_path, metadata)

def _add_logdir(local_path: Path, logdir: Path):
  """向仓库中添加日志目录。
  :param local_path: 仓库目录
  :param logdir: 日志目录
  """
  if logdir.exists() and logdir.is_dir():
    # 在名为 logs 的新目录下将 logdir 添加到仓库中
    repo_logdir = local_path / "logs"

    # 如果当前日志存在，则删除
    if repo_logdir.exists():
      shutil.rmtree(repo_logdir)

    # 将 logdir 复制到仓库的日志目录中
    shutil.copytree(logdir, repo_logdir)

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # 尽量不要修改：播种
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # 环境设置
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "仅支持离散动作空间"

    agent = Agent(envs).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # 算法逻辑：存储设置
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # 尽量不要修改：开始游戏
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # 如果指定，则进行学习率退火。
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # 算法逻辑：动作逻辑
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # 尽量不要修改：执行游戏并记录数据。
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break

        # 如果游戏未结束，则进行 bootstrap value 计算
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # 展平 batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # 优化策略和价值网络
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # 策略损失
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # 价值损失
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # 尽量不要修改：记录奖励以用于绘图
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

    # 创建评估环境
    eval_env = gym.make(args.env_id)

    package_to_hub(repo_id = args.repo_id,
                model = agent, # 我们想要保存的模型
                hyperparameters = args,
                eval_env = gym.make(args.env_id),
                logs= f"runs/{run_name}",
                )
