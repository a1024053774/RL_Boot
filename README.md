# RL_Boot

本仓库是一个强化学习与机器学习的代码学习与实践仓库，涵盖了基础 Python、NumPy、PyTorch、经典机器学习内容，以及强化学习（Reinforcement Learning, RL）相关的实战案例。仓库以 Jupyter Notebook 和 Python 脚本为主，适合初学者和进阶者系统学习与实验。

## 目录结构与主要内容

- **PythonBase/**  
  Python 基础语法与数据类型练习，包括变量、数据类型、列表、集合、文件操作等示例。

- **NumPyLearn/**  
  NumPy 数值计算基础，包括数组操作、向量化计算等高效数据处理技巧。

- **Pytorch/**  
  PyTorch 框架基础，数据加载（如 CIFAR10 数据集）、TensorBoard 可视化等深度学习相关内容。

- **MachineLearning/**  
  机器学习基础专题，内容参考 Coursera 相关课程，包括监督学习（回归与分类）练习题、可选实验、作业讲解等，适合入门和巩固理论。
    - 详细内容请参考 [`MachineLearning/README.md`](MachineLearning/README.md) 和各周目录下的 Readme。

- **Hugging_Face/**  
  深度强化学习与 Hugging Face 相关实战，包括 Sample Factory、ViZDoom 等环境的训练与评测，涵盖 RL 智能体训练、模型上传、视频可视化等完整流程。
    - 典型案例：`unit8-part2-Doom/unit8_part2.ipynb`，实现了基于 Sample Factory 的 Doom 智能体训练与评测。

## 依赖环境

- Python 3.11
- Jupyter Notebook
- numpy
- gymnasium
- torch, torchvision, tensorboard
- stable-baselines3
- tqdm
- sample-factory
- vizdoom
- huggingface_hub
- ipywidgets
- ffmpeg（用于视频生成与可视化）

> 建议使用虚拟环境（如 venv 或 conda）进行依赖管理，部分 RL 相关依赖仅支持 Linux/Mac。

## 参考与鸣谢

部分学习内容参考了 Coursera、Hugging Face Deep RL Course 及相关开源项目，详见各目录下说明文档与原始链接。
