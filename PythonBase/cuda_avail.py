import torch
import time
import sys


# -------------------------------------------------------------------
#  主验证函数
# -------------------------------------------------------------------
def verify_cuda():
    """
    检查PyTorch的CUDA环境是否可用，并执行一个简单的GPU计算任务。
    """
    print("=" * 50)
    print("开始检测PyTorch CUDA环境...")
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    print("=" * 50)

    # 核心检查：torch.cuda.is_available()
    if torch.cuda.is_available():

        # 打印成功的消息和设备信息
        print("\n🎉🎉🎉 恭喜！检测到可用的CUDA设备！🎉🎉🎉")

        device_count = torch.cuda.device_count()
        print(f"检测到 {device_count} 个可用的GPU。")

        # 获取当前GPU设备的详细信息
        current_device_id = torch.cuda.current_device()
        current_device_name = torch.cuda.get_device_name(current_device_id)
        print(f"当前使用的设备ID: {current_device_id}, 设备名称: {current_device_name}\n")

        # --- 执行一个简单的GPU计算任务来证明其可用性 ---
        print("下一步，我们将尝试在GPU上执行一个计算任务...")

        # 定义一个在GPU上执行的设备
        device = torch.device("cuda")

        # 创建两个大规模的随机张量（tensor）并移动到GPU上
        # 使用 try-except 来捕捉可能的显存不足错误
        try:
            print("正在创建大规模张量并移动到GPU...")
            size = (15000, 15000)
            tensor_a = torch.randn(size, device=device)
            tensor_b = torch.randn(size, device=device)
            print("张量已成功创建在GPU上。")

            # 执行矩阵乘法，这是一个典型的GPU密集型任务
            print("\n正在执行大规模矩阵乘法 (15000x15000)...")
            start_time = time.time()
            result = torch.matmul(tensor_a, tensor_b)
            end_time = time.time()

            # 打印计算结果
            print(f"计算完成！耗时: {end_time - start_time:.4f} 秒。")
            print("如果程序没有报错并顺利打印出耗时，说明您的CUDA不仅可用，而且可以正常执行计算！")

        except torch.cuda.OutOfMemoryError:
            print("\n❌ 计算失败：GPU显存不足！")
            print("这是一个正常的错误，如果你的GPU显存较小。但这同样证明了PyTorch正在尝试使用GPU。")

        except Exception as e:
            print(f"\n❌ 在计算过程中发生未知错误: {e}")

    else:
        # 打印失败的消息和可能的解决方案
        print("\n❌ 检测失败：PyTorch无法找到可用的CUDA设备。")
        print("请检查以下几点：")
        print("  1. 确认您的电脑拥有NVIDIA显卡，并且最新的驱动程序已正确安装。")
        print("  2. 确认您安装的是PyTorch的GPU版本（版本号中不应包含 '+cpu'）。")
        print("  3. 可以尝试在您的终端（Anaconda Prompt或cmd）中运行 `nvidia-smi` 命令，检查是否能看到GPU信息。")

    print("\n" + "=" * 50)
    print("检测结束。")
    print("=" * 50)


# -------------------------------------------------------------------
#  程序入口
# -------------------------------------------------------------------
if __name__ == "__main__":
    verify_cuda()