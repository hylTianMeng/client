import torch

def check_torch_gpu():
    print("=" * 50)
    print("PyTorch GPU 可用性检查")
    print("=" * 50)

    # 1. 检查 CUDA 是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 是否可用: {cuda_available}")

    if not cuda_available:
        print("\n❌ 未检测到可用的 GPU / CUDA，将使用 CPU 运行")
        print("建议：检查是否安装了 GPU 版 PyTorch、显卡驱动、CUDA")
        return

    # 2. 检测 GPU 数量
    gpu_count = torch.cuda.device_count()
    print(f"可用 GPU 数量: {gpu_count}")

    # 3. 输出每块 GPU 的详细信息
    for i in range(gpu_count):
        print(f"\n--- GPU {i} 信息 ---")
        print(f"设备名称: {torch.cuda.get_device_name(i)}")
        print(f"显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"当前 CUDA 设备: {torch.cuda.current_device()}")

    # 4. 验证张量能否在 GPU 上创建（最关键的实际测试）
    try:
        test_tensor = torch.tensor([1.0, 2.0]).cuda()
        print(f"\n✅ 张量成功创建在 GPU 上: {test_tensor.device}")
        print("🎉 GPU 可用且正常工作！")
    except Exception as e:
        print(f"\n❌ GPU 测试失败: {e}")

if __name__ == "__main__":
    check_torch_gpu()