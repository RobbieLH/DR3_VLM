import torch
print(torch.cuda.device_count())  # 打印可用的 GPU 数量
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")