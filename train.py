import torch

from data.math10k import Math10k
from utility.initialize import initialize

def train():
    # 命令行参数
    args = {}
    args["batch_size"] = 256
    args["threads"] = 2

    # 初始化
    # index_num = random.randint(1, 2000)
    index_num = 42
    initialize(index_num)

    # 检测 CUDA
    print('Cuda:', torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    args["storage_size"] = 9919
    dataset = Math10k(args.batch_size, args.threads)

    
