from peft import LoraConfig, get_peft_model
import torch
from torch.optim import SGD
import yaml
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from data.math10k import Math10k
from models.llama import MODEL_TYPES
from optimizer.asyflat import AsyFlat_LoRA
from utility.initialize import initialize

tokenizer = None 

def train():
    global tokenizer

    with open("./conf/model/llama.yaml", 'r') as f:
        config = yaml.safe_load(f)

    print(config)

    # 命令行参数
    args = {}
    args["batch_size"] = 256
    args["threads"] = 0
    args["model_type"] = "llama"
    args["model_name"] = "meta-llama/Meta-Llama-3-8B"
    args["learning_rate"] = 2.0e-4
    args["epochs"] = 3

    # 初始化
    # index_num = random.randint(1, 2000)
    index_num = 42
    initialize(index_num)

    # 检测 CUDA
    print('Cuda:', torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    args["storage_size"] = 9919
    dataset = Math10k(args["batch_size"], args["threads"])

    # 使用模型 llama3-8B
    model_name = config['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 设置 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        # quantization_config=bnb_config,
        device_map="cpu"
    )
    
    peft_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules=config['lora_target_modules'],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print(type(model.parameters()))

    # 配置不同的优化器
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    base_optimizer = SGD(trainable_params, lr=args["learning_rate"], momentum=0.9)
    # asyflat_optimizer = AsyFlat_LoRA(trainable_params, base_optimizer, )

    for epoch in range(args["epochs"]):
        model.train()
        start_time = time.time()

        for batch in dataset.train:
            inputs, targets, index = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            # tf 是采样之后的样本集
            tf = optimizer.sampledata_index(args, epoch, index, fmax_)


if __name__ == "__main__":
    train()