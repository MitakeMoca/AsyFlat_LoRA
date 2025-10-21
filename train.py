import math
import os

from data.gsm8k import GSM8k
from utility.scheduler import CosineScheduler, ProportionScheduler, CosineRhoScheduler
from utility.loss import extract_final_answer, smooth_crossentropy
from utility.evaluate import evaluate_model
from trainer.basic_trainer import BaseTrainer
from trainer.asyflat_trainer import AsyFlatTrainer
from trainer.flat_trainer import FlatLoRATrainer
from trainer.eflat_trainer import EFlatLoRATrainer
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from peft import LoraConfig, get_peft_model
import torch
from torch.optim import SGD
import yaml
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from data.math10k import Math10k
from utility.initialize import initialize
import torch.nn.functional as F

tokenizer = None

import os

os.makedirs('logs', exist_ok=True)

def train():
    global tokenizer

    with open("./conf/model/llama.yaml", 'r') as f:
        config = yaml.safe_load(f)

    print(config)

    with open('logs/training_eflat_time.log', 'a') as f:  # 'a' 模式追加，'w' 模式覆盖
        f.write(f"hello\n")

    # 命令行参数
    args = {}
    args["batch_size"] = 2
    args["threads"] = 0
    args["model_type"] = "llama"
    args["model_name"] = "meta-llama/Meta-Llama-3-8B"
    args["learning_rate"] = 2.0e-4
    args["epochs"] = 3
    args["rho"] = 0.1
    args["rho_max"] = 0.05
    args["rho_min"] = 0.02
    args["adaptive"] = False
    args["alpha"] = 0.5
    args["beta"] = 0.5

    args["fmin"] = 0.1
    args["fmax"] = 0.5

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
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    # 设置 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        # quantization_config=bnb_config,
        device_map="auto"
    )

    peft_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules=config['lora_target_modules'],
        bias="none",
        task_type="CAUSAL_LM"
    )
    print(peft_config)
    model = get_peft_model(model, peft_config)
    model = model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.print_trainable_parameters()

    for name, m in model.named_modules():
        if hasattr(m, "lora_A"):
            print(f"{name}: {type(m.lora_A)}")
            if isinstance(m.lora_A, torch.nn.ModuleDict):
                print(" keys:", list(m.lora_A.keys()))
            break


    # 配置不同的优化器
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    base_optimizer = SGD(trainable_params, lr=args["learning_rate"], momentum=0.9)

    scheduler = CosineScheduler(T_max=args["epochs"] * len(dataset.train), max_value=args["learning_rate"],
                                min_value=0.0, optimizer=base_optimizer)

    rho_scheduler = CosineRhoScheduler(max_value=args["rho_max"], min_value=args["rho_min"], total_steps=args["epochs"] * math.ceil(len(dataset.train) / args["batch_size"]))

    whole_time = 0

    text = "Hello world!"
    ids = tokenizer.encode(text)
    print(tokenizer.batch_decode(ids, skip_special_tokens=True))
    print(tokenizer.decode(ids))

    trainer = BaseTrainer(model, tokenizer, base_optimizer, device)
    # trainer = FlatLoRATrainer(model, tokenizer, base_optimizer, device, rho=0.05, rho_schedule=rho_scheduler)
    # trainer = EFlatLoRATrainer(model, tokenizer, base_optimizer, device, rho=0.05, rho_schedule=rho_scheduler, beta=0.9)
    # trainer = AsyFlatTrainer(model, tokenizer, base_optimizer, device, args["storage_size"], args["fmin"], args["fmax"], rho=args["rho"], rho_schedule=rho_scheduler, alpha=args["alpha"])

    for epoch in range(args["epochs"]):
        model.train()
        tt = 0

        for batch in dataset.train:
            start_time = time.time()
            tt += 1
            trainer.train_step(batch)

            end_time = time.time()
            es_time = end_time - start_time
            whole_time += es_time

            if tt % 10 == 0:
                print(whole_time)


        avg_loss, acc = evaluate_model(model, tokenizer, device, args["batch_size"], args["threads"])
        print("final: ", avg_loss, acc)
    print(whole_time)
    # 将时间写入文件
    with open('logs/training_eflat_time.log', 'a') as f:  # 'a' 模式追加，'w' 模式覆盖
        f.write(f"Training time: {trainer.time}\n")

if __name__ == "__main__":
    train()