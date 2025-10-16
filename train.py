import os

from data.gsm8k import GSM8k
from utility.scheduler import CosineScheduler, ProportionScheduler
from utility.bypass_bn import disable_running_stats, enable_running_stats
from utility.loss import extract_final_answer, smooth_crossentropy
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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
import torch.nn.functional as F

tokenizer = None 

def train():
    global tokenizer

    with open("./conf/model/llama.yaml", 'r') as f:
        config = yaml.safe_load(f)

    print(config)

    # 命令行参数
    args = {}
    args["batch_size"] = 4
    args["threads"] = 0
    args["model_type"] = "llama"
    args["model_name"] = "meta-llama/Meta-Llama-3-8B"
    args["learning_rate"] = 2.0e-4
    args["epochs"] = 3
    args["rho"] = 0.1
    args["rho_max"] = 0.1
    args["rho_min"] = 0.1
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
    model = model.to(device)
    model.print_trainable_parameters()

    # 配置不同的优化器
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    base_optimizer = SGD(trainable_params, lr=args["learning_rate"], momentum=0.9)

    scheduler = CosineScheduler(T_max=args["epochs"] * len(dataset.train), max_value=args["learning_rate"],
                                min_value=0.0, optimizer=base_optimizer)

    rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=args["learning_rate"], min_lr=0.0,
                                        max_value=args["rho_max"], min_value=args["rho_min"])

    asyflat_optimizer = AsyFlat_LoRA(trainable_params, base_optimizer, rho=args["rho"], rho_scheduler=rho_scheduler, adaptive=args["adaptive"],
                         storage_size=args["storage_size"], alpha=args["alpha"], beta=args["beta"])

    whole_time = 0

    for epoch in range(args["epochs"]):
        model.train()
        start_time = time.time()

        # 规约化后的最大值，在不断上升
        fmax_ = args["fmax"] - ((args["epochs"] - epoch) / args["epochs"]) * (args["fmax"] - args["fmin"]) + 0.000000001

        for batch in dataset.train:
            tt += 1
            inputs, targets, index = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            # base_optimizer.zero_grad()
            # outputs = model(inputs).logits
            # loss = F.cross_entropy(
            #     outputs.view(-1, outputs.size(-1)),
            #     targets.view(-1),
            #     ignore_index=tokenizer.pad_token_id,
            #     label_smoothing=0.1
            # )
            # loss.backward()
            # base_optimizer.step()

            # tf 是采样之后的样本集
            # tf = asyflat_optimizer.sample_index(args, epoch, index, fmax_)

            # enable_running_stats(model)
            # loss_bef = smooth_crossentropy(model(inputs[tf]).logits, targets[tf])
            # loss_bef.mean().backward()
            # asyflat_optimizer.first_step(zero_grad=True)

            # disable_running_stats(model)
            # loss_aft = smooth_crossentropy(model(inputs[tf]).logits, targets[tf])
            # loss_aft.mean().backward()
            # asyflat_optimizer.second_step_without_norm(zero_grad=True)

            # 更新权重梯度的估计
            # roc = torch.abs(loss_aft - loss_bef)
            # asyflat_optimizer.impt_roc(epoch, index, tf, roc)

            # 更新 rho
            # with torch.no_grad():
            #     scheduler.step()

        end_time = time.time()
        es_time = end_time - start_time
        whole_time += es_time
        print(whole_time)

        test_dataset = GSM8k(args["batch_size"], args["threads"])
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        num_loss = 0

        with torch.no_grad():
            for sample in test_dataset.test:
                prompt, label, _ = batch[0].to(device), batch[1].to(device), batch[2]

                enc = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(device)
                ans = tokenizer(
                    label,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                ).to(device)

                # 拼接输入: [prompt + answer[:-1]] -> 预测 answer[1:]
                input_ids = torch.cat([enc.input_ids, ans.input_ids[:, :-1]], dim=1)
                labels = torch.cat(
                    [torch.full_like(enc.input_ids, -100), ans.input_ids], dim=1
                )  # prompt 部分不计入 loss

                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                num_loss += 1

                # --- 2. 推理生成 ---
                output = model.generate(
                    tokenizer(prompt, return_tensors="pt").to(device),
                    max_new_tokens=256,
                    temperature=0.0,
                )
                pred_text = tokenizer.decode(output[0], skip_special_tokens=True)

                # --- 3. 提取答案并比较 ---
                pred_ans = extract_final_answer(pred_text)
                gold_ans = extract_final_answer(label)

                if pred_ans == gold_ans:
                    correct += 1
                total += 1

            avg_loss = total_loss / max(1, num_loss)
            acc = correct / max(1, total)
            print(f"Accuracy: {acc:.2%}, Avg Loss: {avg_loss:.4f}")

        # if tt % 10 == 0:
        #     print(tt, " ", whole_time)
        

if __name__ == "__main__":
    train()