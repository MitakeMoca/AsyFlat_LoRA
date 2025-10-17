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
    model.config.pad_token_id = tokenizer.pad_token_id
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

    text = "Hello world!"
    ids = tokenizer.encode(text)
    print(tokenizer.batch_decode(ids, skip_special_tokens=True))
    print(tokenizer.decode(ids))

    for epoch in range(args["epochs"]):
        model.train()
        tt = 0

        # 规约化后的最大值，在不断上升
        fmax_ = args["fmax"] - ((args["epochs"] - epoch) / args["epochs"]) * (args["fmax"] - args["fmin"]) + 0.000000001

        for batch in dataset.train:
            start_time = time.time()
            tt += 1
            inputs, targets, index = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            base_optimizer.zero_grad()
            outputs = model(inputs).logits
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                targets.view(-1),
                ignore_index=tokenizer.pad_token_id,
                label_smoothing=0.1
            )
            loss.backward()
            base_optimizer.step()

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

            if tt % 10 == 0 :
                print(whole_time)

        end_time = time.time()
        es_time = end_time - start_time
        whole_time += es_time
        print(whole_time)

        test_dataset = GSM8k(int(args["batch_size"] / 2), args["threads"])
        model.eval()
        total_loss = 0.0
        total_samples = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_dataset.test:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                final_answers = batch["final_answer"]  # list[str]

                # === 计算 loss ===
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item() * input_ids.size(0)
                total_samples += input_ids.size(0)

                # === 生成预测 ===
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

                for gen_text, true_text in zip(
                    tokenizer.batch_decode(generated, skip_special_tokens=True),
                    tokenizer.batch_decode(labels, skip_special_tokens=True)
                ):
                    pred_ans = extract_final_answer(gen_text)
                    true_ans = extract_final_answer(true_text)
                    print("pred_ans: ", pred_ans)
                    print("true_ans: ", true_ans)

                    if pred_ans is not None and true_ans is not None and pred_ans == true_ans:
                        correct += 1
                    total += 1

                    print(total_loss / total_samples, correct / total)

        # === 打印结果 ===
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = correct / total if total > 0 else 0
        print(total_loss, correct, total_samples, total)
        print(f"Average loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.2%}")

        # if tt % 10 == 0:
        #     print(tt, " ", whole_time)


if __name__ == "__main__":
    train()