import torch
from utility.loss import extract_final_answer
from data.gsm8k import GSM8k

import re

def normalize_number(text: str):
    """
    将数字字符串规范化为 float，处理逗号、百分号、小数点等
    若无法解析则返回 None
    """
    if text is None:
        return None

    text = text.strip()
    if text == "":
        return None

    # 去掉逗号和空格
    text = text.replace(",", "").replace(" ", "")

    # 匹配百分号（转化为小数）
    if text.endswith("%"):
        try:
            val = float(text[:-1])
            return val / 100.0
        except ValueError:
            return None

    # 尝试直接解析为 float
    try:
        return float(text)
    except ValueError:
        pass

    # 匹配整数形式
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None

    return None


def is_equiv_answer(pred: str, truth: str, tol: float = 1e-4) -> bool:
    """
    判断两个答案在语义上是否相等
    - 支持格式差异（5600 vs 5,600 vs 5600.0）
    - 支持百分比 (56% vs 0.56)
    - 支持微小浮点误差
    """
    if pred is None or truth is None:
        return False

    if pred.strip().lower() == truth.strip().lower():
        return True

    # 尝试数值比较
    pred_val = normalize_number(pred)
    true_val = normalize_number(truth)

    if pred_val is not None and true_val is not None:
        return abs(pred_val - true_val) < tol * max(1.0, abs(true_val))

    return False


def evaluate_model(model, tokenizer, device, batch_size, threads):
    test_dataset = GSM8k(batch_size, threads)
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_dataset.test:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].clone().to(device)
            labels[labels == tokenizer.pad_token_id] = -100 
            final_answers = batch["final_answer"]

            # === loss ===
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=16,
                do_sample=True,
                temperature = 0.3,
                top_p = 0.6,
                repetition_penalty = 1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            for gen_text, true_text in zip(
                tokenizer.batch_decode(generated, skip_special_tokens=True),
                final_answers
            ):
                pred_ans = extract_final_answer(gen_text)
                true_ans = true_text

                print("pred_ans:", pred_ans)
                print("true_ans:", true_ans)

                if is_equiv_answer(pred_ans, true_ans):
                    correct += 1
                total += 1

                print(f"Current avg_loss: {total_loss / total_samples:.4f}, accuracy: {correct / total:.4f}")

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = correct / total if total > 0 else 0

    print("\n=== Evaluation Summary ===")
    print(f"Total samples: {total_samples}, Correct: {correct}, Evaluated: {total}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2%}")
    
    return avg_loss, accuracy
