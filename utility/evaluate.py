import torch
from utility.loss import extract_final_answer
from data.gsm8k import GSM8k
import datetime
import os
import re
from typing import Optional

# 严格解析：整个字符串必须仅表示一个数字（允许千分号，货币符号，百分号，科学计数法）
_STRICT_NUM_RE = re.compile(
    r"""^\s*                # 可选前导空白
        [\+\-]?             # 可选正负号
        (?:[\$¥€£])?        # 可选货币符号（前缀）
        (?:
            (?:\d{1,3}(?:,\d{3})*(?:\.\d+)?)   # 带千分逗号的整数或小数，如 1,234 or 1,234.56
            |
            (?:\d+(?:\.\d+)?)                 # 或普通不带逗号的数字 1234 或 1234.56
            |
            (?:\d+(?:\.\d+)?[eE][\+\-]?\d+)   # 或科学计数法 1.2e3
        )
        %?                   # 可选百分号（末尾）
        \s*                  # 可选尾随空白
        [\.\,]?              # 允许末尾的句点或逗号（如 "1500."）
        \s*$                 # 结束
    """,
    re.VERBOSE,
)

def _strict_normalize_number(text: str) -> Optional[float]:
    """
    仅当整个 text 表示一个数字（可能含货币符号/逗号/百分号/科学计数法）时，
    返回对应的 float；否则返回 None。
    """
    if text is None:
        return None
    s = text.strip()
    if s == "":
        return None

    # 先用正则检查整体格式，拒绝像 "2 boxes" 之类
    if not _STRICT_NUM_RE.match(s):
        return None

    # 去掉货币符号（若有）、去掉尾部句点或逗号，再去千分逗号
    s = s.strip()
    s = re.sub(r'^[\+\-]?\s*([\$¥€£])', lambda m: m.group(0).replace(m.group(1), ''), s)  # remove currency if prefix
    s = s.rstrip('.,')  # remove trailing '.' or ',' that we allowed
    # Remove spaces then commas used as thousand separator
    s = s.replace(" ", "").replace(",", "")

    # 百分号处理
    is_percent = False
    if s.endswith("%"):
        is_percent = True
        s = s[:-1]

    # 现在尝试直接转换为 float（支持科学计数法）
    try:
        val = float(s)
    except ValueError:
        return None

    if is_percent:
        val = val / 100.0
    return val


def is_equiv_answer(pred: str, truth: str, tol: float = 1e-4) -> bool:
    """
    更严格的答案匹配函数（默认行为）：
    - 首先做去空格、小写化的字符串精确比较（对文本答案）
    - 然后尝试严格解析两边是否为**完整的数字表达式**
      （拒绝包含额外词语的预测，如 "2 boxes of pizza"）
    - 若两边均可解析为数字，则按 tol 比较数值相等
    - 否则返回 False
    """
    if pred is None or truth is None:
        return False

    p = pred.strip()
    t = truth.strip()

    # 先做宽松的字符串匹配（忽略大小写/多余空格/逗号）
    if p.lower() == t.lower():
        return True

    # 尝试严格数值解析（要求整个字符串是数字形式）
    p_val = _strict_normalize_number(p)
    t_val = _strict_normalize_number(t)

    if p_val is not None and t_val is not None:
        # 使用相对容忍度比较
        return abs(p_val - t_val) < tol * max(1.0, abs(t_val))

    # 如果任一 side 不是严格数字形式，则不要认为等价
    return False


def evaluate_model(model, tokenizer, device, batch_size, threads, log_dir="./logs"):
    # === 创建日志文件 ===
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"eval_log_{timestamp}.txt")
    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg):
        """打印 + 写入日志"""
        print(msg)
        log_file.write(msg + "\n")

    # === 加载测试集 ===
    test_dataset = GSM8k(batch_size, threads)
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    total = 0

    log("=== Starting Evaluation ===")
    log(f"Device: {device}")
    log(f"Batch size: {batch_size}, Threads: {threads}")
    log(f"Log file: {log_path}")
    log("=" * 50)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataset.test):
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

            # === 生成 ===
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # === 比较预测 ===
            for gen_text, true_text in zip(
                tokenizer.batch_decode(generated, skip_special_tokens=True),
                final_answers
            ):
                pred_ans = extract_final_answer(gen_text)
                true_ans = true_text

                log(f"[Batch {batch_idx}] pred_ans: {pred_ans}")
                log(f"[Batch {batch_idx}] true_ans: {true_ans}")

                if is_equiv_answer(pred_ans, true_ans):
                    correct += 1
                total += 1

                log(f"Current avg_loss: {total_loss / total_samples:.4f}, accuracy: {correct / total:.4f}")
                log("-" * 40)

                log("ASY")

    # === 汇总结果 ===
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = correct / total if total > 0 else 0

    log("\n=== Evaluation Summary ===")
    log(f"Total samples: {total_samples}, Correct: {correct}, Evaluated: {total}")
    log(f"Average loss: {avg_loss:.4f}")
    log(f"Accuracy: {accuracy:.2%}")
    log("=" * 50)

    log_file.close()
    print(f"\n✅ Evaluation log saved to: {log_path}")

    return avg_loss, accuracy

    
    return avg_loss, accuracy
