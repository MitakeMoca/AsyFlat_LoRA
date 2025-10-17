import torch

from utility.loss import extract_final_answer
from data.gsm8k import GSM8k

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
            labels = batch["labels"].to(device)
            final_answers = batch["final_answer"]

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
                max_new_tokens=8,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            for gen_text, true_text in zip(
                tokenizer.batch_decode(generated, skip_special_tokens=True),
                tokenizer.batch_decode(labels, skip_special_tokens=True)
            ):
                pred_ans = extract_final_answer(gen_text)
                true_ans = extract_final_answer(true_text)

                print("pred_ans:", pred_ans)
                print("true_ans:", true_ans)

                if pred_ans is not None and true_ans is not None and pred_ans == true_ans:
                    correct += 1
                total += 1

                print("Current avg_loss:", total_loss / total_samples, "accuracy:", correct / total)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = correct / total if total > 0 else 0

    print("\n=== Evaluation Summary ===")
    print(f"Total samples: {total_samples}, Correct: {correct}, Evaluated: {total}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2%}")
    
    return avg_loss, accuracy
