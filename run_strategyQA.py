from tqdm import tqdm
import pickle
import os
import json
import argparse
from llms import OpenAIModel, LlamaModel
from data_client import StrategyQA_Client
import re

def get_args():
    parser = argparse.ArgumentParser(description="Args")
    parser.add_argument("--data_size", type=int, default=100, help="Size of the dataset to use.")
    parser.add_argument("--few_shot_num", type=int, default=5, help="Number of few-shot examples to include.")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--onlyq", action='store_true', help="Include this flag to enable only questions")

    
    return parser.parse_args()


def parse_answers(text):
    # Regular expression to match the pattern 'A[n]: <answer>'
    pattern = r"A\[(\d+)\]:\s*(.*)"
    matches = re.findall(pattern, text)
    if len(matches) == 0:
        pattern = r"Q\[(\d+)\]:\s*(.*)"
        matches = re.findall(pattern, text)
    answers = []
    for _, answer in matches:
        answer = answer.strip().lower()
        if 'true' in answer or 'yes' in answer:
            answers.append(True)
        elif 'false' in answer or 'no' in answer:
            answers.append(False)
        else:
            answers.append(answer)

    return answers


def evaluate(parsed_answers, groundtruth):
    correct_count = 0
    total = len(groundtruth)

    for i, (parsed, true_value) in enumerate(zip(parsed_answers, groundtruth)):
        if isinstance(parsed, bool) and parsed == true_value:
            correct_count += 1
        else:
            print(f"Answer {i+1} is incorrect: Expected {true_value}, but got {parsed}")
    
    accuracy = correct_count / total * 100

    return accuracy


if __name__ == "__main__":
    args = get_args()
    # data
    data_client = StrategyQA_Client()
    experiment_data = data_client.generate_dataset(
        data_size=args.data_size, 
        few_shot_num=args.few_shot_num, 
        batch_size=args.batch_size, 
        seed=args.seed,
        onlyq=args.onlyq
    )
    # llm
    llm = OpenAIModel(model="gpt-4o-mini", max_tokens=4096, temperature=0.6)
    
    total_accs = []
    for idx, instance in enumerate(tqdm(experiment_data)):
        response = llm.generate(prompt=instance['input'])[0]
        accuracy = evaluate(parse_answers(response), instance['ans'])
        total_accs.append(accuracy)
        print(f"================ Batch {idx} ================\n"
            f"{instance['input']}\n"
            f"{response}\n"
            f"Batch Accuracy: {accuracy:.2f}%")
        
    overall_acc = sum(total_accs) / len(total_accs)
    print("Overall Accuracy:", overall_acc)




