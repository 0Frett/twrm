from tqdm import tqdm
import pickle
import os
import json
import argparse
from llms import OpenAIModel, LlamaModel
from data_client import GSM8K_Client
import re

def get_args():
    parser = argparse.ArgumentParser(description="Args")
    parser.add_argument("--data_size", type=int, default=100, help="Size of the dataset to use.")
    parser.add_argument("--few_shot_num", type=int, default=5, help="Number of few-shot examples to include.")
    parser.add_argument("--strategy", type=str, default='random', help="random, semantic")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--unsupervised", action='store_true', help="Include this flag to enable only questions")

    
    return parser.parse_args()


def parse_answers(text):
    # print("llm response:", text)
    answer = text.split("The answer to the question is")[-1]
    cleaned_text = re.sub(r'[,$.]', '', answer)
    numbers = re.findall(r'-?\d+', cleaned_text)
    answer = 2e-5 if len(numbers) == 0 else int(numbers[0])
    print(f"parsed : {answer}")

    return answer


def parse_ground_truth(text):
    # Find the number after "####"
    match = re.search(r'#### (-?\d+)', text)
    if match:
        answer = int(match.group(1))

    return answer


def evaluate(parsed_answers, groundtruth):
    return parsed_answers==groundtruth


if __name__ == "__main__":
    args = get_args()
    # data
    data_client = GSM8K_Client()
    experiment_data = data_client.generate_dataset(
        datasize=args.data_size, 
        shotnum=args.few_shot_num, 
        strategy=args.strategy, 
        unsupervised=args.unsupervised
    )

    # llm
    llm = OpenAIModel(model="gpt-4o-mini", max_tokens=512, temperature=1.0)
    
    total_correctness = []
    for idx, instance in enumerate(tqdm(experiment_data)):
        response = llm.generate(prompt=instance['input'])[0]
        correctness = 1 if evaluate(parse_answers(response), parse_ground_truth(instance['ans'])) else 0
        total_correctness.append(correctness)
        
    overall_acc = sum(total_correctness) / len(total_correctness)
    
    print(f"Overall Accuracy:{overall_acc:2f}")

    result = {'indices_result':total_correctness, 'acc':overall_acc}
    with open(f"gsm8k_{args.strategy}_{args.few_shot_num}_{args.unsupervised}_correctness.json", "w") as f:
        json.dump(result, f)




