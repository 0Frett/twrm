import json
import random
from typing import List, Dict, Union


class StrategyQA_Client:
    """ return batches of prompt"""
    def __init__(self):
        with open("./data/strategyqa_test.json", 'r') as f:
            dataset = json.load(f)
        self.dataset = list(dataset)


    def random_select(
            self, 
            pool:List[Union[Dict, str]], 
            select_num:int, 
            seed:int
        ):
        random.seed(seed)
        if select_num > len(pool):
            raise ValueError("select_num is greater than the number of items in the pool")
        selected_items = random.sample(pool, select_num)
        
        return selected_items


    def generate_basic_instance(
            self, 
            few_shot_batch:int, 
            q_batch:int
        ):
        few_shot_examples = "\n".join([f"Q[{idx+1}]:{ex[0]}" for idx, ex in enumerate(few_shot_batch)]) + "\n" + \
                            "\n".join([f"A[{idx+1}]:{ex[1]}" for idx, ex in enumerate(few_shot_batch)])
        qs = "\n".join([f"Q[{idx+1}]:{q[0]}" for idx, q in enumerate(q_batch)])
        ans = [q[1] for q in q_batch]
        prompt = f"{few_shot_examples}\n\n{qs}"

        return {'input':prompt, 'ans':ans}


    def generate_dataset(
            self, 
            data_size:int, 
            few_shot_num:int, 
            batch_size:int, 
            seed:int
        ):
        select_data = self.dataset[:data_size+few_shot_num*10]
        few_shot_pools = self.random_select(pool=select_data, select_num=few_shot_num*10, seed=seed)
        remaining_data = [item for item in select_data if item not in few_shot_pools]
        assert len(remaining_data)+len(few_shot_pools) == len(select_data)
        test_instances = [[f"{instance['question']}", instance['answer']] for instance in remaining_data]
        few_shots = [[f"{shot['question']}", f"{shot['answer']}"] for shot in few_shot_pools]

        random.seed(seed)
        random.shuffle(test_instances)
        batched_dataset = []
        for i in range(0, len(test_instances), batch_size):
            q_batch = test_instances[i:i + batch_size]
            few_shot_batch = random.sample(few_shots, few_shot_num)
            batch_prompt_instance = self.generate_basic_instance(few_shot_batch, q_batch)
            batched_dataset.append(batch_prompt_instance)

        return batched_dataset
    
    
        



