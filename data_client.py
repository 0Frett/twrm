import json
import random
from datasets import load_from_disk
from typing import List, Dict, Union
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class StrategyQA_Client:
    """ return batches of prompt"""
    def __init__(self):
        with open("./data/strategyQA/strategyqa_test.json", 'r') as f:
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


    def generate_qa_instance(
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
    

    def generate_q_instance(
            self, 
            q_batch:int
        ):
        qs = "\n".join([f"Q[{idx+1}]:{q[0]}" for idx, q in enumerate(q_batch)])
        ans = [q[1] for q in q_batch]
        prompt = f"Answer the following questions in True or False and respond answer in format: A[1]:<answer to question1> \n\n{qs}"

        return {'input':prompt, 'ans':ans}
    

    def generate_dataset(
            self, 
            data_size:int, 
            few_shot_num:int, 
            batch_size:int, 
            seed:int,
            onlyq:bool=False
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
            if onlyq:
                batch_prompt_instance = self.generate_q_instance(q_batch)
            else:
                few_shot_batch = random.sample(few_shots, few_shot_num)
                batch_prompt_instance = self.generate_qa_instance(few_shot_batch, q_batch)
            batched_dataset.append(batch_prompt_instance)

        return batched_dataset
    


class GSM8K_Client:
    def __init__(self):
        dataset = load_from_disk("data/gsm8k")
        with open("./data/gsm8k/embeds/testQ_embs.pkl", "rb") as f:
            self.testQ_embs = pickle.load(f)
        with open("./data/gsm8k/embeds/trainQ_embs.pkl", "rb") as f:
            self.trainQ_embs = pickle.load(f)
        self.trainQ = dataset['train']['question']
        self.trainA = dataset['train']['answer']
        self.testQ = dataset['test']['question']
        self.testA = dataset['test']['answer']
    
    def generate_dataset(self, datasize:int, unsupervised:bool, strategy:str, shotnum:int):
        dataQ = self.testQ[:datasize]
        dataA = self.testA[:datasize]
        dataQ_emb = self.testQ_embs[:datasize]
        data_pairs = []
        for idx in range(datasize):
            source_q = dataQ[idx]
            source_a = dataA[idx]
            source_emb = dataQ_emb[idx]
            if shotnum > 0:
                # few shot
                if unsupervised:
                    data_pairs.append(self.generate_unsupervised_single_qa_pair(source_q, source_a, source_emb, strategy, shotnum))
                else:
                    data_pairs.append(self.generate_supervised_single_qa_pair(source_q, source_a, source_emb, strategy, shotnum))
            else:
                # zero shot
                prompt = f"""Please answer the following question. 
                    Your answer must ends with the sentence: The answer to the question is <numeric answer>.
                    Q:{source_q}\n
                    A:"""
                data_pairs.append({'input':prompt, 'ans':source_a})

        return data_pairs


    def generate_unsupervised_single_qa_pair(self, source_q, source_a, source_emb, strategy, shotnum):
        if strategy == "random":
            q_batch = self.random_select_instances(shotnum)
        elif strategy == "semantic":
            q_batch = self.find_semantic_similar_instances(source_emb, shotnum)
        # q_batch = [(q,a), (q,a), ...]

        qs = "\n".join([f"Q[{idx+1}]:{q[0]}" for idx, q in enumerate(q_batch)])
        # ans = [q[1] for q in q_batch]
        prompt = f"""You are given a collection of questions. Please answer the last question. 
                    Your answer must ends with the sentence: The answer to the question is <numeric answer>.\n
                    {qs}\n
                    Q[{shotnum+1}]:{source_q}\n
                    A[{shotnum+1}]:"""
        # prompt = f"""You are given a collection of questions. Please answer each question. 
        #     Your answer must ends with the sentence: The answer to the question is <numeric answer>.\n
        #     {qs}\n
        #     Q[{shotnum+1}]:{source_q}\n
        #     A[{shotnum+1}]:"""

        return {'input':prompt, 'ans':source_a}
    
    def generate_supervised_single_qa_pair(self, source_q, source_a, source_emb, strategy, shotnum):
        if strategy == "random":
            q_batch = self.random_select_instances(shotnum)
        elif strategy == "semantic":
            q_batch = self.find_semantic_similar_instances(source_emb, shotnum)
        # q_batch = [(q,a), (q,a), ...]
        qs = "\n".join([f"Q[{idx+1}]:{q[0]}\nA[{idx+1}]:{q[1]}" for idx, q in enumerate(q_batch)])
        prompt = f"""You are given a collection of questions. Please answer the last question. 
                    Your answer must ends with the sentence: The answer to the question is ...\n
                    {qs}\n
                    Q[{shotnum+1}]:{source_q}\n
                    A[{shotnum+1}]:"""

        return {'input':prompt, 'ans':source_a}
    
    def find_semantic_similar_instances(self, source_emb, instance_num):
        topk_pairs = self.get_top_k_similar_embeddings(source_emb, self.trainQ_embs, instance_num) # (idx, embeddings[idx], similarities[idx])
        textQ = []
        for pair in topk_pairs:
            textQ.append((self.trainQ[pair[0]], self.trainA[pair[0]]))

        return textQ
    
    def random_select_instances(self, instance_num):
        selected_indices = random.sample(range(len(self.trainQ)), instance_num)
        selected_instances = [(self.trainQ[idx], self.trainA[idx]) for idx in selected_indices]

        return selected_instances

    def get_top_k_similar_embeddings(self, source_embedding, embedding_pool, k):
        # Compute cosine similarity between the source and all other embeddings
        similarities = cosine_similarity([source_embedding], embedding_pool)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]  # Sort and get the largest k values in descending order
        top_k_results = [(idx, embedding_pool[idx], similarities[idx]) for idx in top_k_indices]

        return top_k_results



