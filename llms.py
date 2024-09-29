import os
import time
import threading
from queue import Queue
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class OpenAIModel():
    def __init__(self, model:str, max_tokens:int = 500, temperature: float = 1.0):
        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.temperature = temperature
    
    def generate(self, prompt: str, num_return_sequences: int = 1, rate_limit_per_min: int = 20, retry: int = 10):
        for i in range(1, retry + 1):
            try:
                if rate_limit_per_min is not None:
                    time.sleep(60 / rate_limit_per_min)

                if ('gpt-4o-2024-08-06' in self.model) or ('gpt-4o-mini' in self.model):
                    messages = [{"role": "user", "content": prompt}]
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        n=num_return_sequences,
                    )
                    self.completion_tokens += response.usage.completion_tokens
                    self.prompt_tokens += response.usage.prompt_tokens

                    text_responses = [choice.message.content for choice in response.choices]

                    return text_responses
                else:
                    print(f"Wrong Model Name !!!")
            
            except Exception as e:
                print(f"An Error Occured: {e}, sleeping for {i} seconds")
                time.sleep(i)

        raise RuntimeError(f"GPTCompletionModel failed to generate output, even after {retry} tries")

    def usage(self):
        if self.model == "gpt-4o-2024-08-06":
            cost = self.completion_tokens / 1000000 * 10.0 + self.prompt_tokens / 1000000 * 2.5
        if self.model == "gpt-4o-mini":
            cost = self.completion_tokens / 1000000 * 0.6 + self.prompt_tokens / 1000000 * 0.15

        print(f"model: {self.model}, completion_tokens: {self.completion_tokens}, prompt_tokens: {self.prompt_tokens}, cost: {cost}")



class LlamaModel():
    def __init__(self, model, temperature:float, max_tokens: int = 500):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = OpenAI(
            base_url="http://140.112.31.182:8000/v1",
            api_key=os.getenv("LLAMA_API_KEY", ""),
        )
    
    def generate(self, prompt: str, num_return_sequences: int = 1):

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": f"{prompt}"}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            n=num_return_sequences,
        )
        text_responses = [choice.message.content for choice in response.choices]

        return text_responses

