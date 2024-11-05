from openai import OpenAI
from datasets import load_from_disk
import os
import pickle
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

def batch_list(input_list, batch_size):
    # Split the input_list into sublists of batch_size
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]

# Load the DatasetDict from the saved path
dataset_dict = load_from_disk("data/gsm8k")
testQ = dataset_dict['test']['question']
trainQ = dataset_dict['train']['question']

# Set your OpenAI API key
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

testQ_embs = []
Qbatches = batch_list(testQ, batch_size=4)
for text_batch in tqdm(Qbatches):
    response = client.embeddings.create(
        input=text_batch,
        model="text-embedding-3-large"
    )
    batch_embeddings = [item.embedding for item in response.data]
    testQ_embs += batch_embeddings

with open("testQ_embs.pkl", "wb") as f:
    pickle.dump(testQ_embs, f)


trainQ_embs = []
Qbatches = batch_list(trainQ, batch_size=4)
for text_batch in tqdm(Qbatches):
    response = client.embeddings.create(
        input=text_batch,
        model="text-embedding-3-large"
    )
    batch_embeddings = [item.embedding for item in response.data]
    trainQ_embs += batch_embeddings

with open("trainQ_embs.pkl", "wb") as f:
    pickle.dump(trainQ_embs, f)