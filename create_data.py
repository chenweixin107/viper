import argparse
import joblib
import json
import os
import pandas as pd

import pdb

# Parser
parser = argparse.ArgumentParser(description="Creating data for ViperGPT...")
parser.add_argument('--data_name', type=str, choices=['MNMath_Add_3digit', 'MNMath_Add_5digit', 'MNMath_Add_10digit', 'MNLogic_XOR_3digit', 'MNLogic_XOR_5digit', 'MNLogic_XOR_10digit', 'KandLogic_3obj', 'KandLogic_5obj', 'CLE4EVR_3obj', 'CLE4EVR_5obj'])
args = parser.parse_args()
data_dir = os.path.join('/home/jovyan/workspace/datasets', args.data_name, 'test')

# Load query
with open('/home/jovyan/workspace/llm_pc/config/task_prompt_viper.json', 'r') as f:
    queries = json.load(f)
query = queries[args.data_name]

# Define the column names
columns = ["index", "sample_id", "possible_answers", "query_type", "query", "answer", "image_name"]

data_list = []
# Load data
if 'cle4evr' in args.data_name.lower():
    base_names = set()
    for filename in os.listdir(data_dir):
        base_name = os.path.splitext(filename)[0]
        base_names.add(base_name)

    for i, base_name in enumerate(base_names):
        img_path = os.path.join(data_dir, f"{base_name}.png")
        data_path = os.path.join(data_dir, f"{base_name}.json")
        with open(data_path, 'r') as f:
            data = json.load(f)
        label = bool(data['label'])
        data_list.append([i, i, '', '', query, str(label), img_path])
else:
    for i in range(3000):
        img_path = os.path.join(data_dir, f"{str(i)}.png")
        data_path = os.path.join(data_dir, f"{str(i)}.joblib")
        data = joblib.load(data_path)
        label, concept_labels = data['label'], data['meta']['concepts']
        data_list.append([i, i, '', '', query, str(label), img_path])

# Save as CSV
df = pd.DataFrame(data_list, columns=columns)
save_path = os.path.join('/home/jovyan/workspace/viper/data', f"{args.data_name}.csv")
df.to_csv(save_path, index=False)