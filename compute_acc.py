import argparse
import os
import pandas as pd

import pdb

# Parser
parser = argparse.ArgumentParser(description="Computing acc for ViperGPT...")
parser.add_argument('--data_name', type=str, choices=['MNMath_Add_3digit', 'MNMath_Add_5digit', 'MNMath_Add_10digit', 'MNLogic_XOR_3digit', 'MNLogic_XOR_5digit', 'MNLogic_XOR_10digit', 'KandLogic_3obj', 'KandLogic_5obj'])
parser.add_argument('--model_name', type=str, help='the simple query model', choices=['blip', 'qwen'])
args = parser.parse_args()

# Load the CSV file
result_dir = os.path.join("/home/jovyan/workspace/viper/results", args.data_name)
result_path = os.path.join(result_dir, f"{args.model_name}_results.csv")
df = pd.read_csv(result_path)

# Compute accuracy
"""
MNMath:
df["result"]: float
df["answer"]: int

MNLogic:
df["result"]: float
df["answer"]: bool

KandLogic:
df["result"]: bool
df["answer"]: bool
"""
def preprocess_pred(x):
    return float(x)

def preprocess_label(x):
    return float(x)

accuracy = (df["result"].apply(preprocess_pred) == df["answer"].apply(preprocess_label)).mean()
print(f"Accuracy: {accuracy:.2%}")
