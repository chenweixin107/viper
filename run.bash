#!/usr/bin/env bash

seed=3

# Use `self.qa_prompt = "Question: {} Please output one number only. Answer:"` in vision_models.py
CONFIG_NAMES=MNLogic_XOR_3digit python main_batch.py > "/home/jovyan/workspace/viper/results/MNLogic_XOR_3digit/log_seed_${seed}.txt" 2>&1
CONFIG_NAMES=MNLogic_XOR_5digit python main_batch.py > "/home/jovyan/workspace/viper/results/MNLogic_XOR_5digit/log_seed_${seed}.txt" 2>&1

CONFIG_NAMES=MNMath_Add_3digit python main_batch.py > "/home/jovyan/workspace/viper/results/MNMath_Add_3digit/log_seed_${seed}.txt" 2>&1
CONFIG_NAMES=MNMath_Add_5digit python main_batch.py > "/home/jovyan/workspace/viper/results/MNMath_Add_5digit/log_seed_${seed}.txt" 2>&1

## Use `self.qa_prompt = "Question: {} Please output one word in lower case only. Answer:"` in vision_models.py
#CONFIG_NAMES=KandLogic_3obj python main_batch.py > "/home/jovyan/workspace/viper/results/KandLogic_3obj/log_seed_${seed}.txt" 2>&1
#CONFIG_NAMES=KandLogic_5obj python main_batch.py > "/home/jovyan/workspace/viper/results/KandLogic_5obj/log_seed_${seed}.txt" 2>&1
