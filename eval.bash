#!/usr/bin/env bash

# --model_name blip
python compute_acc.py --data_nam MNMath_Add_3digit --model_name blip # Accuracy: 0.00%
python compute_acc.py --data_nam MNMath_Add_5digit --model_name blip # Accuracy: 0.00%

python compute_acc.py --data_nam MNLogic_XOR_3digit --model_name blip # Accuracy: 32.23%
python compute_acc.py --data_nam MNLogic_XOR_5digit --model_name blip # Accuracy: 17.47%

python compute_acc.py --data_nam KandLogic_3obj --model_name blip # Accuracy: 49.67%
python compute_acc.py --data_nam KandLogic_5obj --model_name blip # Accuracy: 48.23%

## --model_name qwen
#python compute_acc.py --data_nam MNMath_Add_3digit --model_name qwen
#python compute_acc.py --data_nam MNMath_Add_5digit --model_name qwen
#
#python compute_acc.py --data_nam MNLogic_XOR_3digit --model_name qwen
#python compute_acc.py --data_nam MNLogic_XOR_5digit --model_name qwen
#
#python compute_acc.py --data_nam KandLogic_3obj --model_name qwen
#python compute_acc.py --data_nam KandLogic_5obj --model_name qwen