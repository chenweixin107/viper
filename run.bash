#!/usr/bin/env bash

CONFIG_NAMES=MNLogic_XOR_3digit python main_batch.py
CONFIG_NAMES=MNLogic_XOR_5digit python main_batch.py

CONFIG_NAMES=MNMath_Add_5digit python main_batch.py
CONFIG_NAMES=MNMath_Add_3digit python main_batch.py

CONFIG_NAMES=KandLogic_3obj python main_batch.py
CONFIG_NAMES=KandLogic_5obj python main_batch.py