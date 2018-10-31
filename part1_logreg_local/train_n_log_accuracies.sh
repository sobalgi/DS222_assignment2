#!/bin/bash

source activate ds222_as1
echo -e "\n>>> Activated ds222_as1 conda environment . \n"

echo -e "\n>>> Preprocessing data . \n"
python data_preprocess.py
echo -e "\n>>> Preprocessing data done. \n"

echo -e "\n>>> Running logreg in local . \n"
python logreg_local_sgd.py
echo -e "\n>>> Running logreg in local done. \n"

source deactivate ds222_as1
echo -e "\n>>> Deactivated ds222_as1 conda environment . \n"
