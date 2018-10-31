#!/bin/bash

# echo -e "\n>>> Preprocessing data . \n"
# python data_preprocess.py
# echo -e "\n>>> Preprocessing data done. \n"

echo -e "\n>>> Running logreg in local with constant learning rate. \n"
python logreg_local_sgd.py --decay_type 'const'
echo -e "\n>>> Running logreg in local with constant learning rate done. \n"

echo -e "\n>>> Running logreg in local with increasing learning rate. \n"
python logreg_local_sgd.py --decay_type 'dec'
echo -e "\n>>> Running logreg in local with increasing learning rate done. \n"

echo -e "\n>>> Running logreg in local with decreasing learning rate. \n"
python logreg_local_sgd.py --decay_type 'inc'
echo -e "\n>>> Running logreg in local with decreasing learning rate done. \n"

