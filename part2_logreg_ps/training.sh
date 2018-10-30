#!/bin/bash

source activate ds222_as1
echo -e "\n>>> Activated ds222_as1 conda environment . \n"

echo -e "\n>>> Running Bulk Synchronous sgd. \n"
python logreg_BSP_sgd.py
echo -e "\n>>> Running Bulk Synchronous sgd done. \n"

echo -e "\n>>> Running Stale Synchronous sgd. \n"
python logreg_SSP_sgd.py
echo -e "\n>>> Running Stale Synchronous sgd done. \n"

echo -e "\n>>> Running Asynchronous sgd. \n"
python logreg_ASP_sgd.py
echo -e "\n>>> Running Asynchronous sgd done. \n"

source deactivate ds222_as1
echo -e "\n>>> Deactivated ds222_as1 conda environment . \n"
