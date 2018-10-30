#!/bin/bash

echo -e "\n>>> Removing all old log files ... \n"
rm -rf final_* log_*.log
echo -e "\n>>> Removed all old log files . \n"

source activate ds222_as1
echo -e "\n>>> Activated ds222_as1 conda environment . \n"

python run_assignment2.py --dataset_size 'verysmall' > log_verysmall.log #  & tail -f log_verysmall.log
echo -e "\n>>> Created log_verysmall.log . \n"
python run_assignment2.py --dataset_size 'small' > log_small.log #  & tail -f log_small.log
echo -e "\n>>> Created log_small.log . \n"
python run_assignment2.py --dataset_size 'full' > log_full.log #  & tail -f log_full.log
echo -e "\n>>> Created log_full.log . \n"

source deactivate ds222_as1
echo -e "\n>>> Deactivated ds222_as1 conda environment . \n"
