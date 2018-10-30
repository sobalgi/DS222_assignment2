#!/bin/bash
MY_HADOOP_PYTHON="/home/sourabhbalgi/anaconda3/bin/python"  # envs/ds222_as1/
MY_HADOOP_JAR_PATH="/usr/hdp/current/hadoop-mapreduce-client/hadoop-streaming.jar"
MY_HADOOP_IN_DIR="/user/sourabhbalgi"
MY_HADOOP_OUT_DIR="$MY_HADOOP_IN_DIR/out_ds222_as1"

#export MY_HADOOP_PYTHON="/home/sourabhbalgi/anaconda3/bin/python"  # envs/ds222_as1/
#export MY_HADOOP_JAR_PATH="/usr/hdp/current/hadoop-mapreduce-client/hadoop-streaming.jar"
#export MY_HADOOP_IN_DIR="/user/sourabhbalgi"
#export MY_HADOOP_OUT_DIR="$MY_HADOOP_IN_DIR/out_ds222_as1"

#: <<'CodeBlockDisable0'
hadoop fs -ls $MY_HADOOP_IN_DIR/                                             
hdfs dfs -rm -r "$MY_HADOOP_IN_DIR/*"                                        hdfs dfs -rm -r "$MY_HADOOP_IN_DIR/*"

hadoop fs -ls $MY_HADOOP_IN_DIR/

# Create output folder to store data
echo -e "\n>>> Creating output folder out_ds222_as1 on hdfs to store output files of mapreduce. \n"
hadoop fs -mkdir $MY_HADOOP_OUT_DIR

chmod +x *.py

#hadoop fs -rm -r -f -skipTrash $MY_HADOOP_IN_DIR/*

#CodeBlockDisable0

#: <<'CodeBlockDisable1'
echo -e "\n>>> Starting get_modelparams_mapred to get model parameters ... \n"
hadoop fs -rm -skipTrash $MY_HADOOP_IN_DIR/modelparams_mapred_$1/*
hadoop fs -rmdir $MY_HADOOP_IN_DIR/modelparams_mapred_$1

hadoop jar $MY_HADOOP_JAR_PATH \
-file "./get_modelparams_map_n.py" -mapper "$MY_HADOOP_PYTHON get_modelparams_map_n.py" \
-file "./get_modelparams_red_n.py" -reducer "$MY_HADOOP_PYTHON get_modelparams_red_n.py" \
-input "/user/ds222/assignment-1/DBPedia.full/full_train.txt" \
-output "$MY_HADOOP_IN_DIR/modelparams_mapred_$1" \
-numReduceTasks $1

echo -e "\n>>> Number of reducers used by get_modelparams_mapred = $1. \n"

echo -e "\n>>> get_modelparams_mapred Job completed. \n"

hadoop fs -rm -skipTrash $MY_HADOOP_IN_DIR/modelparams_mapred/*
hadoop fs -rmdir $MY_HADOOP_IN_DIR/modelparams_mapred

hadoop jar $MY_HADOOP_JAR_PATH \
-file "./get_modelparams_map_final.py" -mapper "$MY_HADOOP_PYTHON get_modelparams_map_final.py" \
-file "./get_modelparams_red_final.py" -reducer "$MY_HADOOP_PYTHON get_modelparams_red_final.py" \
-input "$MY_HADOOP_IN_DIR/modelparams_mapred_$1" \
-output "$MY_HADOOP_IN_DIR/modelparams_mapred" \
-numReduceTasks 1

echo -e "\n>>> Completed get_modelparams_mapred Job. \n"

rm ./modelparams_$1.txt
echo -e "\n>>> Deleted previous file ./modelparams_$1.txt. \n"

# combine the output files from get_modelparams_mapred
hadoop fs -cat $MY_HADOOP_IN_DIR/modelparams_mapred/part-* > modelparams_$1.txt
echo -e "\n>>> Combined outputs from get_modelparams_mapred into single file ./modelparams_$1.txt. \n"

# get model param details
cat modelparams_$1.txt | grep '_vocablen'
cat modelparams_$1.txt | grep '_vocab ' -c
cat modelparams_$1.txt | grep '01prior' -c
cat modelparams_$1.txt | grep '\-UNK\-' -c
cat modelparams_$1.txt | grep '02cond' -c

#CodeBlockDisable1
