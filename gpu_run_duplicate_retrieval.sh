#!/bin/bash
#SBATCH --mail-user=mhu05@qub.ac.uk
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH --job-name=run_eclipse
#SBATCH -p gpu
#SBATCH --time=72:00:00
#SBATCH --gres gpu:a100:1
#SBATCH --output=./result_out/eclipse_gpt_e12_n1_max_seq400_lr1e5_duplicate_out-%j.out
#SBATCH --partition=k2-gpu

#default mem is 7900M
#SBATCH --mem 80000M

source ../pre_language_model/bin/activate

module add nvidia-cuda
module add apps/python3

nvidia-smi

echo "the job start "


# eclipse,mozilla, net_beans, open_office,red_hat, hadoop, mongodb

#roberta

python  main_duplicate_bug_retrieval.py   --task_name kg    --do_duplicate_bug_retrieval   --data_dir ./data/eclipse      --model_type  roberta   --model_name_or_path  roberta-base    --max_seq_length   400   --per_gpu_train_batch_size 16    --learning_rate 1e-5    --gradient_accumulation_steps  2   --eval_batch_size  64    --pre_process_data  ./pre_process_data           --negative   1       --num_train_epochs   12   --output_dir  ./output_eclipse_e12_1_roberta_base_max_seq_400_lr1e5/   --test_file   test2id_r1000.txt   --do_train


#python  main_duplicate_bug_retrieval.py   --task_name kg    --do_duplicate_bug_retrieval   --data_dir ./data/mozilla      --model_type  roberta   --model_name_or_path  roberta-base    --max_seq_length 300   --per_gpu_train_batch_size 16    --learning_rate 5e-5    --gradient_accumulation_steps  2   --eval_batch_size  512    --pre_process_data  ./pre_process_data           --negative   3       --num_train_epochs   1   --output_dir  ./output_mozilla_e1_3_roberta_base_max_seq_300/   --test_file   test_positive2id.txt  --do_train

 #python  main_duplicate_bug_retrieval.py   --task_name kg    --do_duplicate_bug_retrieval   --data_dir ./data/net_beans    --model_type  roberta   --model_name_or_path  roberta-base    --max_seq_length 300   --per_gpu_train_batch_size 16    --learning_rate 5e-5    --gradient_accumulation_steps  2   --eval_batch_size  512    --pre_process_data  ./pre_process_data           --negative   3       --num_train_epochs   1   --output_dir  ./output_net_beans_e1_3_roberta_base_max_seq_300/   --test_file   test_positive2id.txt 

#python  main_duplicate_bug_retrieval.py   --task_name kg    --do_duplicate_bug_retrieval   --data_dir ./data/open_office  --model_type  roberta   --model_name_or_path  roberta-base    --max_seq_length 300   --per_gpu_train_batch_size 16    --learning_rate 5e-5    --gradient_accumulation_steps  2   --eval_batch_size  512    --pre_process_data  ./pre_process_data           --negative   3       --num_train_epochs   1   --output_dir  ./output_open_office_e1_3_roberta_base_max_seq_300/   --test_file   test_positive2id.txt  --do_train

#python  main_duplicate_bug_retrieval.py   --task_name kg    --do_duplicate_bug_retrieval   --data_dir ./data/red_hat      --model_type  roberta   --model_name_or_path  roberta-base    --max_seq_length 300   --per_gpu_train_batch_size 16    --learning_rate 5e-5    --gradient_accumulation_steps  2   --eval_batch_size  512    --pre_process_data  ./pre_process_data           --negative   3       --num_train_epochs   1   --output_dir  ./output_red_hat_e1_3_roberta_base_max_seq_300/   --test_file   test_positive2id.txt  --do_train

  #python  main_duplicate_bug_retrieval.py   --task_name kg    --do_duplicate_bug_retrieval   --data_dir ./data/hadoop       --model_type  roberta   --model_name_or_path  roberta-base    --max_seq_length 300   --per_gpu_train_batch_size 16    --learning_rate 5e-5    --gradient_accumulation_steps  2   --eval_batch_size  512    --pre_process_data  ./pre_process_data           --negative   3       --num_train_epochs   1   --output_dir  ./output_hadoop_e1_3_roberta_base_max_seq_300/   --test_file   test_positive2id.txt 

#python  main_duplicate_bug_retrieval.py   --task_name kg    --do_duplicate_bug_retrieval   --data_dir ./data/mongodb      --model_type  roberta   --model_name_or_path  roberta-base    --max_seq_length 300   --per_gpu_train_batch_size 16    --learning_rate 5e-5    --gradient_accumulation_steps  2   --eval_batch_size  512    --pre_process_data  ./pre_process_data           --negative   3       --num_train_epochs   1   --output_dir  ./output_mongodb_e1_3_roberta_base_max_seq_300/   --test_file   test_positive2id.txt  --do_train


#gpt2

 #python  main_duplicate_bug_retrieval.py   --task_name kg    --do_duplicate_bug_retrieval   --data_dir ./data/eclipse      --model_type  gpt2   --model_name_or_path  gpt2-medium    --max_seq_length   400   --per_gpu_train_batch_size 16    --learning_rate  1e-5    --gradient_accumulation_steps  2   --eval_batch_size  64    --pre_process_data  ./pre_process_data           --negative  1       --num_train_epochs   1   --output_dir  ./output_eclipse_e1_1_gpt_max_seq_400/   --test_file   test2id_r1000.txt  --do_train

 #  python  main_duplicate_bug_retrieval.py   --task_name kg    --do_duplicate_bug_retrieval   --data_dir ./data/mozilla      --model_type  gpt2   --model_name_or_path  gpt2-medium    --max_seq_length 300   --per_gpu_train_batch_size 16    --learning_rate 5e-5    --gradient_accumulation_steps  2   --eval_batch_size  64    --pre_process_data  ./pre_process_data           --negative   1       --num_train_epochs   1   --output_dir  ./output_mozilla_e1_1_gpt_max_seq_300/   --test_file   test_positive2id.txt  --do_train

# python  main_duplicate_bug_retrieval.py   --task_name kg    --do_duplicate_bug_retrieval   --data_dir ./data/net_beans    --model_type  gpt2   --model_name_or_path  gpt2-medium    --max_seq_length   400   --per_gpu_train_batch_size 16    --learning_rate 5e-5    --gradient_accumulation_steps  2   --eval_batch_size   64    --pre_process_data  ./pre_process_data           --negative   1       --num_train_epochs   1   --output_dir  ./output_net_beans_e1_1_gpt2_max_seq_400/   --test_file   test_positive2id.txt  --gap_time 180

 #python  main_duplicate_bug_retrieval.py   --task_name kg    --do_duplicate_bug_retrieval   --data_dir ./data/open_office  --model_type  gpt2   --model_name_or_path  gpt2-medium    --max_seq_length 300   --per_gpu_train_batch_size 16    --learning_rate 5e-5    --gradient_accumulation_steps  2   --eval_batch_size  64    --pre_process_data  ./pre_process_data           --negative   1       --num_train_epochs   1   --output_dir  ./output_open_office_e1_1_gpt2_max_seq_300/   --test_file   test_positive2id.txt  --do_train

  #python  main_duplicate_bug_retrieval.py   --task_name kg    --do_duplicate_bug_retrieval   --data_dir ./data/red_hat      --model_type  gpt2   --model_name_or_path  gpt2-medium    --max_seq_length 300   --per_gpu_train_batch_size 16    --learning_rate 5e-5    --gradient_accumulation_steps  2   --eval_batch_size  32    --pre_process_data  ./pre_process_data           --negative   3       --num_train_epochs   1   --output_dir  ./output_red_hat_e1_3_roberta_base_max_seq_300/   --test_file   test_positive2id.txt 

 # python  main_duplicate_bug_retrieval.py   --task_name kg    --do_duplicate_bug_retrieval   --data_dir ./data/hadoop       --model_type  gpt2   --model_name_or_path  gpt2-medium    --max_seq_length  400   --per_gpu_train_batch_size 16    --learning_rate 5e-5    --gradient_accumulation_steps  2   --eval_batch_size  512    --pre_process_data  ./pre_process_data           --negative   2       --num_train_epochs   2   --output_dir  ./output_hadoop_e2_2_gpt_max_seq_400/   --test_file   test_positive2id.txt   --do_train


# python  main_duplicate_bug_retrieval.py   --task_name kg    --do_duplicate_bug_retrieval   --data_dir ./data/mongodb      --model_type  gpt2   --model_name_or_path  gpt2-medium    --max_seq_length  400   --per_gpu_train_batch_size 16    --learning_rate 5e-5    --gradient_accumulation_steps  2   --eval_batch_size  64    --pre_process_data  ./pre_process_data           --negative   3       --num_train_epochs   1   --output_dir  ./output_mongodb_77K_e1_3_gpt_max_seq_400/   --test_file   test_positive2id.txt   --do_train





echo "the job end "
 




