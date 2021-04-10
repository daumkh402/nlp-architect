export CUDA_VISIBLE_DEVICES=0,1

project_name=Q8bert_EMA_SCALE
#   "qnli"  "sst-2" "qqp" "wnli" "mnli" 
for task in "cola"  #"mrpc" "rte" "sst-2" "cola" "sts-b"
do
    case $task in 
       cola) data="CoLA"; logging_steps=10;;
       mrpc) data="MRPC"; logging_steps=4;; 
       sts-b) data="STS-B"; logging_steps=6;;
       mnli) data="MNLI"; logging_steps=10;;
       rte) data="RTE"; logging_steps=3;;
       wnli) data="WNLI"; logging_steps=10;;
       sst-2) data="SST-2"; logging_steps=100;; 
       qqp) data="QQP"; logging_steps=500;; 
       qnli) data="QNLI"; logging_steps=160;;
    esac
    
    for warmup in 0 #50 100 150
    do
	for lr in 4e-5  #2e-5 3e-5 4e-5 5e-5 
	    do
		for i in 1 2 3 
		    do
			seed=$((i*1000))
			result_dir="../nlp_arch_results/${project_name}/${task}/lr_${lr}/${i}"
			if [ ! -d ${result_dir} ]
				then
				echo "${result_dir} does not exist"
				mkdir -p ${result_dir}
			fi
			logname="${result_dir}/${project_name}_${task}_${i}.txt"

			nlp-train transformer_glue \
				--task_name ${task} \
				--model_name_or_path bert-base-uncased \
				--model_type quant_bert \
				--output_dir ${result_dir} \
				--evaluate_during_training \
				--data_dir ../glue_data/${data} \
				--do_lower_case \
				--overwrite_output_dir \
				--seed $RANDOM \
				--wandb_project_name ${project_name} \
				--wandb_run_name "${task}${i}_lr_${lr}_loggingstep_${logging_steps}" \
				--num_train_epochs 3 \
				--per_gpu_train_batch_size 16 \
				--per_gpu_eval_batch_size 16 \
				--learning_rate ${lr} \
				--logging_steps $logging_steps  \
				--warmup_steps ${warmup} \
                --save_steps 0 
		    done

        done	 
    
    done  
     
done



