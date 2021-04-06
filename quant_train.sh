export CUDA_VISIBLE_DEVICES=2

project_name=Q8bert
# "cola" "mrpc"  "qnli" "rte" "sst-2" "wnli" "mnli" "qqp"

for task in "sts-b" 
do
    logging_steps=25;
    case $task in 
       cola) data="CoLA";; mrpc) data="MRPC";; sts-b) data="STS-B";;
       mnli) data="MNLI";;  rte) data="RTE";; wnli) data="WNLI";;
       sst-2) data="SST-2"; logging_steps=100;; qqp) data="QQP"; logging_steps=1000;; qnli) data="QNLI"; logging_steps=300;;
    esac

    for i in 7 8 9
    do
        seed=$((i*1000))
        result_dir="../nlp_arch_results/${project_name}/${task}/${i}"
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
                --seed ${seed} \
                --wandb_project_name ${project_name} \
                --wandb_run_name "${task}${i}_loggingstep_${logging_steps}" \
                --num_train_epochs 3 \
                --logging_steps $logging_steps  \
                --save_steps 0 
    done 
done



