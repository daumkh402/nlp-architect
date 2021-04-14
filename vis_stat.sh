export CUDA_VISIBLE_DEVICES=0,1

project_name=FPbert_visualization
# "cola" "mrpc" "qnli" "rte" "sts-b" "sst-2" "qqp"              "wnli" "mnli" 
for task in  "cola" "mrpc" "rte" "sst-2" "qqp"  #"sts-b" 
do
    logging_steps=25;
    case $task in 
        cola) data="CoLA"; logging_steps=530; lr=2e-5;; 
	mrpc) data="MRPC"; logging_steps=225; lr=3e-5;; 
	sts-b) data="STS-B"; logging_steps=350; lr=3e-5;;
        mnli) data="MNLI"; logging_steps=24500; lr=2e-5;;  
	rte) data="RTE"; logging_steps=150; lr=2e-5;;
        wnli) data="WNLI"; logging_steps=40; lr=2e-5;;
        sst-2) data="SST-2"; logging_steps=4200; lr=2e-5;;
	qqp) data="QQP"; logging_steps=22500; lr=3e-5;;
	qnli) data="QNLI"; logging_steps=6500; lr=2e-5;;
    esac

    run_name="${task}_${i}_lr_${lr}_loggingstep_${logging_steps}" 
    h=0
    writer_dir="../tensorboard/${project_name}/${run_name}_${h}"
    while [ -d ${writer_dir} ]
    do
        h=$((h+1))
        writer_dir="../tensorboard/${project_name}/${run_name}_${h}"
    done



    for i in 1
    do  
	result_dir="../nlp_arch_results/${project_name}/${task}/lr_${lr}/${i}"
	if [ ! -d ${result_dir} ]
	then
           echo "${result_dir} does not exist"
           mkdir -p ${result_dir}
	fi
        
        nlp-train transformer_glue \
                --task_name ${task} \
                --model_name_or_path bert-base-uncased \
                --model_type bert \
                --output_dir ${result_dir} \
                --evaluate_during_training \
                --data_dir ../glue_data/${data} \
                --do_lower_case \
                --overwrite_output_dir \
                --seed $RANDOM \
                --learning_rate ${lr} \
                --wandb_project_name ${project_name} \
                --wandb_run_name ${run_name} \
                --num_train_epochs 3 \
                --logging_steps $logging_steps  \
                --save_steps 0 \
                --per_gpu_train_batch_size 16 \
                --per_gpu_eval_batch_size 16  \
                --writer_dir ${writer_dir}    
    done 
done

