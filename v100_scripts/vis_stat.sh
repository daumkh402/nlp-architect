export CUDA_VISIBLE_DEVICES=0,1

project_name=FPbert_vis_attentions
# "cola" "mrpc" "qnli" "rte" "sts-b" "sst-2" "qqp"              "wnli" "mnli" 
for task in  "qqp" #"cola" "mrpc" "rte" "sts-b" "sst-2" "qqp"  #"sts-b" 
do
    logging_steps=25;
    case $task in 
       cola) data="CoLA"; lr=2e-5; logging_steps=40;;	 #40      			
       mrpc) data="MRPC"; lr=3e-5; logging_steps=20;;     #20
       sts-b) data="STS-B"; lr=4e-5; logging_steps=40;;   #40
       rte) data="RTE"; lr=3e-5; logging_steps=15;;       #15
       sst-2) data="SST-2"; lr=2e-5; logging_steps=400;;  #400
       qqp) data="QQP"; lr=3e-5; logging_steps=2200;;     #2200
       qnli) data="QNLI"; lr=2e-5; logging_steps=600;;    #600
       mnli) data="MNLI"; lr=2e-5; logging_steps=2400;;   #2400
       wnli) data="WNLI"; lr=2e-5; logging_steps=8;;      #8      
    esac

    run_name="${task}_${i}_lr_${lr}_loggingstep_${logging_steps}" 
    h=0
    writer_dir="../ssd/tensorboard/${project_name}/${run_name}_${h}"
    while [ -d ${writer_dir} ]
    do
        h=$((h+1))
        writer_dir="../ssd/tensorboard/${project_name}/${run_name}_${h}"
    done

    for i in 1
    do  
	result_dir="../ssd/nlp_arch_results/${project_name}/${task}/lr_${lr}/${i}"
	if [ ! -d ${result_dir} ]
	then
           echo "${result_dir} does not exist"
           mkdir -p ${result_dir}
	fi
        
        nlp-train transformer_glue \
                --task_name ${task} \
                --model_name_or_path ../bert_model \
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
                --writer_dir ${writer_dir} \
                --dump_distributions	
    done 
done

