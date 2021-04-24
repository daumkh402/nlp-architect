export CUDA_VISIBLE_DEVICES=2

project_name=0425_FP_barPlot
# "cola" "mrpc" "qnli" "rte" "sts-b" "sst-2" "wnli" "mnli" 
for task in  "cola" "mrpc" "rte" "sts-b"
do
       case $task in 
       cola) data="CoLA"; lr=2e-5; logging_steps=10;;	 #40      			
       mrpc) data="MRPC"; lr=3e-5; logging_steps=5;;     #20
       sts-b) data="STS-B"; lr=4e-5; logging_steps=10;;   #40
       rte) data="RTE"; lr=3e-5; logging_steps=3;;       #15
       sst-2) data="SST-2"; lr=2e-5; logging_steps=100;;  #400
       qqp) data="QQP"; lr=3e-5; logging_steps=550;;     #2200
       qnli) data="QNLI"; lr=2e-5; logging_steps=150;;    #600
       mnli) data="MNLI"; lr=2e-5; logging_steps=480;;   #2400
       wnli) data="WNLI"; lr=2e-5; logging_steps=8;;      #8  
       esac

    for i in 1
    do  
        run_name="${task}_${i}_lr_${lr}" 
        h=0
        writer_dir="../../tensorboard/${project_name}/${run_name}_${h}"
        while [ -d ${writer_dir} ]
        do
            h=$((h+1))
            writer_dir="../../tensorboard/${project_name}/${run_name}_${h}"
        done

        result_dir="../../nlp_arch_results/${project_name}/${task}/lr_${lr}/${i}"
        if [ ! -d ${result_dir} ]
        then
            echo "${result_dir} does not exist"
            mkdir -p ${result_dir}
        fi
        nlp-train transformer_glue \
                --task_name ${task} \
                --model_name_or_path ../../bert_model \
                --model_type bert \
                --output_dir ${result_dir} \
                --evaluate_during_training \
                --data_dir ../../glue_data/${data} \
                --do_lower_case \
                --overwrite_output_dir \
                --seed $RANDOM \
                --wandb_project_name ${project_name} \
                --wandb_run_name ${run_name} \
                --num_train_epochs 3 \
                --logging_steps $logging_steps  \
                --save_steps 0 \
                --per_gpu_train_batch_size 32 \
                --per_gpu_eval_batch_size 32  \
                --writer_dir $writer_dir    
    done 
done

