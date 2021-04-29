export CUDA_VISIBLE_DEVICES=0

#project_name=0427_FPBERT_evalscore
project_name=Test
# "cola" "mrpc" "qnli" "rte" "sts-b" "sst-2" "wnli" "mnli" 
for task in    "rte" #"mrpc" "cola" "sts-b" 
do
       case $task in 
       cola) data="CoLA"; lr=3e-5; logging_steps=80; bsz=4;;	    #10      			
       mrpc) data="MRPC"; lr=2e-5; logging_steps=40; bsz=4;;         #5
       sts-b) data="STS-B"; lr=4e-5; logging_steps=10; bsz=32;;      #10
       rte) data="RTE"; lr=2e-5; logging_steps=24;  bsz=4;;                  #3
       sst-2) data="SST-2"; lr=2e-5; logging_steps=100; bsz=32;;          #100
       qqp) data="QQP"; lr=3e-5; logging_steps=550; bsz=32;;                 #550
       qnli) data="QNLI"; lr=2e-5; logging_steps=150; bsz=32;;               #150
       mnli) data="MNLI"; lr=2e-5; logging_steps=480; bsz=32;;              #480
       wnli) data="WNLI"; lr=2e-5; logging_steps=2; bsz=32;;                #82  
       esac
    logging_steps=1
    for i in 1
    do  
        run_name="${task}_${i}_${lr}" 
        h=0
        writer_dir="../../tensorboard/${project_name}/${run_name}_${h}"
        while [ -d ${writer_dir} ]
        do
            h=$((h+1))
            writer_dir="../../tensorboard/${project_name}/${run_name}_${h}"
        done

        result_dir="../../nlp_arch_results/${project_name}/${task}/${run_name}"
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
                --num_train_epochs 1 \
                --logging_steps $logging_steps  \
                --save_steps 0 \
                --per_gpu_train_batch_size $bsz \
                --per_gpu_eval_batch_size 8  \
                --writer_dir $writer_dir 
                # --freeze_bert


    done 
done

