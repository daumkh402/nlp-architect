export CUDA_VISIBLE_DEVICES=0

project_name=0429_Qcomp
# "cola" "mrpc" "qnli" "rte" "sts-b" "sst-2" "qqp" "wnli" "mnli" 
for task in  "mrpc" "cola" "rte"
do
    bsz=32
    case $task in 
       cola) data="CoLA"; lr=5e-5; logging_steps=40; bsz=4;;	 #40      			
       mrpc) data="MRPC"; lr=3e-5; logging_steps=20; bsz=4;;     #20
       sts-b) data="STS-B"; lr=4e-5; logging_steps=40;;          #40
       rte) data="RTE"; lr=3e-5; logging_steps=15; bsz=4;;       #15
       sst-2) data="SST-2"; lr=2e-5; logging_steps=400;;         #400
       qqp) data="QQP"; lr=3e-5; logging_steps=2200;;            #2200
       qnli) data="QNLI"; lr=2e-5; logging_steps=600;;           #600
       mnli) data="MNLI"; lr=2e-5; logging_steps=2400;;          #2400
       wnli) data="WNLI"; lr=2e-5; logging_steps=8;;             #8      
    esac


    for q in "True False False True False " "False True False True False" "False False True True False" "False False False True True" 
    do
        qc=($q)
        for i in 1 2 3
        do  
            run_name="${task}_${i}_lr_${lr}_qc_${qc[0]}${qc[1]}${qc[2]}${qc[3]}${qc[4]}" 
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
                    --model_name_or_path ../../Qbert_model \
                    --model_type quant_bert \
                    --output_dir ${result_dir} \
                    --evaluate_during_training \
                    --data_dir ../../glue_data/${data} \
                    --do_lower_case \
                    --overwrite_output_dir \
                    --seed $RANDOM \
                    --learning_rate ${lr} \
                    --wandb_project_name ${project_name} \
                    --wandb_run_name ${run_name} \
                    --num_train_epochs 3 \
                    --logging_steps $logging_steps  \
                    --save_steps 0 \
                    --per_gpu_train_batch_size ${bsz} \
                    --per_gpu_eval_batch_size 32  \
                    --writer_dir ${writer_dir} \
                    --qcomp "{'q_Vout' : ${qc[0]}, 'q_COM2': ${qc[1]}, 'q_COM3': ${qc[2]}, 'q_COM4': ${qc[3]}, 'q_COM5': ${qc[4]}}" 
                   # --dump_distributions
        done
    done  
done

