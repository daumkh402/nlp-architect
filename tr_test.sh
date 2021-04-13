export CUDA_VISIBLE_DEVICES=0
project_name=test
task="rte"
for i in 1
do
       
       case $task in 
              cola) data="CoLA"; logging_steps=100;; 
              mrpc) data="MRPC"; logging_steps=45;; 
              sts-b) data="STS-B"; logging_steps=70;;
              mnli) data="MNLI"; logging_steps=4900;;  
              rte) data="RTE"; logging_steps=30;; 
              wnli) data="WNLI"; logging_steps=8;;
              sst-2) data="SST-2"; logging_steps=840;; 
              qqp) data="QQP"; logging_steps=4500;; 
              qnli) data="QNLI"; logging_steps=1300;;
       esac

       logging_steps=300;
       run_name="${task}_${i}_lr_${lr}_loggingstep_${logging_steps}"
       writer_dir="../tensorboard/${project_name}/${run_name}"


       result_dir="../nlp_arch_results/${project_name}/test"
       if [ ! -d ${result_dir} ]
              then
              echo "${result_dir} does not exist"
              mkdir -p ${result_dir}
       fi

       # h=0
       # writer_dir="${result_dir}/tensorboard/${h}"
       # while [ -d ${writer_dir} ]
       # do
       #               h=$((h+1))
       #               writer_dir="${result_dir}/tensorboard/${h}"
       # done

       run_name="${task}_${i}_loggingstep_${logging_steps}"
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
              --num_train_epochs 1 \
              --logging_steps $logging_steps \
              --wandb_project_name ${project_name} \
              --wandb_run_name ${run_name} \
              --writer_dir $writer_dir \
              --warmup_steps 0 \
              --save_steps 0 \
              --per_gpu_train_batch_size 8 \
              --per_gpu_eval_batch_size 8 \
              --dump_distributions 

done 
