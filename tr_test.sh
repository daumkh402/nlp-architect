export CUDA_VISIBLE_DEVICES=3
project_name=testtest
task="rte"
for i in 1
do   
       # based on batch size 32 
       case $task in 
       cola) data="CoLA"; logging_steps=40;;	       			
       mrpc) data="MRPC"; logging_steps=20;; 
       sts-b) data="STS-B"; logging_steps=40;;
       mnli) data="MNLI"; logging_steps=2400;;
       rte) data="RTE"; logging_steps=15;;
       wnli) data="WNLI"; logging_steps=8;;
       sst-2) data="SST-2"; logging_steps=400;; 
       qqp) data="QQP"; logging_steps=2200;; 
       qnli) data="QNLI"; logging_steps=600;;
       esac

       logging_steps=1;
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
       
       for q in "True False False" "True True False" "True True True"
       do
       qc=($q)
       run_name="${task}_${i}_loggingstep_${logging_steps}"
       nlp-train transformer_glue \
              --task_name ${task} \
              --model_name_or_path ../Qbert_model\
              --model_type quant_bert \
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
              --per_gpu_train_batch_size 32 \
              --per_gpu_eval_batch_size 32 \
              --qcomp "{'q_Vout' : ${qc[0]}, 'q_COM2': ${qc[1]}, 'q_COM3': ${qc[2]}}" \
	       --wandb_off
             #--dump_distributions \
              #--wandb_off
       done

done 
