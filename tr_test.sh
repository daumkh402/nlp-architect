export CUDA_VISIBLE_DEVICES=2
project_name=FPbert
task="sts-b"
for i in 1
do
       logging_steps=50;
       case $task in 
          cola) data="CoLA";; mrpc) data="MRPC";; sts-b) data="STS-B";;
          mnli) data="MNLI";;  rte) data="RTE";; wnli) data="WNLI";;
          sst-2) data="SST-2"; logging_steps=50;; qqp) data="QQP"; logging_steps=50;; qnli) data="QNLI"; logging_steps=50;;
       esac

       seed=$((i*1000))
       result_dir="../nlp_arch_results/${project_name}/test/"
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
              --wandb_run_name "${task}_test" \
              --wandb_off \
              --num_train_epochs 1 \
              --logging_steps $logging_steps \
              --per_gpu_train_batch_size 8 \
              --per_gpu_eval_batch_size 8 \
              --warmup_steps 50 \ 
              --save_steps 0   
done 
