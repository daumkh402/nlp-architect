export CUDA_VISIBLE_DEVICES=3



project_name=FPbert
task="mrpc"
for i in 1 2 3
do
       seed=$((i*1000))
       result_dir="../nlp_arch_results/${project_name}/${i}"
       if [ ! -d ${result_dir} ]
              then
              echo "${result_dir} does not exist"
              mkdir -p ${result_dir}
       fi
       logname="${result_dir}/${project_name}_${task}_${i}.txt"
 
       nlp-train transformer_glue \
              --task_name ${task} \
              --model_name_or_path bert-base-uncased \
              --model_type bert \
              --output_dir ${result_dir} \
              --evaluate_during_training \
              --data_dir ../glue_data/MRPC \
              --do_lower_case \
              --overwrite_output_dir \
              --seed ${seed} \
              --wandb_project_name ${project_name} \
              --num_train_epochs 3 \
              --logging_steps 25 \
      
done 
