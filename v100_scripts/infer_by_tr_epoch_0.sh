export CUDA_VISIBLE_DEVICES=2

# task : "cola" "mrpc" "qnli" "rte" "sts-b" "sst-2" "wnli" "mnli"
# data : "CoLA" "MRPC" "QNLI" "RTE" "STS-B" "SST-2" "WNLI" "MNLI"

model="cola"
task="sts-b"
data="STS-B"
result_dir="../../ssd/nlp_arch_results/infer_by_train_epoch0/model_${model}_input_${task}"
pretrained_model="/home/imza/ssd/nlp_arch_results/new_models/model_${model}_data_${task}"

run_name="${task}_${i}_lr_${lr}" 

if [ ! -d ${result_dir} ]
then
    echo "${result_dir} does not exist"
    mkdir -p ${result_dir}
fi
nlp-train transformer_glue \
        --task_name ${task} \
        --model_name_or_path $pretrained_model \
        --model_type bert \
        --output_dir ${result_dir} \
        --evaluate_during_training \
        --data_dir ../../ssd/glue_data/${data} \
        --do_lower_case \
        --overwrite_output_dir \
        --seed $RANDOM \
        --wandb_project_name '' \
        --wandb_run_name '' \
        --num_train_epochs 0 \
        --save_steps 0 \
        --per_gpu_eval_batch_size 16  \
        --wandb_off   
 

