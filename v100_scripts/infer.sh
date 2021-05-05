export CUDA_VISIBLE_DEVICES=3
# for i in 1
# do   
#        # based on batch size 32 
#        # case $task in 
#        # cola) data="CoLA"; logging_steps=40;;	       			
#        # mrpc) data="MRPC"; logging_steps=20;; 
#        # sts-b) data="STS-B"; logging_steps=40;;
#        # mnli) data="MNLI"; logging_steps=2400;;
#        # rte) data="RTE"; logging_steps=15;;
#        # wnli) data="WNLI"; logging_steps=8;;
#        # sst-2) data="SST-2"; logging_steps=400;; 
#        # qqp) data="QQP"; logging_steps=2200;; 
#        # qnli) data="QNLI"; logging_steps=600;;
#        # esac

#model="mrpc"
#task="sst-2"
#data="SST-2"

for model in "mrpc" "cola" "rte" "sts-b" "sst-2"
do
for task in "mrpc" "cola" "rte" "sts-b" "sst-2"
do
case $task in
mrpc) data="MRPC";;
cola) data="CoLA";;
rte) data="RTE";;
sts-b) data="STS-B";;
sst-2) data="SST-2";;
esac

result_dir="../../ssd/nlp_arch_results/inference/model_${model}_input_${data}"
pretrained_model="/home/imza/ssd/nlp_arch_results/new_models/model_${model}_data_${task}"
if [ ! -d ${result_dir} ]
       then
       echo "${result_dir} does not exist"
       mkdir -p ${result_dir}
fi

nlp-inference transformer_glue \
       --task_name $task \
       --model_path $pretrained_model \
       --data_dir ../../ssd/glue_data/${data} \
       --model_type quant_bert \
       --output_dir ${result_dir} \
       --do_lower_case \
       --per_gpu_eval_batch_size 16 \
       --wandb_project_name '' \
       --wandb_run_name '' \
       --wandb_off \
       --evaluate \
       --overwrite_output_dir

done
done


