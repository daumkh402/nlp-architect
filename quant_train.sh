export CUDA_VISIBLE_DEVICES=2

for i in 1 
do
nlp-train transformer_glue \
       --task_name mrpc \
       --model_name_or_path bert-base-uncased \
       --model_type quant_bert \
       --output_dir ../nlp_arch_results/mrpc-8bit  \
       --evaluate_during_training \
       --data_dir ../glue_data/MRPC \
       --do_lower_case \
       --overwrite_output_dir  

done

