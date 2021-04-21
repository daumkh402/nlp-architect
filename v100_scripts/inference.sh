nlp-inference run transformer_glue \
    --model_path ../nlp_arch_results/mrpc-8bit \
    --task_name mrpc \
    --model_type quant_bert \
    --output_dir ../nlp_arch_results \
    --data_dir ../glue_data/MRPC
    --do_lower_case \
    --overwrite_output_dir
