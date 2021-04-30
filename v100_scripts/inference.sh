nlp-inference run transformer_glue \
    --model_path ../../bert_model \
    --task_name mrpc \
    --model_type quant_bert \
    --output_dir ../../ssd/nlp_arch_results \
    --data_dir ../../ssd/glue_data/MRPC
    --do_lower_case \
    --overwrite_output_dir
