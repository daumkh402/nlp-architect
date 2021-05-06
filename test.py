import transformers
import torch
import torch.nn as nn
from  nlp_architect.models.transformers.quantized_bert import QuantizedBertForSequenceClassification as Qbert

dirs = "/home/hs402/nlp_arch_results/0506_Qcomp/mrpc/mrpc_1_lr_2e-5_qc_FalseFalseFalseTrueFalse/best_dev"
bert = Qbert.from_pretrained(dirs)



if __name__ == '__main__':

    for n, m in bert.named_modules():
        if 'value' in n:
            print('name : {}         quant_output: {}'.format(n,m.requantize_output))
            


