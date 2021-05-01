import transformers
import pdb
import torch
import torch.nn as nn
import os

make = True
if make:
    dirs = {'sst-2'   : "/home/imza/ssd/nlp_arch_results/0425_FP_barPlot/sst-2/lr_2e-5/1/best_dev",
            'cola'  : "/home/imza/ssd/nlp_arch_results/0425_FP_barPlot/cola/lr_2e-5/1/best_dev",
            'rte'   : "/home/imza/ssd/nlp_arch_results/0425_FP_barPlot/rtelr_3e-5/1/best_dev",
            'sts-b' : "/home/imza/ssd/nlp_arch_results/0425_FP_barPlot/sts-b/lr_4e-5/1/best_dev",
            'mrpc'  : "/home/imza/ssd/nlp_arch_results/0425_FP_barPlot/mrpc/lr_3e-5/1/best_dev" }

    task='sst-2'
    data='cola'
    bert_dir = dirs[task]
    classfier_dir = dirs[data]
    save_dir = "../ssd/nlp_arch_results/new_models/model_" + task + "_data_" + data

    bert = transformers.BertModel.from_pretrained(bert_dir)
    classfier = transformers.BertModel.from_pretrained(classfier_dir)
    bert_state_dict = bert.state_dict()
    # pdb.set_trace()

    new_model_name = 'pytorch_model.bin'
    new_model_ = {}

    for n,p in bert.named_parameters():
        if 'classifier' not in n:
            new_model_[n] = p

    for n,p in classfier.named_parameters():
        if 'classifier' in n:
            new_model_[n] = p

    bert_state_dict.update(new_model_)
    bert.load_state_dict(bert_state_dict, strict=True)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    torch.save(bert_state_dict, os.path.join(save_dir,new_model_name))      ##Have to load state_dict

else:
    pass