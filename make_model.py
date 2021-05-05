import transformers
import pdb
import torch
import torch.nn as nn
import os
import argparse


TASK = ['sst-2', 'cola', 'rte', 'sts-b', 'mrpc']

parser = argparse.ArgumentParser()
parser.add_argument(
    "--make",
    action = "store_true"
)

parser.add_argument(
    "--data",
    required=True,
    type=str,
    choices=TASK
)

parser.add_argument(
    "--model",
    required=True,
    type=str,
    choices=TASK
)

args = parser.parse_args()

dirs = {'sst-2'   : "/home/imza/ssd/nlp_arch_results/0425_FP_barPlot/sst-2/lr_2e-5/1/best_dev",
        'cola'  : "/home/imza/ssd/nlp_arch_results/0425_FP_barPlot/cola/lr_2e-5/1/best_dev",
        'rte'   : "/home/imza/ssd/nlp_arch_results/0425_FP_barPlot/rte/lr_3e-5/1/best_dev",
        'sts-b' : "/home/imza/ssd/nlp_arch_results/0425_FP_barPlot/sts-b/lr_4e-5/1/best_dev",
        'mrpc'  : "/home/imza/ssd/nlp_arch_results/0425_FP_barPlot/mrpc/lr_3e-5/1/best_dev" }

task=args.model
data=args.data
bert_dir = dirs[task]
classifier_dir = dirs[data]
bert = transformers.BertForSequenceClassification.from_pretrained(bert_dir)
classifier = transformers.BertForSequenceClassification.from_pretrained(classifier_dir)
#pdb.set_trace()

if args.make:

    save_dir = "../ssd/nlp_arch_results/new_models/model_" + task + "_data_" + data

    bert_state_dict = bert.state_dict()

    new_model_name = 'pytorch_model.bin'
    new_model_ = {}

    # bert._modules['classifier'] = classifier._modules['classifier']

    for n,p in bert.named_parameters():
        if 'classifier' not in n:
            new_model_[n] = p

    for n,p in classifier.named_parameters():
        if 'classifier' in n:
            new_model_[n] = p

    bert_state_dict.update(new_model_)
    # bert.load_state_dict(bert_state_dict, strict=True)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    torch.save(bert_state_dict, os.path.join(save_dir,new_model_name))      ##Have to load state_dict

else:
    d = "/home/imza/ssd/nlp_arch_results/new_models/model_" + task + "_data_" + data
    made_bert = transformers.BertForSequenceClassification.from_pretrained(d)
    made_bert_state_dict = made_bert.state_dict()
    # pdb.set_trace()
    for n,p in bert.named_parameters():

        if not torch.all(torch.eq(made_bert_state_dict[n],p)):
            print("%s is not same in bert and made_bert" %n)


    for n,p in classifier.named_parameters():
        if torch.all(torch.eq(made_bert_state_dict[n],p)):
            print("%s is same in made_bert and classifier" %n)         

