# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
import io
import logging
import os
from typing import List, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    BertConfig,
    BertTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    XLMConfig,
    XLMTokenizer,
    XLNetConfig,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)

from nlp_architect.models import TrainableModel
from nlp_architect.models.transformers.quantized_bert import QuantizedBertConfig

import pdb
import wandb
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import seaborn
import numpy as np
import pickle
logger = logging.getLogger(__name__)


ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, XLNetConfig, XLMConfig)
    ),
    (),
)


def get_models(models: List[str]):
    if models is not None:
        return [m for m in ALL_MODELS if m.split("-")[0] in models]
    return ALL_MODELS


Record=True
HM_MODE = ['l_to_h', 'seq_to_seq']
class Recorder():

    def __init__(self,
                 tokenizer = None, 
                 wandb_project_name=None,
                 wandb_run_name=None,
                 wandb_off=False,
                 writer_dir=None,
                 dump_distributions=False,
                 model_type=None):

        self.hook_list = []
        self.step_count = 0
        self.model = None
        self.config=None
        self.tokenizer = tokenizer     
        self.hm_mode = HM_MODE[0]

        self.wandb_off = wandb_off   
        self.model_type=model_type
        self.wandb_project_name = wandb_project_name
        self.wandb_run_name = wandb_run_name
        if not self.wandb_off:
            self.WANDB = wandb
            self.WANDB.init(name=wandb_run_name, project=wandb_project_name, dir = '../../') 
            self.WANDB.tensorboard.patch(tensorboardX=False, pytorch=True)

        self.writer = SummaryWriter(writer_dir) if writer_dir is not None else None
        self.l_to_h_score = None
        self.input_sequence = None

        #flag for choosing data to gather 
        self.per_batch_heatmap = False
        self.need_com3_hist = True and self.model_type == 'bert'
        self.need_input_hook = False
        self.need_param_stat = False
        self.need_attention_output = False
        self.need_distributions = dump_distributions

        if self.need_param_stat:
            self.stat_groups = ['Embedding', 'Attention', 'FeedForward', 'Pooler', 'Classifier']
            self.stat_keys = ['name','max','min','mean','std','skew']
            self.pstat_by_layer = {}  #calculate statistics from weights stored in self.param_to_save.
            self.param_to_save = {} #store weights of layers by different groups

        if self.need_com3_hist:
            self.com3_in_temp = {}
            self.com3_out_temp = {}

    def WANDB_log(self, tgt, **kwargs):
        if self.wandb_off:
            return
        else:
            self.WANDB.log(tgt, **kwargs)

    def register(self, model, config, dump_interval):

        self.model = model
        self.config = config

        self.rec_p()

        for name, layer in model.named_modules():

            if name == 'bert':
                if self.need_input_hook:
                    handle = layer.register_forward_pre_hook(self.Input_hook(name, dump_interval))
                    self.hook_list.append(handle)

            if self.need_param_stat:
                if 'dense' in name or 'key' in name or 'query' in name or 'value' in name or '_embeddings' in name:   
                    handle = layer.register_forward_hook(self.QLayer_hook(name, dump_interval))
                    self.hook_list.append(handle)

            if name == 'bert.encoder' and self.need_attention_output:   
                self.config.output_attentions = True
                handle = layer.register_forward_hook(self.encoder_hook(name, dump_interval))
                self.hook_list.append(handle)   
            
            if self.need_com3_hist:
                if name.split('.')[-1] == 'self':
                    layer.need_com3_hist = True
                    n = 'L' + name.split('.')[3]+'_com3'  
                    handle = layer.register_forward_hook(self.com3_hook(n, dump_interval))          #check variation of attention score distribution 
                    self.hook_list.append(handle) 

    def convert_tensor_to_string(self, input):    
        input_strings = []
        for sequence in input:
            seq = []
            for word_id in sequence:   
                if word_id == 0:
                    break          
                seq.append(self.tokenizer.ids_to_tokens[word_id.item()])
            input_strings.append(seq)
        
        return input_strings


        def draw(self,data, x, y, ax):
            seaborn.heatmap(data, 
                            xticklabels=x, square=True, yticklabels=y,  
                            cbar=True, ax=ax)

    def cal_pstats(self, param):
        p_mean = torch.mean(param).detach()
        p_std = torch.std(param).detach().cpu()
        p_min = torch.min(param).detach().cpu().data.numpy()
        p_max = torch.max(param).detach().cpu().data.numpy()
             
        p_diffs = (param - p_mean).detach().cpu()
        p_zscores = p_diffs / p_std 
        p_skews = torch.mean(torch.pow(p_zscores,3.0)).cpu().data.numpy()


        p_mean = p_mean.cpu().data.numpy()
        p_std = p_std.cpu().data.numpy()

        # pdb.set_trace()
        
        return p_max, p_min, p_mean, p_std, p_skews
    
    def rec_p(self):
        if self.need_param_stat:
            for g in self.stat_groups:
                self.param_to_save[g] = []
                self.pstat_by_layer[g] = {}
                for k in self.stat_keys:
                    self.pstat_by_layer[g][k] = []

            for n , l in self.model.named_modules(): 
                if hasattr(l,'weight') and 'LayerNorm' not in n:
                    name, group = self.parse_lname(n) 
                    # if name == None:
                    #     pdb.set_trace()   
                    self.pstat_by_layer[group]['name'].append(name)
                    self.param_to_save[group].append(l.weight)

    def weights_to_stats(self):
        if self.need_param_stat:
            for g, params in self.param_to_save.items():
                for param in params:
                    pmax, pmin, pmean, pstd, pskew = self.cal_pstats(param)
                    self.pstat_by_layer[g]['max'].append(pmax)
                    self.pstat_by_layer[g]['min'].append(pmin)
                    self.pstat_by_layer[g]['mean'].append(pmean)
                    self.pstat_by_layer[g]['std'].append(pstd)
                    self.pstat_by_layer[g]['skew'].append(pskew)
        else:
            return

    def save_pstat(self):
        from datetime import date
        today = (date.today()).strftime("%Y%m%d")

        save_dir = os.path.join('../../nlp_arch_results/wstat/', self.wandb_project_name, today)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_file = os.path.join(save_dir, self.wandb_run_name + '.pickle')
        if self.need_param_stat or self.need_com3_hist:
            with open( save_file, 'wb') as f:
                if self.need_param_stat:
                    pickle.dump(self.pstat_by_layer, f)
                if self.need_com3_hist:
                    pickle.dump({'com3_in' : self.com3_in_temp},f)
                    pickle.dump({'com3_out' : self.com3_out_temp},f)


    def pstat_to_tensorboard(self):
        if self.need_param_stat:
            for group, gdata in self.pstat_by_layer.items():    # Embedding, Attention, FeedForward
                for k,v in gdata.items():                       #name, max, min, mean, std, skew
                    if k == 'name':
                        name = v
                        # pdb.set_trace()
                    else:   
                        if self.WANDB: 
                            data = [[x,y.item()] for (x,y) in zip(name,v)]
                            table = self.WANDB.Table(data=data, columns=["layer", "value"])
                            hist_title = group + '/' + k                      
                            fields = {"label" : "layer",
                                    "value" : "value"}
                            custom_chart = self.WANDB.plot_table(vega_spec_name="maruebaleitni/weight",
                                                                data_table=table, 
                                                                fields = fields,
                                                                string_fields={'title' : hist_title})
                            self.WANDB.log({hist_title : custom_chart})
                            # if group == 'Embedding' and k == 'mean':
                            #     pdb.set_trace()
                            # hist = self.WANDB.plot.bar(table, "layer", "value", title="test")
                            # self.WANDB.log({hist_title : hist})

    def parse_lname(self, lname):
        parsed = ''
        parsed = lname.split('.')
        parsed = parsed[:-1] if 'weight' in parsed else parsed

        name = None
        group = None
        if '_embeddings' in lname:   
            group = 'Embedding'
            name = parsed[2].split('_')[0] #word, position, token

        if 'encoder' in lname:
            lnum = parsed[3]
            if lnum == '10':
                lnum = 'X0'
            if lnum == '11':
                lnum = 'X1' 

            if parsed[-1] in ['query', 'key', 'value']:
                group = 'Attention'
                ltype = '_' + parsed[-1]
                name = 'L' + lnum + ltype

            elif 'attention.output.dense' in lname:
                group = 'Attention'
                name = 'L' + lnum + '_AO'

            elif 'intermediate.dense' in lname: 
                group = 'FeedForward'
                name = 'L' + lnum + '_FFN0'

            elif ('output.dense' in lname and 'attention' not in lname):
                group = 'FeedForward'
                name = 'L' + lnum + '_FFN1'
            
            else:
                pass

        elif 'pooler.dense' in lname:
            group = 'Pooler'
            name = 'pooler' 
        
        elif 'classifier' == lname:
            group = 'Classifier'
            name = 'Classifier'

        else:
            pass

        return name, group

    def Input_hook(self, layer_name, dump_interval):
        def hook(module, input):
            if not self.model.training :
                return

            if self.need_distributions and self.per_batch_heatmap:
                if (self.step_count+1) % dump_interval == 0:                   
                    inp = input[0] if isinstance(input, tuple) else input
                    self.input_sequence = self.convert_tensor_to_string(inp) 
        return hook

    def QLayer_hook(self, layer_name, dump_interval):    
        def hook(module, input, output):

            if not self.model.training :
                return
            
            wmax, wmin, wmean, wstd, wskews = self.cal_pstats(module.weight)
            # if 'pooler' in layer_name:
            #     pdb.set_trace()

            name, _ = self.parse_lname(layer_name)

            self.writer.add_scalar(name + '_stats/w_mean', wmean, self.step_count)
            self.writer.add_scalar(name +'_stats/w_std', wstd, self.step_count)
            self.writer.add_scalar(name +'_stats/w_max', wmax, self.step_count)
            self.writer.add_scalar(name +'_stats/w_min', wmin, self.step_count)
        
            if self.model_type == 'quant_bert':
                self.writer.add_scalar(name + '_stats/w_scale', module.weight_scale.clone().detach().cpu().data.numpy(), self.step_count)
                if 'embeddings' not in layer_name: # for linear layer
                    in_thresh = module.input_thresh.clone().detach().cpu().data.numpy()
                    self.writer.add_scalar(name + '_stats/i_thresh', in_thresh, self.step_count)

                    if hasattr(module, 'output_thresh'):
                        out_thresh= module.output_thresh.clone().detach().cpu().data.numpy()
                        self.writer.add_scalar(name + '_stats/o_thresh', out_thresh, self.step_count) 
                       
            if self.need_distributions:
                if (self.step_count+1) % dump_interval == 0:                 
                    self.writer.add_histogram(name + '/w_hist', 
                                            module.weight.clone().detach().cpu().data.numpy(), 
                                            self.step_count) 

                    if 'embeddings' not in layer_name: # for linear layer

                        inp = input[0] if isinstance(input, tuple) else input
                        out = output[0] if isinstance(output, tuple) else output

                        self.writer.add_histogram(name + '/o_hist', 
                                                output.clone().detach().cpu().data.numpy(), 
                                                self.step_count)

                        self.writer.add_histogram(name + '/i_hist', 
                                                inp.clone().detach().cpu().data.numpy(), 
                                                self.step_count) 

                                                
        return hook
    
    def com3_hook(self,name, dump_interval):
        def hook(module, input, output):
            if not self.model.training :
                return

            if self.need_com3_hist:
                if (self.step_count+1) % dump_interval == 0:
                    self.writer.add_histogram(name + '/' + 'in', module.com3_in.clone().detach().view(-1), self.step_count)
                    self.writer.add_histogram(name + '/' + 'out', module.com3_out.clone().detach().view(-1), self.step_count)
                    self.com3_in_temp[name] = module.com3_in
                    self.com3_out_temp[name] = module.com3_out
            else:
                pass

        return hook

    def encoder_hook(self, layer_name, dump_interval):
        def hook(module, input, output) :
            if not self.model.training :
                return

            # pdb.set_trace() 
            if self.need_distributions:
                if self.config.output_hidden_states:
                    attention_weights = output[2]
                else:
                    attention_weights = output[1]           # tuple of length #num_layer. num_layers x (bsz x num_heads x max_seq x max_seq)  

                for i,layer in enumerate(attention_weights):
                    if i == 0:
                        attentions = layer.unsqueeze(dim=0)
                    else:                       
                        attentions = torch.cat((attentions, layer.unsqueeze(dim=0)), dim = 0)

                attentions = attentions.clone().detach()  
                if self.hm_mode == 'l_to_h':
                    attentions = torch.max(attentions,dim=-1).values  # num_layers x bsz x num_heads x max_seq
                    # attentions = attentions.permute(0,2,1,3)          # #l x #h x bsz x max_seq
                    # pdb.set_trace()      
                    attentions = attentions.permute(1,0,2,3)            # bsz x #l x #h x max_seq
                    attentions = attentions.mean(dim=-1)                # bsz x #l x #h

                if self.hm_mode == 'seq_to_seq':
                    pass


                if self.per_batch_heatmap:
                    if (self.step_count+1) % dump_interval == 0:
                        for i, b in enumerate(attentions):
                            # pdb.set_trace()
                            seq = ' '.join(self.input_sequence[i])
                            self.l_to_h_score = b
                            self.l_to_h_heatmap('batch_' + str(self.step_count) + '/sequence/' + seq)

                

                # # new_shape = attentions.size()
                # # new_shape = (new_shape[0],new_shape[1],-1)
                # # attentions = attentions.reshape(new_shape)        # num_layers x num_heads x (bsz * max_seq)
                # attentions = attentions.mean(dim=-1)              # num_layers x num_heads

                # pdb.set_trace()

                attentions = attentions.mean(dim=0) # #l x #h
                if self.l_to_h_score is None:
                    self.l_to_h_score = attentions
                else:
                    self.l_to_h_score = torch.mean(torch.stack((self.l_to_h_score, attentions),dim=0),dim=0)

 

                
            #     if (self.step_count+1) % dump_interval == 0:
                # attention weight  
                    # if self.model.config.output_attentions:       #self attention layer
                         
                #         # output[0] : context vector of size (bsz, max_seq_length, hidden_size)
                #         # output[1] : attention for all heads (bsz, num_heads, max_seq_length, max_seq_length)

                #         #for Testing 
                #         # for i in range(self.model.config.num_attention_heads)
                #         # self.writer.add_scalar(prefix + layer_name + '_head' + str(i), output[1][0][i])

                #         # heatmap = axes.pcolor(output[1][0][0].clone().cpu().detach(), cmap=plt.cm.Blues)
                #         # axes.set_xticklabels(self.input_sequence[0], minor=False)
                #         # target words -> row labels
                #         # axes.set_yticklabels(self.input_sequence[0], minor=False)

                #         plt.title('head')
                #         fig = plt.figure(figsize=(100,100))
                #         axes = fig.add_subplot(111)
                #         # plt.pcolor(output[1][0][0].clone().cpu().detach(), self.input_sequence[0], self.input_sequence[0])
                #         seq_len = len(self.input_sequence[0])
                #         plt.tight_layout()
                #         self.draw(data = output[1][0][0][:seq_len][:seq_len].clone().cpu().detach(), x=self.input_sequence[0], y=self.input_sequence[0], ax = axes)
                #         plt.savefig('test2.png')
                #         # pdb.set_trace()

        return hook

    def l_to_h_heatmap(self, heatmap_title = 'layer_to_head_avg_score'):
        if self.l_to_h_score is not None:
            fig = plt.figure()
            seaborn.heatmap(data=self.l_to_h_score.cpu().numpy(), linewidth=0.5, cbar=True)
            self.writer.add_figure(heatmap_title, fig)

    def remove(self):
        for handle in self.hook_list:
            handle.remove()
        if self.writer is not None:
            self.writer.close()


class TransformerBase(TrainableModel):
    """
    Transformers base model (for working with pytorch-transformers models)
    """

    MODEL_CONFIGURATIONS = {
        "bert": (BertConfig, BertTokenizer),
        "quant_bert": (QuantizedBertConfig, BertTokenizer),
        "xlnet": (XLNetConfig, XLNetTokenizer),
        "xlm": (XLMConfig, XLMTokenizer),
        "roberta": (RobertaConfig, RobertaTokenizer),
    }

    def __init__(
        self,
        model_type: str,
        model_name_or_path: str,
        labels: List[str] = None,
        num_labels: int = None,
        config_name=None,
        tokenizer_name=None,
        do_lower_case=False,
        output_path=None,
        device="cpu",
        n_gpus=0,
        wandb_project_name=None,
        wandb_run_name=None,
        wandb_off=False,
        writer_dir=None,
        dump_distributions=None,
        qcomp = None,
        freeze_bert=None):
        
    
        """
        Transformers base model (for working with pytorch-transformers models)

        Args:
            model_type (str): transformer model type
            model_name_or_path (str): model name or path to model
            labels (List[str], optional): list of labels. Defaults to None.
            num_labels (int, optional): number of labels. Defaults to None.
            config_name ([type], optional): configuration name. Defaults to None.
            tokenizer_name ([type], optional): tokenizer name. Defaults to None.
            do_lower_case (bool, optional): lower case input words. Defaults to False.
            output_path ([type], optional): model output path. Defaults to None.
            device (str, optional): backend device. Defaults to 'cpu'.
            n_gpus (int, optional): num of gpus. Defaults to 0.

        Raises:
            FileNotFoundError: [description]
        """
        assert model_type in self.MODEL_CONFIGURATIONS.keys(), "unsupported model_type"
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.labels = labels
        self.num_labels = num_labels
        self.do_lower_case = do_lower_case
        if output_path is not None and not os.path.exists(output_path):
            raise FileNotFoundError("output_path is not found")
        self.output_path = output_path

        self.model_class = None
        config_class, tokenizer_class = self.MODEL_CONFIGURATIONS[model_type]
        # pdb.set_trace()
        self.config_class = config_class
        self.tokenizer_class = tokenizer_class

        self.tokenizer_name = tokenizer_name
        self.tokenizer = self._load_tokenizer(self.tokenizer_name)
        self.config_name = config_name
        self.config = self._load_config(config_name)

        ##########################################################################
        # pdb.set_trace()
        if model_type == 'quant_bert' and qcomp is not None:
            self.config.attention_value["requantize_output"] = qcomp["q_Vout"] 
            self.config.quant_COM2 = qcomp["q_COM2"]
            self.config.quant_COM3 = qcomp["q_COM3"]

            # self.config.quant_COM4 = qcomp["q_COM4"]
            # self.config.attention_output["quant_input"] = not qcomp["q_COM4"]

            self.config.attention_output["quant_input"] = qcomp["q_COM4"]
            self.config.attention_output["requantize_output"] = qcomp["q_COM5"]
        ##########################################################################

        self.model = None
        self.device = device
        self.n_gpus = n_gpus

        self._optimizer = None
        self._scheduler = None
        self.training_args = None
        # pdb.set_trace()
        ##############################################################################

        if Record:
            self.recorder = Recorder(tokenizer=self.tokenizer, 
                                        wandb_project_name=wandb_project_name,
                                        wandb_run_name=wandb_run_name,
                                        wandb_off=wandb_off,
                                        writer_dir=writer_dir,
                                        dump_distributions=dump_distributions,                                
                                        model_type=model_type)

        self.freeze_bert = freeze_bert
        #############################################################################
    
    def freeze_pretrained(self):
        for name, param in self.model.named_parameters():
            if 'classifier' in name:
               pass
            else: 
                param.requires_grad = False
        
        #check 



    def to(self, device="cpu", n_gpus=0):
        if self.model is not None:
            self.model.to(device)
            if n_gpus > 1:
                self.model = torch.nn.DataParallel(self.model)
        self.device = device
        self.n_gpus = n_gpus

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt

    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(self, sch):
        self._scheduler = sch

    def setup_default_optimizer(
        self,
        weight_decay: float = 0.0,
        learning_rate: float = 5e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        total_steps: int = 0,):
    
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

    def _load_config(self, config_name=None):
        config = self.config_class.from_pretrained(
            config_name if config_name else self.model_name_or_path, num_labels=self.num_labels
        )
        return config

    def _load_tokenizer(self, tokenizer_name=None):
        tokenizer = self.tokenizer_class.from_pretrained(
            tokenizer_name if tokenizer_name else self.model_name_or_path,
            do_lower_case=self.do_lower_case,
        )
        return tokenizer

    def save_model(self, output_dir: str, save_checkpoint: bool = False, args=None):
        """
        Save model/tokenizer/arguments to given output directory

        Args:
            output_dir (str): path to output directory
            save_checkpoint (bool, optional): save as checkpoint. Defaults to False.
            args ([type], optional): arguments object to save. Defaults to None.
        """
        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger.info("Saving model checkpoint to %s", output_dir)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(output_dir)
        if not save_checkpoint:
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
            with io.open(output_dir + os.sep + "labels.txt", "w", encoding="utf-8") as fw:
                for label in self.labels:
                    fw.write("{}\n".format(label))
            if args is not None:
                torch.save(args, os.path.join(output_dir, "training_args.bin"))

    @classmethod
    def load_model(cls, model_path: str, model_type: str, *args, **kwargs):
        """
        Create a TranformerBase deom from given path

        Args:
            model_path (str): path to model
            model_type (str): model type

        Returns:
            TransformerBase: model
        """
        # Load a trained model and vocabulary from given path
        if not os.path.exists(model_path):
            raise FileNotFoundError
        with io.open(model_path + os.sep + "labels.txt") as fp:
            labels = [line.strip() for line in fp.readlines()]
        return cls(
            model_type=model_type, model_name_or_path=model_path, labels=labels, *args, **kwargs
        )

    @staticmethod
    def get_train_steps_epochs(
        max_steps: int, num_train_epochs: int, gradient_accumulation_steps: int, num_samples: int):
    
        """
        get train steps and epochs

        Args:
            max_steps (int): max steps
            num_train_epochs (int): num epochs
            gradient_accumulation_steps (int): gradient accumulation steps
            num_samples (int): number of samples

        Returns:
            Tuple: total steps, number of epochs
        """
        if max_steps > 0:
            t_total = max_steps
            num_train_epochs = max_steps // (num_samples // gradient_accumulation_steps) + 1
        else:
            t_total = num_samples // gradient_accumulation_steps * num_train_epochs
        return t_total, num_train_epochs

    def get_logits(self, batch):
        self.model.eval()
        inputs = self._batch_mapper(batch)
        outputs = self.model(**inputs)
        return outputs[-1]

    def _train(
        self,
        data_set: DataLoader,
        dev_data_set: Union[DataLoader, List[DataLoader]] = None,
        test_data_set: Union[DataLoader, List[DataLoader]] = None,
        gradient_accumulation_steps: int = 1,
        per_gpu_train_batch_size: int = 8,
        max_steps: int = -1,
        num_train_epochs: int = 3,
        max_grad_norm: float = 1.0,
        logging_steps: int = 50,    
        save_steps: int = 100,
        best_result_file: str = None,):
    
        """Run model training
        batch_mapper: a function that maps a batch into parameters that the model
                      expects in the forward method (for use with custom heads and models).
                      If None it will default to the basic models input structure.
        logging_callback_fn: a function that is called in each evaluation step
                      with the model as a parameter.

        """
        t_total, num_train_epochs = self.get_train_steps_epochs(
            max_steps, num_train_epochs, gradient_accumulation_steps, len(data_set)
        )
        if self.optimizer is None and self.scheduler is None:
            logger.info("Loading default optimizer and scheduler")
            self.setup_default_optimizer(total_steps=t_total)

        train_batch_size = per_gpu_train_batch_size * max(1, self.n_gpus)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(data_set.dataset))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per GPU/CPU = %d", per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            train_batch_size * gradient_accumulation_steps,
        )
        logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        best_dev = 0
        dev_test = 0
        best_model_path = os.path.join(self.output_path, "best_dev")
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        train_iterator = trange(num_train_epochs, desc="Epoch")

        ######################################################################
        pure_training_time = 0
        eval_time = 0
        if Record:
            self.recorder.register(model=self.model, config=self.config, dump_interval=logging_steps)

        best_step = 0
        prev_best_eval = 0.0

        ######################################################################
        for epoch, _ in enumerate(train_iterator):
            print("****** Epoch: " + str(epoch))
            epoch_iterator = tqdm(data_set, desc="Train iteration")       
            for step, batch in enumerate(epoch_iterator):
                pure_tr_time_start = time.time() 
                self.model.train()
                          
                # pdb.set_trace()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = self._batch_mapper(batch)
                
                outputs = self.model(**inputs)

                ################################################################
                # if global_step == logging_steps + 1:
                #     break
                    # pdb.set_trace()
                # self.model.check_quantize(check_weight=True)
                # self.model.check_quantize(check_feature=True)
                ################################################################       

                loss = outputs[0]  # get loss

                if self.n_gpus > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                    ################################################################
                    pure_tr_time_end = time.time()
                    pure_training_time += pure_tr_time_end - pure_tr_time_start

                    if Record:
                        self.recorder.step_count += 1
                    ################################################################

                    if logging_steps > 0 and global_step % logging_steps == 0:
                        # Log metrics and run evaluation on dev/test
                        eval_time_start = time.time()
                        best_dev, dev_test, eval_loss, f1 = self.update_best_model(
                            dev_data_set,
                            test_data_set,
                            best_dev,
                            dev_test,
                            best_result_file,
                            save_path=best_model_path,
                        )

                        ############################################################
                        eval_time_end = time.time()
                        eval_time += eval_time_end - eval_time_start
                        # self.recorder.WANDB_log({"eval_loss":eval_loss}) 
                        if prev_best_eval != best_dev:
                            best_step = global_step
                            prev_best_eval = best_dev
                            # self.recorder.rec_pstat()
                        self.recorder.writer.add_scalar('stats/eval_loss', eval_loss, self.recorder.step_count)  
                        self.recorder.writer.add_scalar('stats/eval_score', f1, self.recorder.step_count)  
                        # pdb.set_trace()
                        ############################################################

                        logger.info("lr = {}".format(self.scheduler.get_lr()[0]))
                        logger.info("loss = {}".format((tr_loss - logging_loss) / logging_steps))
                        logging_loss = tr_loss
                        # if global_step == 2:
                        #     pdb.set_trace()
 
                    if save_steps > 0 and global_step % save_steps == 0:
                        # Save model checkpoint
                        self.save_model_checkpoint(
                            output_path=self.output_path, name="checkpoint-{}".format(global_step)
                        )

                ########################################################################################################################
                # self.recorder.WANDB_log({"train_loss": tr_loss / global_step, "learning rate":self.scheduler.get_lr()[0],"global_step":global_step })
                self.recorder.writer.add_scalar("stats/train_loss", tr_loss / global_step, self.recorder.step_count)
                self.recorder.writer.add_scalar("stats/train_loss", tr_loss / global_step, self.recorder.step_count)
                self.recorder.writer.add_scalar("stats/learning_rate", self.scheduler.get_lr()[0], self.recorder.step_count)
                ########################################################################################################################

                if 0 < max_steps < global_step:
                    epoch_iterator.close()
                    break
            if 0 < max_steps < global_step:
                train_iterator.close()
                break



        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        logger.info("lr = {}".format(self.scheduler.get_lr()[0]))
        logger.info("loss = {}".format((tr_loss - logging_loss) / logging_steps))
        logger.info("best_dev = {} at step = {}".format(prev_best_eval, best_step))
        tr_hour = int(pure_training_time / 3600)
        tr_minute = int((pure_training_time % 3600) / 60)
        tr_second = int(pure_training_time % 60)
        ev_hour = int(eval_time / 3600)
        ev_minute = int((eval_time % 3600) / 60) 
        ev_second = int((eval_time) % 60)
        logger.info(f'pure_training_time = {tr_hour:02d} : {tr_minute:02d} : {tr_second:02d}')
        logger.info(f'eval_time = {ev_hour:02d} : {ev_minute:02d} : {ev_second:02d}')
        # final evaluation:
        self.update_best_model(
            dev_data_set,
            test_data_set,
            best_dev,
            dev_test,
            best_result_file,
            save_path=best_model_path,
        )

        if Record:
            self.recorder.l_to_h_heatmap()
            self.recorder.weights_to_stats()
            self.recorder.save_pstat()
            self.recorder.pstat_to_tensorboard()
            self.recorder.remove()

    def update_best_model(
        self,
        dev_data_set,
        test_data_set,
        best_dev,
        best_dev_test,
        best_result_file,
        save_path=None,):
    
        new_best_dev = best_dev
        new_test_dev = best_dev_test
        set_test = False


        for i, ds in enumerate([dev_data_set, test_data_set]):
            if ds is None:  # got no data loader
                continue
            if isinstance(ds, DataLoader):
                ds = [ds]
            
            #
            eval_loss = 0.0
            #
            for d in ds:
                logits, label_ids, ev_loss = self._evaluate(d)
                eval_loss += ev_loss
                f1 = self.evaluate_predictions(logits, label_ids)
                if i == 0 and f1 > best_dev:  # dev set
                    new_best_dev = f1
                    set_test = True
                    if save_path is not None:
                        self.save_model(save_path, args=self.training_args)

                elif set_test:
                    new_test_dev = f1
                    set_test = False
                    if best_result_file is not None:
                        with open(best_result_file, "a+") as f:
                            f.write(
                                "best dev= " + str(new_best_dev) + ", test= " + str(new_test_dev)
                            )

            eval_loss /= len(ds)   

        logger.info("\n\nBest dev=%s. test=%s\n", str(new_best_dev), str(new_test_dev))
        return new_best_dev, new_test_dev, eval_loss, f1

    def _evaluate(self, data_set: DataLoader):
        logger.info("***** Running inference *****")
        logger.info(" Batch size: {}".format(data_set.batch_size))
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(data_set, desc="Inference iteration"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = self._batch_mapper(batch)
                outputs = self.model(**inputs)
                if "labels" in inputs:
                    tmp_eval_loss, logits = outputs[:2]
                    #pdb.set_trace()
                    eval_loss += tmp_eval_loss.mean().item()
                    
                else:
                    logits = outputs[0]
            nb_eval_steps += 1
            model_output = logits.detach().cpu()
            model_out_label_ids = inputs["labels"].detach().cpu() if "labels" in inputs else None

            if preds is None:
                preds = model_output
                out_label_ids = model_out_label_ids
            else:
                preds = torch.cat((preds, model_output), dim=0)
                out_label_ids = (
                    torch.cat((out_label_ids, model_out_label_ids), dim=0)
                    if out_label_ids is not None
                    else None
                )
        if out_label_ids is None:
            return preds
        
        
        return preds, out_label_ids, eval_loss/nb_eval_steps

    def _batch_mapper(self, batch):
        mapping = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            # XLM don't use segment_ids
            "token_type_ids": batch[2]
            if self.model_type in ["bert", "quant_bert", "xlnet"]
            else None,
        }
        if len(batch) == 4:
            mapping.update({"labels": batch[3]})
        return mapping

    def evaluate_predictions(self, logits, label_ids):
        raise NotImplementedError(
            "evaluate_predictions method must be implemented in order to"
            "be used for dev/test set evaluation"
        )

    def save_model_checkpoint(self, output_path: str, name: str):
        """
        save model checkpoint

        Args:
            output_path (str): output path
            name (str): name of checkpoint
        """
        output_dir_path = os.path.join(output_path, name)
        self.save_model(output_dir_path, save_checkpoint=True)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id=None, valid_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
