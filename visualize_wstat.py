import matplotlib.pyplot as plt
import os 
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import glob
import pickle
import numpy as np
import pdb
# import seaborn as sns

test_dir = '/home/hs402/nlp_arch_results/wstat/Test/20210429'
pickle_pattern = test_dir +'/*.pickle'
# parser = argparse.ArgumentParser()
# parser.add_argument('--wstat_dir')
# args = parser.parse_args()
# wstat_dir = args.wstat_dir 
# pickle_pattern = wstat_dir + '/*.pickle'


#data format
#{"Group" : {"name" : [], "max" : [], "min" : [], "mean" : [], "std" : [], "skew" : []}}

groups = ['Embedding', 'Attention', 'FeedForward', 'Pooler']
stats = ['max', 'min', 'mean', 'std', 'skew']
width = 0.2
color = ['r','g','b', 'c', 'm']
test = True

def main():
    
    files = [f for f in glob.glob(pickle_pattern)]
    files.sort()

    for i,g in enumerate(groups):

        for j,s in enumerate(stats):
            axes = plt.subplot(1,1,1)
            
            for h, f in enumerate(files): 
                with open(f,'rb') as pf:
                    fdata = pickle.load(pf)
                    com_in = pickle.load(pf)
                    com_out = pickle.load(pf)

                pdb.set_trace()    

                if test:
                    if g == 'Attention' and s == 'mean':
                        name = np.array(fdata[g]['name'])             
                        xlen = np.arange(len(name))     
                        value = fdata[g][s]
                        axes.bar(xlen + h * width, value, width=width, color = color[h], align='center')
                        plt.xticks(xlen, name, rotation = 'vertical', fontsize=5)
                        plt.tight_layout()
                        
                        # plt.grid(True)
                        title = g + ' Weight ' + s.capitalize()
                        plt.savefig(title + ".png")
            

            


if __name__=='__main__':
    main()

# x = plt.subplots(1,1,figsize=(20,5))
# ax.set_xticks(np.arange(length))
# ax.set_xticklabels(name, rotation='vertical',fontsize=8)
# plt.bar(name, v)
# for index, data in enumerate(v):
#     plt.text(x = index, y = data, s="{:.5e}".format(data) , fontdict=dict(fontsize=6))
#     self.writer.add_figure(group+'/'+str(k), fig) 
