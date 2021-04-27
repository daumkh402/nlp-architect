import matplotlib.pyplot as plt
import os 
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--wstat_dir')
args = parser.parse_args()
wstat_dir args.wstat_dir 




x = plt.subplots(1,1,figsize=(20,5))
ax.set_xticks(np.arange(length))
ax.set_xticklabels(name, rotation='vertical',fontsize=8)
plt.bar(name, v)
for index, data in enumerate(v):
    plt.text(x = index, y = data, s="{:.5e}".format(data) , fontdict=dict(fontsize=6))
    self.writer.add_figure(group+'/'+str(k), fig) 
