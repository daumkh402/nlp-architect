import matplotlib.pyplot as plt
x = plt.subplots(1,1,figsize=(20,5))
ax.set_xticks(np.arange(length))
ax.set_xticklabels(name, rotation='vertical',fontsize=8)
plt.bar(name, v)
for index, data in enumerate(v):
    plt.text(x = index, y = data, s="{:.5e}".format(data) , fontdict=dict(fontsize=6))
    self.writer.add_figure(group+'/'+str(k), fig) 
