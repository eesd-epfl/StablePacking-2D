import numpy as np
import matplotlib.pyplot as plt
import os
plt.style.use('science')
colors =  [
			"#045275",
			"#089099",
			"#7CCBA2",
			"#FCDE9C",
			"#F0746E",
			"#DC3977",
			"#7C1D6F",
            "green"
		]
markers = ['o','v','^','<','>','s','p','*','h','+','x','D','d','|','_']
# read metrics.txt
data_dir = "./result_all/"


for noise_level_i,noise_level in enumerate(range(10,90,10)):
    metrics_file = data_dir+f'metrics_{noise_level}.txt'
    if not os.path.exists(metrics_file):
        continue
    #read id, wall_stone_ratio, FAV, FAH, lm line by line
    all_metrics = None
    for line in open(metrics_file):
        line = line.split(';')
        if line[0] == 'wall_id':
            continue
        if float(line[0]) >9:
            continue
        metrics = np.asarray(line[0:-1]).astype(float)
        metrics = np.expand_dims(metrics, axis=0)
        try:
            all_metrics = np.concatenate((all_metrics, metrics), axis=0)
        except:
            all_metrics = metrics

    fillings = all_metrics[:,1]
    lms = all_metrics[:,4]/0.58
    plt.scatter(fillings, lms, marker=markers[noise_level_i], label=f'Noise {noise_level}',\
                facecolors='none', edgecolors=colors[noise_level_i],alpha = 1.0)
plt.xlabel(r'$F_{SF}$')
plt.ylabel(r'$F_{LR}$')
#put legend on top right inside two columns
plt.legend(loc='upper right', ncol=2, fontsize=8)
plt.savefig(data_dir+f'Filling_vs_lm.png', dpi=600, bbox_inches='tight')
plt.close()

for noise_level_i,noise_level in enumerate(range(10,90,10)):
    metrics_file = data_dir+f'metrics_{noise_level}.txt'
    if not os.path.exists(metrics_file):
        continue
    #read id, wall_stone_ratio, FAV, FAH, lm line by line
    all_metrics = None
    for line in open(metrics_file):
        line = line.split(';')
        if line[0] == 'wall_id':
            continue
        if float(line[0]) >9:
            continue
        metrics = np.asarray(line[0:-1]).astype(float)
        metrics = np.expand_dims(metrics, axis=0)
        try:
            all_metrics = np.concatenate((all_metrics, metrics), axis=0)
        except:
            all_metrics = metrics

    fillings = all_metrics[:,1]
    lms = all_metrics[:,4]/0.58
    plt.scatter(fillings, lms, marker=markers[noise_level_i], label=f'Noise {noise_level}',\
                facecolors='none', edgecolors=colors[noise_level_i],alpha = 1.0)
plt.xlim([0.6,0.8])
plt.ylim([0,0.5])
#remove axis
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)
#remove frame
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.savefig(data_dir+f'Filling_vs_lm_part.png', dpi=600, bbox_inches='tight')
plt.close()
    


