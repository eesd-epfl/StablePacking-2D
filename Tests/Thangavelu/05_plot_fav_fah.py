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

#read metrics for stacked wall
all_metrics = None
for noise_level_i,noise_level in enumerate(range(10,90,10)):
    metrics_file = data_dir+f'metrics_{noise_level}.txt'
    if not os.path.exists(metrics_file):
        continue
    #read id, wall_stone_ratio, FAV, FAH, lm line by line
    for line in open(metrics_file):
        line = line.split(';')
        if line[0] == 'wall_id':
            continue
        if float(line[0]) >9:
            continue
        # if float(line[1])<=0.6:
        #     continue
        metrics = np.asarray(line[0:-1]).astype(float)
        # add noise level as first column
        metrics = np.insert(metrics, 0, noise_level)
        metrics = np.expand_dims(metrics, axis=0)
        try:
            all_metrics = np.concatenate((all_metrics, metrics), axis=0)
        except:
            all_metrics = metrics
#read metrics for reference wall
metrics_file = data_dir+f'metrics_reference_wall.txt'
ref_metrics = None
for line in open(metrics_file):
    line = line.split(';')
    if line[0] == 'wall_id':
        continue
    # if float(line[0])>=60:
    #     continue
    metrics = np.asarray(line[0:-1]).astype(float)
    metrics = np.expand_dims(metrics, axis=0)
    try:
        ref_metrics = np.concatenate((ref_metrics, metrics), axis=0)
    except:
        ref_metrics = metrics

 

plt.scatter(all_metrics[all_metrics[:,2]<0.6][:,0], all_metrics[all_metrics[:,2]<0.6][:,4], marker=markers[1],\
            facecolors='none', edgecolors=colors[3],alpha = 1.0,label = r'Ours: $F_{SF} < 0.6$')
plt.scatter(all_metrics[all_metrics[:,2]>=0.6][:,0], all_metrics[all_metrics[:,2]>=0.6][:,4], marker=markers[0],\
            facecolors='none', edgecolors=colors[0],alpha = 1.0,label = r'Ours: $F_{SF} \geq 0.6$')

plt.plot(ref_metrics[:,0], ref_metrics[:,3],'--',color = colors[2], label='Reference')
plt.xlabel(r'Noise ($\%$)')
plt.ylabel(r'$F_{AH}$')
#put legend on top right inside two columns
plt.legend(loc='upper left', ncol=2, fontsize=8)
plt.ylim([2,12])
plt.xlim([5,85])
plt.savefig(data_dir+f'FAH_noiselevel.png', dpi=600, bbox_inches='tight')
plt.close()

plt.scatter(all_metrics[all_metrics[:,2]<0.6][:,0], all_metrics[all_metrics[:,2]<0.6][:,3], marker=markers[1],\
            facecolors='none', edgecolors=colors[3],alpha = 1.0,label = r'Ours: $F_{SF} < 0.6$')
plt.scatter(all_metrics[all_metrics[:,2]>=0.6][:,0], all_metrics[all_metrics[:,2]>=0.6][:,3], marker=markers[0],\
            facecolors='none', edgecolors=colors[0],alpha = 1.0,label = r'Ours: $F_{SF} \geq 0.6$')
plt.plot(ref_metrics[:,0], ref_metrics[:,2],'--',color = colors[2], label='Reference')
plt.xlabel(r'Noise ($\%$)')
plt.ylabel(r'$F_{AV}$')
#put legend on top right inside two columns
plt.legend(loc='upper left', ncol=2, fontsize=8)
plt.ylim([5,40])
plt.xlim([5,85])
plt.savefig(data_dir+f'FAV_noiselevel.png', dpi=600, bbox_inches='tight')
plt.close()


plt.scatter(all_metrics[all_metrics[:,2]<0.6][:,0], all_metrics[all_metrics[:,2]<0.6][:,2], marker=markers[1],\
            facecolors='none', edgecolors=colors[3],alpha = 1.0,label = r'Ours: $F_{SF} < 0.6$')
plt.scatter(all_metrics[all_metrics[:,2]>=0.6][:,0], all_metrics[all_metrics[:,2]>=0.6][:,2], marker=markers[0],\
            facecolors='none', edgecolors=colors[0],alpha = 1.0,label = r'Ours: $F_{SF} \geq 0.6$')
plt.plot(ref_metrics[:,0], ref_metrics[:,1],'--',color = colors[2], label='Reference')
plt.xlabel(r'Noise ($\%$)')
plt.ylabel(r'$F_{SF}$')
#put legend on top right inside two columns
plt.legend(loc='upper right', ncol=1, fontsize=8)
plt.ylim([0,1])
plt.xlim([5,85])
plt.savefig(data_dir+f'Filling_noiselevel.png', dpi=600, bbox_inches='tight')
plt.close()

plt.scatter(all_metrics[all_metrics[:,2]<0.6][:,0], all_metrics[all_metrics[:,2]<0.6][:,5], marker=markers[1],\
            facecolors='none', edgecolors=colors[3],alpha = 1.0,label = r'Ours: $F_{SF} < 0.6$')
plt.scatter(all_metrics[all_metrics[:,2]>=0.6][:,0], all_metrics[all_metrics[:,2]>=0.6][:,5], marker=markers[0],\
            facecolors='none', edgecolors=colors[0],alpha = 1.0,label = r'Ours: $F_{SF} \geq 0.6$')
ref_lms = [0.333333333,
0.066666667,
0.066666667,
0.125,
0.166666667,
0.005,
0.026666667,
0.15,
]
plt.plot(ref_metrics[:,0], ref_lms,'--',color = colors[2], label='Reference')
plt.xlabel(r'Noise ($\%$)')
plt.ylabel(r'$F_{LM}$')
#put legend on top right inside two columns
plt.legend(loc='upper right', ncol=1, fontsize=8)
#plt.ylim([4,10])
plt.xlim([5,85])
plt.savefig(data_dir+f'LM_noiselevel.png', dpi=600, bbox_inches='tight')
plt.close()

    