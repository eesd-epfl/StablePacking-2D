"""
Plot the filling vs stability of all the walls.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
plt.style.use('science')
colors =  [
			"#045275",
			#"#089099",
			#"#7CCBA2",
			"#FCDE9C",
			"#F0746E",
			"#DC3977",
			"#7C1D6F",
            "green"
		]
markers = ['o','v','s','p','^','<','>','*','h','+','x','D','d','|','_']
# read metrics.txt
data_dir = "./result_all/"

#read metrics for reference wall
metrics_file = data_dir+f'metrics_all.txt'
all_metrics = dict()
for line in open(metrics_file):
    line = line.split(';')
    if line[0] == 'dataset':
        continue
    metrics = np.asarray(line[1:-1]).astype(float)
    metrics = np.expand_dims(metrics, axis=0)
    if line[0] in all_metrics.keys():
        all_metrics[line[0]] = np.concatenate((all_metrics[line[0]], metrics), axis=0)
    else:
        all_metrics[line[0]] = metrics

plt.scatter(all_metrics['D2=0 r=30'][:,1], all_metrics['D2=0 r=30'][:,4]/0.58, marker=markers[0],\
            facecolors='none', edgecolors=colors[1],alpha = 1.0,label = 'D2=0, r=30') 
plt.scatter(all_metrics['D2=0.2 r=30'][:,1], all_metrics['D2=0.2 r=30'][:,4]/0.58, marker=markers[1],\
            facecolors='none', edgecolors=colors[1],alpha = 1.0,label = 'D2=0.2, r=30')
plt.scatter(all_metrics['D2=0 r=10-50'][:,1], all_metrics['D2=0 r=10-50'][:,4]/0.58, marker=markers[0],\
            facecolors='none', edgecolors=colors[2],alpha = 1.0,label = 'D2=0, r=10-50')
plt.scatter(all_metrics['D2=0.2 r=10-50'][:,1], all_metrics['D2=0.2 r=10-50'][:,4]/0.58, marker=markers[1],\
            facecolors='none', edgecolors=colors[2],alpha = 1.0,label = 'D2=0.2, r=10-50')

#tight layout
plt.legend(ncol=2,loc='upper center',fontsize=8)
plt.xlim([0,1])
plt.ylim([0.08,0.16])
plt.xlabel(r'$F_{SF}$')
plt.ylabel(r'$F_{LR}$')
plt.savefig(data_dir+f'filling_stability.png', dpi=600, bbox_inches='tight')
plt.close()



plt.scatter(all_metrics['D2=0 r=30'][:,2], all_metrics['D2=0 r=30'][:,3], marker=markers[0],\
            facecolors='none', edgecolors=colors[1],alpha = 1.0,label = 'D2=0, r=30',s = 50*all_metrics['D2=0 r=30'][:,1]) 
plt.scatter(all_metrics['D2=0.2 r=30'][:,2], all_metrics['D2=0.2 r=30'][:,3], marker=markers[1],\
            facecolors='none', edgecolors=colors[1],alpha = 1.0,label = 'D2=0.2, r=30',s = 50*all_metrics['D2=0.2 r=30'][:,1])
plt.scatter(all_metrics['D2=0 r=10-50'][:,2], all_metrics['D2=0 r=10-50'][:,3], marker=markers[0],\
            facecolors='none', edgecolors=colors[2],alpha = 1.0,label = 'D2=0, r=10-50',s = 50*all_metrics['D2=0 r=10-50'][:,1])
plt.scatter(all_metrics['D2=0.2 r=10-50'][:,2], all_metrics['D2=0.2 r=10-50'][:,3], marker=markers[1],\
            facecolors='none', edgecolors=colors[2],alpha = 1.0,label = 'D2=0.2, r=10-50',s = 50*all_metrics['D2=0.2 r=10-50'][:,1])

#tight layout
plt.legend(ncol=2,loc='upper center',fontsize=8)
# plt.xlim([0,1])
plt.ylim([0,12])
plt.xlabel(r'$F_{AV}$')
plt.ylabel(r'$F_{AH}$')
plt.savefig(data_dir+f'FAH_FAV.png', dpi=600, bbox_inches='tight')
plt.close()

plt.scatter(all_metrics['D2=0 r=30'][:,1], all_metrics['D2=0 r=30'][:,2], marker=markers[0],\
            facecolors='none', edgecolors=colors[0],alpha = 1.0,label = 'D2=0, r=30') 
plt.scatter(all_metrics['D2=0.2 r=30'][:,1], all_metrics['D2=0.2 r=30'][:,2], marker=markers[1],\
            facecolors='none', edgecolors=colors[0],alpha = 1.0,label = 'D2=0.2, r=30')
plt.scatter(all_metrics['D2=0 r=10-50'][:,1], all_metrics['D2=0 r=10-50'][:,2], marker=markers[0],\
            facecolors='none', edgecolors=colors[2],alpha = 1.0,label = 'D2=0, r=10-50')
plt.scatter(all_metrics['D2=0.2 r=10-50'][:,1], all_metrics['D2=0.2 r=10-50'][:,2], marker=markers[1],\
            facecolors='none', edgecolors=colors[2],alpha = 1.0,label = 'D2=0.2, r=10-50')

#tight layout
plt.legend(ncol=1,loc='lower right',fontsize=8)
plt.xlim([0,1])
plt.ylim([0,13])
plt.xlabel(r'$F_{SF}$')
plt.ylabel(r'$F_{AV}$')
plt.savefig(data_dir+f'Filling_FAV.png', dpi=600, bbox_inches='tight')
plt.close()

    