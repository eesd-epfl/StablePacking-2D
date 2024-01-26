
import pandas as pd
import pathlib
import os
#----------------------------------------------------------------
# Set working directory

_root_dir = pathlib.Path(__file__).resolve().parent
_root_dir = os.path.abspath(_root_dir)
#----------------------------------------------------------------
# read txt file as table
df = pd.read_csv(_root_dir+'/result/time.txt', sep=';', header='infer')

nb_stones = 32
number_proc = 10
nb_pose = 4
nb_cand = 7
factor = nb_cand*nb_pose/min(nb_cand*nb_pose,number_proc)

ites_full = list(range(1,1+int(max(df['iteration']))))
total_time = df['placement'].sum()+df['typology evaluation'].sum()+df['stabilization'].sum()+df['kinematics evaluation'].sum()
# write total time to txt file
with open(_root_dir+'/result/total_time.txt', 'a') as f:
    f.write(f"number_proc:{number_proc},total time (min): {total_time/number_proc/60}")