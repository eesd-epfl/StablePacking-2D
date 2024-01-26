import os
import pathlib
from datetime import datetime
_root_dir = pathlib.Path(__file__).resolve().parent
# write excuting time and date to file
with open(_root_dir/'finished_runs.txt', 'w') as f:
    f.write("excuting time: "+datetime.now().strftime("%d/%m/%Y %H:%M:%S")+'\n')
# write excuting time to file
with open(_root_dir/'finished_time.txt', 'w') as f:
    f.write("excuting time: "+datetime.now().strftime("%d/%m/%Y %H:%M:%S")+'\n')
# execute all run.py files in subdirectory
for root, dirs, files in os.walk(_root_dir):
    for file in files:
        if file == "run.py":
            print(root)
            status = os.system('python '+root+'/run.py')
            if status != 0:
                # continue if fails
                continue
            else:
                # write result to txt file
                with open(_root_dir/'finished_runs.txt', 'a+') as f:
                    f.write(root+'\n')
with open(_root_dir/'finished_runs.txt', 'w') as f:
    f.write("finishing time: "+datetime.now().strftime("%d/%m/%Y %H:%M:%S")+'\n')
# execute all view.py files in subdirectory
for root, dirs, files in os.walk(_root_dir):
    for file in files:
        if file == "view.py":
            print(root)
            status = os.system('python '+root+'/view.py')
            if status != 0:
                # continue if fails
                continue
            else:
                # write result to txt file
                with open(_root_dir/'finished_runs.txt', 'a+') as f:
                    f.write(root+'\n')
#summarize time for all runs
import pandas as pd
number_proc = 10
for root, dirs, files in os.walk(_root_dir):
    for i_f, file in enumerate(files):
        if file == "time.txt":
            # read txt file as table
            df = pd.read_csv(str(root)+'/'+file, sep=';', header='infer')
            ites_full = list(range(1,1+int(max(df['iteration']))))          
            total_time = df['placement'].sum()+df['typology evaluation'].sum()+df['stabilization'].sum()+df['kinematics evaluation'].sum()
            # write total time to txt file
            with open(_root_dir/'finished_time.txt', 'a+') as f:
                f.write(root+'\n')
                f.write(f"number_proc:{number_proc},total time (min): {total_time/number_proc/60}\n")

