import os
import pandas as pd
os.chdir('dataset/tcga/')

# data available at: https://github.com/ioanabica/SCIGAN
# download tcga.p file and store at dataset/tcga
# then run this script

tcga_p = pd.read_pickle('tcga.p')
df_tcga = pd.DataFrame(tcga_p['rnaseq'])
df_tcga = df_tcga/df_tcga.sum(1).values.reshape(-1,1)
df_tcga.to_pickle('df_tcga_x.pkl')