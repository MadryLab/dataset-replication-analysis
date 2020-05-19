import torch as ch
import pandas as pd
import numpy as np 

from pathlib import Path
from argparse import ArgumentParser
from pathos.multiprocessing import Pool
from time import time
from numpy.random import seed

import matplotlib as mpl
from matplotlib import rc
from matplotlib import pyplot as plt
import seaborn as sns 
sns.set()
mpl.style.use('ggplot')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

parser = ArgumentParser()
parser.add_argument('--out-dir', required=True)
parser.add_argument('--df-path', help='Path to dataframe', required=True)
parser.add_argument('--delete-d', default=1, type=int, help='d for delete-d jackknife')
parser.add_argument('--num-replicates', default=100, type=int, help='number of replicates to use')
parser.add_argument('--save-justification', action='store_true', 
                    help='also plot the value of the estimator vs 1/n')
parser.add_argument('--workers', type=int, default=2)
args = parser.parse_args()

df = ch.load(args.df_path)
CLA_KEYS = [k for k in df.columns if k.startswith('correct_')]
# Remove models that use more data
CLA_FILTER = lambda x: not any([p in x for p in ('swsl_', 'ssl_', 'ig_')])
CLA_KEYS = list(filter(CLA_FILTER, CLA_KEYS))
MY_PATH = Path(args.out_dir)

def make_aggs(df, n, frac_samples=1.0, extra_keys=None):
    if extra_keys is None:
        extra_keys = filter(lambda x: x.startswith('correct_'), df.columns)
    good_urls = pd.Series(df['url'].unique()).sample(frac=frac_samples)
    _df = df[df['url'].isin(good_urls)]
    _df = _df.sample(frac=1.0).groupby('path').head(n)
    _df['selected'] = _df['selected'].astype(float)
    agg_args = {
        'sel_freq': ('selected', 'mean'),
        'count': ('selected', 'count'),
        **{k: (k, 'first') for k in extra_keys}
    }
    v1_agg = _df[_df['dataset'] == 'v1'].groupby('path').agg(**agg_args)
    v2_agg = _df[_df['dataset'] == 'v2'].groupby('path').agg(**agg_args)
    return v1_agg, v2_agg

def adjusted_accuracy(n):
    seed()
    v1_df, v2_df = make_aggs(df, n)
    v1_hist = v1_df.groupby('sel_freq').count()['count']
    v1_hist = v1_hist / v1_hist.sum()
    v2_hist = v2_df.groupby('sel_freq').agg({k: 'mean' for k in CLA_KEYS})
    return v2_hist.mul(v1_hist, axis=0).sum()

p = Pool(args.workers)
DF_LEN = df.groupby('url').count()['selected'].unique()[0]
K = args.delete_d

cfgs = [DF_LEN - K] * args.num_replicates
replicates = pd.DataFrame(p.map(adjusted_accuracy, cfgs))

point_est = adjusted_accuracy(DF_LEN)
jk_est = (DF_LEN * point_est - (DF_LEN - K) * replicates.mean()) / K
v1_accs = make_aggs(df, DF_LEN)[0][CLA_KEYS].mean()
results = pd.DataFrame([jk_est, v1_accs])

ch.save(results, str(MY_PATH / 'jk_results_data.pt'))
print(results, results.T.mean())

if args.save_justification:
    palette = sns.color_palette("tab10", 10)
    just_cfgs = np.repeat(np.arange(10, DF_LEN+1, 5).astype(int), 10)
    justification = pd.DataFrame(index=just_cfgs, data=p.map(adjusted_accuracy, just_cfgs))
    justification = justification.reset_index().rename(columns={'index': 'annos'})
    ch.save(justification, MY_PATH / 'jk_justification_data.pt')

    to_plot = np.random.choice(CLA_KEYS, 10, replace=False)
    justification = justification.melt(value_vars=CLA_KEYS, var_name='Classifier', 
                                        id_vars=['annos'], value_name='adj_acc')
    justification['Reciprocal number of annotators (1/n)'] = 1 / justification['annos']
    sns.lineplot(data=justification[justification['Classifier'].isin(to_plot)], 
        x='Reciprocal number of annotators (1/n)', y='adj_acc', hue='Classifier', palette=sns.color_palette("tab10"))
    plt.savefig(str(MY_PATH / 'jackknife_justification.png'))
