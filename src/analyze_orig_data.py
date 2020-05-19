import torch as ch
import pandas as pd
import numpy as np
from pathlib import Path
from pathos.multiprocessing import Pool
from argparse import ArgumentParser
from numpy.random import seed

import matplotlib as mpl
from matplotlib import rc
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('ggplot')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def agg(_df):
    agged = _df.groupby(['id', 'wnid']).agg(sel_freq=('selected', 'mean'))
    return agged.reset_index().set_index('id')

## Copied verbatim from Recht et al code release
def round_histogram(hist, target_sum):
    fractional_hist = target_sum * hist / np.sum(hist)
    floor_hist = np.floor(fractional_hist)
    floor_sum = int(np.round(np.sum(floor_hist)))
    remainder_hist = fractional_hist - floor_hist
    remainder = target_sum - floor_sum
    top_buckets = list(reversed(sorted(enumerate(remainder_hist), key=lambda x:(x[1], x[0]))))
    result = np.copy(floor_hist).astype(np.int64)
    for ii in range(remainder):
        result[top_buckets[ii][0]] += 1
    return result

def split_df(_df, head_size=5, tail_size=None):
    shuffled = _df.sample(frac=1.0)
    first_5 = shuffled.groupby('id').head(head_size)
    if tail_size is not None:
        last_5 = shuffled.groupby('id').tail(tail_size)
    else:
        last_5 = shuffled.loc[~shuffled.index.isin(first_5.index)]
    first_5, last_5 = map(agg, (first_5, last_5))
    return first_5, last_5

def match_datasets(v1, cands, N):
    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.001)]
    bins = pd.IntervalIndex.from_tuples(bins, closed='left')
    # Add a column that contains the "bin" each image belongs to
    v1['bin'] = pd.cut(v1['sel_freq'], bins, include_lowest=True)
    cands['bin'] = pd.cut(cands['sel_freq'], bins, include_lowest=True)

    all_ims = []
    total_missing = 0
    for wnid in v1['wnid'].unique():
        hist_v1 = v1[v1['wnid'] == wnid].groupby('bin').count()['sel_freq']
        hist_v1 = round_histogram(hist_v1, N)
        residual = 0 # Upwards sampling
        for (b, n) in zip(bins, hist_v1):
            src = cands[(cands['bin'] == b) & (cands['wnid'] == wnid)]
            max_ims = src.sample(n=min(n+residual, len(src)))
            residual = n + residual - len(max_ims)
            all_ims.append(max_ims)
        if residual > 0:
            print(f"Missing {residual} images from class {wnid}  ({len(cands[cands['wnid'] == wnid])} total images)")
            total_missing += residual

    return pd.concat(all_ims)

def acc(im_df): return df.set_index('id').loc[im_df.index][CLA_KEYS].mean().T.mean()
# Selection frequency ops
def sf(im_df): return im_df['sel_freq'].mean()
def heldout_sf(im_df, heldout): return heldout.loc[im_df.index]['sel_freq'].mean()
# Bootstrap
def bootstrap(arr):
    inds = np.random.choice(np.arange(len(arr)), size=1000)
    return [np.percentile(arr[inds], c, axis=0) for c in (2.5, 97.5)]

def flickr_ez_exp(_):
    seed()
    v1_df = agg(df[df['dataset'] == 'v1'])
    cand_df, heldout = split_df(df[df['dataset'] == 'v2'])
    samples, noise = split_df(df[df['dataset'] == 'v2'], tail_size=4)
    cand_ez_df = samples.loc[noise[noise['sel_freq'] >= 0.5].index]
    v2_ims = match_datasets(v1_df, cand_df, N=4)
    v2_ez_ims = match_datasets(v1_df, cand_ez_df, N=4)
    return [acc(v1_df), acc(v2_ims), acc(v2_ez_ims),
        sf(v1_df), sf(v2_ims), sf(v2_ez_ims),
        heldout_sf(v2_ims, heldout)]

def heldout_sf_exp(_):
    seed()
    v1_df = agg(df[df['dataset'] == 'v1'])
    cand_df, heldout = split_df(df[df['dataset'] == 'v2'], tail_size=5)
    v2_ims = match_datasets(v1_df, cand_df, N=4)
    stats = [sf(v1_df), sf(v2_ims), heldout_sf(v2_ims, heldout)]
    return stats

def naive_est_exp(xs):
    seed()
    pred_df = df.set_index('id')[CLA_KEYS]
    ys = []
    for nw in xs:
        v1_w, v2_w = [split_df(df[df['dataset'] == x], head_size=nw)[0]['sel_freq'] for x in ('v1', 'v2')]
        tot = 0.
        for b in v1_w.unique():
            f_given_s = pred_df.loc[v2_w[v2_w == b].index].mean()
            p_1 = (v1_w == b).mean()
            tot = tot + p_1 * f_given_s
        ys.append(df[df['dataset'] == 'v1'][CLA_KEYS].mean().T.mean() - tot.T.mean())
    return ys

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--experiment', required=True,
                choices=['heldout', 'naiveest', 'ezflickr'])
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--df-path', required=True)
    args = parser.parse_args()

    print("Loading data...")
    MY_PATH = Path(args.out_dir)
    df = ch.load(args.df_path)
    print(f"Loaded data (currently {len(df)} annotations)")

    CLA_KEYS = [k for k in df.columns if k.startswith('correct_')]
    p = Pool(args.workers)
    if args.experiment == 'ezflickr':
        FORMAT_STR = "Accs (v1, v2, v2_EZ): ({0}, {1}, {2}) | " \
             "SFs (v1, v2, v2_EZ): ({3}, {4}, {5}) | " \
             "v2 heldout SF: {6}"
        res = p.map(flickr_ez_exp, range(args.trials))
        print(FORMAT_STR.format(*list(np.array(res).mean(0))))

    elif args.experiment == 'heldout':
        FORMAT_STR = "SFs (v1, v2): ({0:.3f}, {1:.3f}) | " \
             "v2 heldout SF: {2:.3f}"
        stats = p.map(heldout_sf_exp, range(args.trials))
        print(FORMAT_STR.format(*list(np.array(stats).mean(0))))

    elif args.experiment == 'naiveest':
        fig, ax = plt.subplots(1, 1, figsize=(6,2))
        xs = [5, 6, 7, 8, 9, 10]
        res = np.array(p.map(naive_est_exp, [xs] * args.trials))
        res_df = pd.DataFrame(columns=xs, data=res).melt(var_name='xs', value_name='adj_acc')
        ch.save(res_df, str(MY_PATH / 'orig_data_naive_est_data.pt'))
        print(f"X: {xs} | Y: {res.mean(0)}")
        sns.lineplot(data=res_df, x='xs', y='adj_acc',
                        ax=ax, palette=sns.color_palette("tab10", 1))
        ax.set(xlabel='Number of annotators per image',
               ylabel='ImageNet v1/v2 accuracy gap')
        plt.tight_layout()
        fig.savefig(str(MY_PATH / 'orig_data_naive_est.png'))
