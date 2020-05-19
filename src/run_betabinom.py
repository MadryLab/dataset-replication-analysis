import matplotlib
import os
import torch as ch
import dill
from uuid import uuid4
import pandas as pd
import numpy as np
import sys
import tqdm
import shutil
from scipy.stats import norm, betabinom
from pathlib import Path
import argparse
from tqdm import tqdm
from cox.store import Store

from utils.ps_fitting import fit_models, get_classifier_data

def load_dfs(path):
    print(f'loading from dfs: {path}')
    return ch.load(path)

def model_fit(debug, num_betas, train_df):
    models = fit_models(train_df, num_betas, mode='direct',
                        cache_or_cache_dir='/tmp/', debug=debug)
    classifier_dists = get_classifier_data(models, train_df, mode='direct')
    return classifier_dists

def sample_annos(df, num_workers):
    shuffled = df.sample(frac=1).reset_index(drop=True)
    return shuffled.groupby('url').head(num_workers)

NUM_BETAS = 3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=str, required=True, help='Out directory to save results to')
    parser.add_argument('--df-path', type=str, required=True, help='Input dataframe to draw annotations from')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    s = Store(str(args.out_dir))

    # held_outs
    s.add_table('out', {
        'dists':s.OBJECT
    })

    df = load_dfs(path=args.df_path)
    dists = model_fit(args.debug, NUM_BETAS, df)

    s['out'].append_row({
        'dists':dists
    })

    print('store located in:', s.path)
    print('In Beta-Binomial Model Analysis.ipynb: set INPUT_DATA = \'f{s.path}\'')


if __name__ == '__main__':
    main()

