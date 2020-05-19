import torch as ch
from os import path
import dill
from uuid import uuid4
import pandas as pd
import numpy as np
import sys
import shutil
from scipy.integrate import quad
import dill
from uuid import uuid4
import cvxpy as cvx
from scipy.stats import norm
from pathlib import Path
import argparse
from glob import glob
from pathlib import Path
import argparse
import tqdm

from .beta_mixtures import fit_model
from .splineqp_utils import fit_conditional

class ClassifierData:
    def __init__(self, name, v1_acc, v2_acc, model_v2, model_v1,
                 llp1, llp2, llcond, s_given_f=None, f_given_s=None):
        self.name = name
        self.v1_acc = v1_acc
        self.v2_acc = v2_acc
        self.model_v2 = model_v2
        self.s_given_f = s_given_f
        self.f_given_s = f_given_s
        self.model_v1 = model_v1
        self.corrected_accuracy = self.p_fx_corrected()
        self.llp1 = llp1
        self.llp2 = llp2
        self.llcond = llcond

    def p2_fx_given_s(self, s):
        if self.s_given_f:
            num = self.v2_acc * self.s_given_f.pdf(s)
            denom = self.model_v2.pdf(s)
            return num / denom
        else:
            return self.f_given_s.pdf(s)

    def to_integrate(self, s):
        return self.p2_fx_given_s(s) * self.model_v1.pdf(s)

    def p_fx_corrected(self):
        return quad(self.to_integrate, 0., 1., limit=200)[0]

def fit_models(df, num_betas, mode, cache_or_cache_dir, verbose=False,
                debug=False):
    models = {}
    num_workers = df.groupby('url').agg(count=('selected',
                                               'count'))['count'].unique()
    assert len(num_workers) == 1
    print(f"{num_workers[0]} workers")

    def direct_fit(dataset, conditioning, ps_dist, aux):
        to_pass = {
            'visualize': False,
            'verbose': verbose,
            'conditioning': conditioning,
            'aux': aux,
            'num_workers': int(num_workers[0]),
            'debug': debug
        }
        return fit_conditional(df, dataset, ps_dist, **to_pass)

    def ez_fit(dataset, conditioning=None):
        to_pass = {
            'visualize':False,
            'verbose':verbose,
            'num_betas':num_betas,
            'conditioning':conditioning,
            'debug': debug
        }
        return fit_model(df, dataset, **to_pass)

    if isinstance(cache_or_cache_dir, str):
        prob_v1, l1 = ez_fit('v1')
        prob_v2, l2 = ez_fit('v2')
        to_save = {
            'v1': (prob_v1, l1),
            'v2': (prob_v2, l2)
        }
        ch.save(to_save, path.join(cache_or_cache_dir, "betas.pt"))
        print(f"Saved to {cache_or_cache_dir} betas.pt")
    else:
        prob_v1, l1 = cache_or_cache_dir['v1']
        prob_v2, l2 = cache_or_cache_dir['v2']

    models['v1'] = (prob_v1, l1)
    models['v2'] = (prob_v2, l2)

    for k in [_k for _k in df.keys() if 'correct_' in _k]:
        if mode == 'direct':
        #    prob_v1_cond, l1 = direct_fit('v1', k, prob_v1.pdf, prob_v2.pdf)
            prob_v2_cond, l2 = direct_fit('v2', k, prob_v2.pdf, prob_v1.pdf)
        elif mode == 'ez':
        #    prob_v1_cond, l1 = ez_fit('v1', k)
            prob_v2_cond, l2 = ez_fit('v2', k)

       # models[f'v1|{k}'] = (prob_v1_cond, l1)
        models[f'v2|{k}'] = (prob_v2_cond, l2)

    return models

def get_classifier_data(models, df, mode='ez'):
    model_v1, ll1 = models['v1']
    model_v2, ll2 = models['v2']

    x_pts = []
    y_pts = []
    y2_pts = []
    names = []

    classifier_datas = []

    sorted_keys = models.keys()
    for classifier in tqdm.tqdm([k for k in sorted_keys if 'v2|' in k]):
        classifier_name = '|'.join(classifier.split('|')[1:])

        v2_cond_model, llc = models[f'v2|{classifier_name}']

        prob_correct_v1 = df[df['dataset'] == 'v1'][classifier_name].mean()
        prob_correct_v2 = df[df['dataset'] == 'v2'][classifier_name].mean()

        kwargs = {
            'name':classifier_name,
            'v1_acc':prob_correct_v1,
            'v2_acc':prob_correct_v2,
            'model_v2':model_v2,
            'model_v1':model_v1,
            'llp1':ll1,
            'llp2':ll2,
            'llcond':llc
        }
        if mode == 'ez':
            kwargs['s_given_f'] = v2_cond_model
        elif mode == 'direct':
            kwargs['f_given_s'] = v2_cond_model

        cd = ClassifierData(**kwargs)
        classifier_datas.append(cd)

    return classifier_datas
