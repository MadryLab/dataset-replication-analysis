import torch as ch
import pandas as pd
import numpy as np
from glob import glob
import cvxpy as cvx
import matplotlib
from matplotlib import pyplot as plt
from scipy.stats import norm
from torch.distributions import binomial, categorical
import scipy.integrate

def make_objective(df, dataset, ps_dist, conditioning,
                    num_workers,  knots=[0.5, 0.7, 0.9], degree=3,
                    granularity=1000, visualize=False, eval_with=None,
                    verbose=False, debug=False, aux=None):
    prop_correct_other = df[df['dataset'] == dataset][conditioning].mean()
    df = df[df['dataset'] == dataset] # dataset
    prop_correct = df[conditioning].mean()
    df = df[df[conditioning]] # limit to f(x) = 1
    agged = df.groupby('url').agg(num_selected=('selected', 'sum'),
                                     num_total=('selected', 'count'))
    s_hat = np.array(agged['num_selected'])
    if debug:
        s_hat = s_hat[:1]
        granularity = 5

    # print(num_workers, s_hat)
    hist = np.array([np.sum(s_hat == i) for i in range(num_workers+1)])
    hist = hist / np.sum(hist) * prop_correct

    poss_s = np.linspace(0, 1, granularity)
    poss_s_hat = np.arange(0, num_workers + 1).astype(np.float32)

    num_vars = degree + len(knots) + 1
    coeffs = cvx.Variable(num_vars)
    if eval_with:
        assert(type(eval_with) == SplineDist)
    sd = eval_with or SplineDist(coeffs, degree, knots, num_workers)
    p_f_given_s = sd.pdf(poss_s)

    p_s = ps_dist(poss_s)
    p_s[p_s == np.inf] = 0
    if type(p_f_given_s) != np.ndarray or type(p_s) != np.ndarray:
        pdf = cvx.multiply(p_f_given_s, p_s)
    else:
        pdf = p_f_given_s * p_s

    # Integral fitting
    dists = binomial.Binomial(total_count=num_workers, probs=ch.tensor(poss_s))
    log_probs = dists.log_prob(ch.tensor(poss_s_hat[:,None]).repeat(1, granularity))
    p_mat = ch.exp(log_probs).numpy()
    pred_hist = p_mat[:,:-1] @ pdf[:-1] * (1 / granularity)
    diff = pred_hist - hist
    if type(diff) != np.ndarray:
        res = cvx.sum_squares(diff)
    else:
        res = (diff**2, pred_hist, hist)

    return res, p_f_given_s, coeffs

def fit_conditional(*args, knots=[0.5, 0.7, 0.9], degree=3, debug=False,
                    **kwargs):
    if debug:
        knots = [0.5]
        degree = 1

    # Solving
    loss, p_f_given_s, coeffs = make_objective(*args, knots=knots,
                                               degree=degree, debug=debug,
                                               **kwargs)
    objective = cvx.Minimize(loss)
    constraints = [p_f_given_s >= 2e-6, p_f_given_s <= 1.0]
    problem = cvx.Problem(objective, constraints)

    if not debug:
        eps_abs = 1e-6
        eps_rel = 1e-5
    else:
        eps_abs = 1
        eps_rel = 1

    print(eps_abs)

    final = problem.solve(eps_abs=eps_abs, eps_rel=eps_rel, max_iter=1000000,
                          verbose=True)

    knots, degree, num_workers = knots, degree, args[-1]

    # pdf at each s, s possibilities, sampler
    sd = SplineDist(coeffs.value, degree, knots, num_workers)
    return sd, final

class SplineDist:
    def __init__(self, coeffs, degree, knots, num_workers):
        self.coeffs = coeffs
        self.degree = degree
        self.knots = knots
        self.num_workers = num_workers

    def pdf(self, s):
        feats = [s ** d for d in range(self.degree + 1)] + \
                [((s-k) * (s > k)) ** self.degree for k in self.knots]
        feat_mat = np.stack(feats, axis=-1)
        return feat_mat @ self.coeffs

    def sample(self, num_samples=100000, binomial_n=None, visualize=True):
        if binomial_n is None: binomial_n = self.num_workers
        raise NotImplementedError
        dist = categorical.Categorical(probs=ch.tensor(pdf_at_s))
        sample = dist.sample((num_samples,))
        new_s_stars = sample.float() / (len(pdf_at_s)-1)
        bin_dist = binomial.Binomial(total_count=binomial_n,
                                        probs=new_s_stars)

        samples = bin_dist.sample().numpy()
        if visualize:
            plt.show()
            xs = np.arange(self.num_workers + 1)
            def make_freqs(ys):
                counts = np.array([(ys == x).sum() for x in xs])
                counts = counts/counts.sum()
                return counts

            plt.bar(xs, make_freqs(samples),
                    label='samples', color='red', alpha=0.5)
            plt.bar(xs, make_freqs(s_hat),
                    label='empirical dist', alpha=0.5)
            plt.legend()
        return samples
