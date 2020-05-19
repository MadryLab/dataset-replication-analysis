import numpy as np
import cvxpy as cvx
import pandas as pd
from uuid import uuid4
import scipy as sp
import torch as ch
from matplotlib import pyplot as plt
from scipy.stats import betabinom, beta
from scipy.optimize import minimize
from scipy.special import betainc, comb
import scipy.special

REL_IMPROVE_THRESH = 1e-3
MAX_RETRIES = 20
def fit_betabinom(samples_N, samples, weights, guess=None, optimize=True,
                  restart=False):
    '''Fit a beta binomial distribution with binomial count N (float) and
    observations samples ([int]) (<=N)
    '''
    def loglike_betabinom(w):
        def _f(params):
            a, b = params
            pmf_val = betabinom(samples_N, a, b).logpmf(samples)
            assert pmf_val.shape == w.shape
            return -(pmf_val * w).mean()
        return _f

    if optimize:
        # x0 = guess or [2, 2]
        if guess is None:
            if not restart:
                guess_gen = lambda : [2, 2]

            guess = [[2, 2] for _ in len(weights)]

        opt_opts = {'disp': True, 'maxiter': 1000000, 'iprint':0}
        results = []
        for w, x0 in zip(weights, guess):
            res = minimize(loglike_betabinom(w), x0=x0, tol=1e-12,
                            method='L-BFGS-B', options=opt_opts)
            if not res['success']:
                return None

            results.append(res.x)
        return results
    else:
        return np.sum([loglike_betabinom(w)(g) for w, g in zip(weights, guess)])


class BMixture:
    def __init__(self, thetas, alphas):
        self.thetas = thetas
        self.alphas = alphas

    def logpdf(self, samples):
        return np.log(self.pdf(samples))

    def pdf(self, samples):
        total_pdf = 0.0
        for (a, b), alpha in zip(self.thetas, self.alphas):
            dist = beta(a, b)
            total_pdf +=  alpha * dist.pdf(samples)
        return total_pdf

    def binom(self, N):
        return BBinomialMixture(N, self.thetas, self.alphas)

class BBinomialMixture:
    def __init__(self, N, thetas, alphas):
        self.thetas = thetas
        self.alphas = alphas
        self.N = N
        self._support = (0, self.N)

    def support(self):
        return self._support

    def logpmf(self, samples):
        return np.log(self.pmf(samples))

    def pmf(self, samples):
        total_pmf = 0.0
        for (a, b), alpha in zip(self.thetas, self.alphas):
            dist = betabinom(self.N, a, b)
            total_pmf +=  alpha * dist.pmf(samples)
        return total_pmf

def fit_betamix(X_N, X, num_betas, debug=False):
    # Thetas
    N = X_N.max()
    thetas = np.random.randint(low=2, high=10, size=(num_betas, 2))
    # Mixing coefficients:
    alphas = np.random.rand(num_betas)
    alphas /= alphas.sum()

    ROUNDS = 200 if not debug else 1
    QUIET = True
    lls = []
    for r in range(ROUNDS):
        dists = [betabinom(X_N, *theta) for theta in thetas]
        # E Step
        memberships = np.stack([a * d.pmf(X) for a, d in zip(alphas, dists)])
        memberships = memberships / memberships.sum(axis=0)

        # M Step
        fit_args = {
            'samples_N':X_N,
            'samples':X,
            'weights':memberships,
            'guess':thetas,
        }

        for i in range(MAX_RETRIES):
            new_thetas = fit_betabinom(**fit_args)
            fit_args['guess'] = np.random.randint(low=2, high=10, size=(num_betas, 2))

            success = not (new_thetas is None)
            if success:
                break

            print('RETRYING...')

        if success:
            pass
        else:
            if len(lls) > 1:
                den = np.max([lls[-1], lls[-2]])
                rel_improv = np.abs(lls[-1] - lls[-2])/den
            else:
                rel_improv = float('inf')

            if rel_improv < 0: # nono REL_IMPROVE_THRESH:
                # if we have just hit convergence early
                new_thetas = thetas
            else:
                uid = uuid4()
                fn = f'err_{uid}_{num_betas}.pt'
                p = f'/data/theory/robustopt/engstrom/store/bayes/{fn}'

                fit_args.update({
                    'round':r,
                    'lls':np.array(lls)
                })

                ch.save(fit_args, p)
                raise ValueError('Opt did not converge!')
                return None, None

        thetas = new_thetas
        likelihood = fit_betabinom(X_N, X, memberships, guess=thetas,
                                   optimize=False)
        lls.append(likelihood)
        alphas = memberships.sum(axis=1) / memberships.sum()

    with np.printoptions(precision=2, suppress=True):
        print('-----')
        print(f"Round {r} done")
        print(f"Theta: \n {np.array(thetas)}")
        print(f"Alpha: {alphas}")
        print(f"Likelihood: {likelihood}")

    ll = likelihood
    ret = BBinomialMixture(N, thetas, alphas), BMixture(thetas, alphas), ll
    return ret


def make_samples(df, dataset, conditioning, bootstrapping=False):
    assert conditioning is None
    counts = df.groupby('url').agg(count=('selected',
                                          'count'))['count'].unique()
    df = df[df['dataset'] == dataset]
    if not (conditioning is None):
        df = df[df[conditioning]]
    if bootstrapping:
        df = df.sample(frac=1, replace=True)
    agged = df.groupby('url').agg(num_selected=('selected', 'sum'),
                                     num_total=('selected', 'count'))
    observed_s = np.array(agged['num_selected'])
    observed_N = np.array(agged['num_total'])

    return observed_s, observed_N


def fit_model(df, dataset, conditioning=None, bootstrapping=False,
              visualize=True, verbose=False, workers_per_im=None,
              model='betabinom', num_betas=3, repeats=5, debug=False):
    observed_s, observed_N = make_samples(df, dataset, conditioning,
                                          bootstrapping)
    workers_per_im = set(observed_N)
    assert len(workers_per_im) == 1
    workers_per_im = list(workers_per_im)[0]

    if debug:
        observed_N = observed_N[:5]
        observed_s = observed_s[:5]

    try:
        best_x = None
        for _ in range(repeats):
            fit, induced, ll = fit_betamix(observed_N, observed_s, num_betas,
                                           debug=debug)
            if (best_x is None) or (ll < best_x[2]): best_x = (fit, induced, ll)
        fit, induced, ll = best_x
    except:
        import pdb
        pdb.set_trace()

    if visualize:
        final_n = fit.support()[1]+1
        xs = np.arange(final_n)

        fig, ax1 = plt.subplots()

        ax1.bar(xs/(final_n-1), fit.pmf(xs), alpha=0.5, label='fit',
                width=0.8/workers_per_im)

        fracs = observed_s/observed_N

        # emp_freq = emp_counts/emp_counts.sum() * workers_per_im
        #plt.bar(xs, emp_freq, alpha=0.5, label='samples')

        num_bins = workers_per_im+1
        rounded = pd.cut(fracs, np.arange(0, 1 + 1e-6, 1/num_bins),
                         labels=list(range(num_bins)), include_lowest=True)

        bins = [(rounded == b).sum() for b in range(num_bins)]
        bins = np.array(bins)
        bins = bins/bins.sum()
        ax1.bar(np.linspace(0, 1, num_bins), bins, alpha=0.5, label='emp',
                width=0.8/workers_per_im)

        #plt.legend()
        #plt.show()
        cts_xs = np.linspace(0, 1, workers_per_im+1)
        pdf = induced.pdf(cts_xs)
        ys = pdf/pdf.sum()
        ax1.plot(cts_xs, ys, label='p(s)')

    return induced, ll
