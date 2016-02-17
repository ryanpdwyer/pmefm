"""
Stan routines for working up phasekick datasets.
These routines compile and store useful models to disk, then provide utility
functions for working with the models;
"""
from __future__ import division, absolute_import, print_function
import cPickle as pickle
import os
import copy
import io
import datetime
import pandas as pd
import numpy as np
import pystan
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import scipy.stats
from scipy import signal
sp = scipy
from scipy.optimize import curve_fit
from six import string_types
import docutils
from phasekick import img2uri, prnDict, percentile_func
from pystan.misc import _array_to_table

directory = os.path.split(__file__)[0]
model_code_dict_fname = os.path.join(directory, 'stan_model_code.pkl')

def memodict(f):
    """ Memoization decorator for a function taking a single argument """
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret 
    return memodict().__getitem__

def df2dict(filename):
    """Extract a csv file into the dictionary of data required by stan models."""
    df = pd.read_csv(filename, index_col=[0, 1])
    df = df.sort_values('tp')
    phi_data = df.xs('data')['dphi_corrected [cyc]'].values*1e3
    tp = df.xs('data')['tp'].values*1e3
    phi_control = df.xs('control')['dphi_corrected [cyc]'].values*1e3
    return {'N': tp.size, 't':tp, 'y_control': phi_control, 'y': phi_data}


# Need a dictionary of just, model_name, model_code pickled.

def model_pkl_file(model_name):
    return os.path.join(directory, 'stanmodels/', model_name+'.pkl')

def update_models_dict(model_code_dict, old_model_code_dict={}, test=False):
    """Compile outdated stan models and return in a dictionary for pickling.

    Models are recompiled (which can take 10-20 seconds) if the model code has
    changed."""
    updated_model_code_dict = {}
    for model_name, model_code in model_code_dict.items():
        if model_name not in old_model_code_dict or model_code != old_model_code_dict[model_name]:
            # test = True bypasses time-intensive compilation,
            # so this function can be tested quickly.
            updated_model_code_dict[model_name] = (model_code)
            if not test:
                sm = pystan.StanModel(model_code=model_code,
                                                  model_name=model_name)
                pickle.dump(sm, open(model_pkl_file(model_name), 'wb'))
        else:
            updated_model_code_dict[model_name] = old_model_code_dict[model_name]

    return updated_model_code_dict



# Priors should be model parameters,
# but defaults can be stored along with the model?
model_code_dict = {
# Default
    'df': """
data {
  int<lower=0> N;
  vector[N] t;
  vector[N] y_control;
  vector[N] y;
  real<upper=0> mu_df_inf;
  real<lower=0> sigma_df_inf;
  real<lower=0> mu_sigma_df;
  real<lower=0> sigma_sigma_df;
  real<lower=0> mu_tau;
  real<lower=0> sigma_tau;
  real<lower=0> mu_sigma;
  real<lower=0> sigma_sigma;
}
transformed data {
  real tmax;
  tmax <- t[N];
}
parameters {
  real<upper=0> df_inf;
  real<lower=0> tau;
  vector[N] ddf;
  real<lower=0> sigma_df;
  vector<lower=0>[N] sigma;
}
model {
  real sig;
  real phi;
  real dt;
  real df;
  sig <- 0;
  phi <- 0;
  df <- 0;
  df_inf ~ normal(mu_df_inf, sigma_df_inf);
  sigma_df ~ normal(mu_sigma_df, sigma_sigma_df);
  tau ~ normal(mu_tau, sigma_tau);
  sigma ~ cauchy(mu_sigma, sigma_sigma);
  ddf ~ normal(df_inf/tau*exp(-t/tau), sigma_df/tau);
  for (i in 1:N) {
    if (i == 1) {
      dt <- t[1];
      } else {
      dt <- t[i] - t[i-1];
    }
    df <- df + ddf[i]*dt;
    phi <- phi + df*dt;
    sig <- sig + sigma[i];
    y_control[i] ~ normal(0, sig);
    y[i] ~ normal(phi, sig);
  }
}""",
'df2':
"""
data {
  int<lower=0> N;
  vector[N] t;
  vector[N] y_control;
  vector[N] y;
  real<upper=0> mu_df_inf;
  real<lower=0> sigma_df_inf;
  real<lower=0> mu_sigma_df;
  real<lower=0> sigma_sigma_df;
  real<lower=0> mu_tau;
  real<lower=0> sigma_tau;
  real<lower=0> mu_sigma;
  real<lower=0> sigma_sigma;
  real<lower=0> mu_sigma0;
}
transformed data {
  real tmax;
  tmax <- t[N];
}
parameters {
  real<upper=0> df_inf;
  real<lower=0> tau;
  vector[N] ddf;
  real<lower=0> sigma_df;
  vector<lower=0>[N] sigma;
}
model {
  real sig;
  real phi;
  real dt;
  real df;
  sig <- 0;
  phi <- 0;
  df <- 0;
  df_inf ~ normal(mu_df_inf, sigma_df_inf);
  sigma_df ~ normal(mu_sigma_df, sigma_sigma_df);
  tau ~ normal(mu_tau, sigma_tau);
  sigma[1] ~ cauchy(mu_sigma0, sigma_sigma);
  tail(sigma, N-1) ~ cauchy(mu_sigma, sigma_sigma);
  ddf ~ normal(df_inf/tau*exp(-t/tau), sigma_df/tau);
  for (i in 1:N) {
    if (i == 1) {
      dt <- t[1];
      } else {
      dt <- t[i] - t[i-1];
    }
    df <- df + ddf[i]*dt;
    phi <- phi + df*dt;
    sig <- sig + sigma[i];
    y_control[i] ~ normal(0, sig);
    y[i] ~ normal(phi, sig);
  }
}""",
'noise_only':
"""
data {
  int<lower=0> N;
  vector[N] y_control;
  real<lower=0> mu_sigma;
  real<lower=0> sigma_sigma;
  real<lower=0> mu_sigma0;
}
parameters {
  vector<lower=0>[N] sigma;
}
model {
    sigma[1] ~ cauchy(mu_sigma0, sigma_sigma);
    tail(sigma, N-1) ~ cauchy(mu_sigma, sigma_sigma);
    y_control ~ normal(0, cumulative_sum(sigma));
}
""",
'noise_square':
"""
data {
  int<lower=0> N;
  vector[N] t;
  vector[N] y_control;
}
parameters {
  real<lower=0> sigma_0;
  real<lower=0> sigma_1;
  real<lower=0> sigma_2;
}
model {
    y_control ~ normal(0, sigma_0 + sigma_1 * t + sigma_2 * t .* t);
}
""",
'dfsq':
"""
data {
  int<lower=0> N;
  vector[N] t;
  vector[N] y_control;
  vector[N] y;
  real<upper=0> mu_df_inf;
  real<lower=0> sigma_df_inf;
  real<lower=0> mu_sigma_df;
  real<lower=0> sigma_sigma_df;
  real<lower=0> mu_tau;
  real<lower=0> sigma_tau;
}
transformed data {
  real tmax;
  tmax <- t[N];
}
parameters {
  real<upper=0> df_inf;
  real<lower=0> tau;
  vector[N] ddf;
  real<lower=0> sigma_df;
  real<lower=0> sigma_0;
  real<lower=0> sigma_1;
  real<lower=0> sigma_2;
}
model {
  real sig;
  real phi;
  real dt;
  real df;
  sig <- 0;
  phi <- 0;
  df <- 0;
  df_inf ~ normal(mu_df_inf, sigma_df_inf);
  sigma_df ~ normal(mu_sigma_df, sigma_sigma_df);
  tau ~ normal(mu_tau, sigma_tau);
  ddf ~ normal(df_inf/tau*exp(-t/tau), sigma_df/tau);
  for (i in 1:N) {
    if (i == 1) {
      dt <- t[1];
      } else {
      dt <- t[i] - t[i-1];
    }
    df <- df + ddf[i]*dt;
    phi <- phi + df*dt;
    sig <- sigma_0 + sigma_1 * t[i] + sigma_2 * t[i] .* t[i];
    y_control[i] ~ normal(0, sig);
    y[i] ~ normal(phi, sig);
  }
}""",
'dfsq_no_control':
"""
data {
  int<lower=0> N;
  vector[N] t;
  vector[N] y;
  real<upper=0> mu_df_inf;
  real<lower=0> sigma_df_inf;
  real<lower=0> mu_sigma_df;
  real<lower=0> sigma_sigma_df;
  real<lower=0> mu_tau;
  real<lower=0> sigma_tau;
}
transformed data {
  real tmax;
  tmax <- t[N];
}
parameters {
  real<upper=0> df_inf;
  real<lower=0> tau;
  vector[N] ddf;
  real<lower=0> sigma_df;
  real<lower=0> sigma_0;
  real<lower=0> sigma_1;
  real<lower=0> sigma_2;
}
model {
  real sig;
  real phi;
  real dt;
  real df;
  sig <- 0;
  phi <- 0;
  df <- 0;
  df_inf ~ normal(mu_df_inf, sigma_df_inf);
  sigma_df ~ normal(mu_sigma_df, sigma_sigma_df);
  tau ~ normal(mu_tau, sigma_tau);
  ddf ~ normal(df_inf/tau*exp(-t/tau), sigma_df/tau);
  for (i in 1:N) {
    if (i == 1) {
      dt <- t[1];
      } else {
      dt <- t[i] - t[i-1];
    }
    df <- df + ddf[i]*dt;
    phi <- phi + df*dt;
    sig <- sigma_0 + sigma_1 * t[i] + sigma_2 * t[i] .* t[i];
    y[i] ~ normal(phi, sig);
  }
}""",
'exp_sq':
"""
data {
int<lower=0> N;
vector[N] t;
vector[N] y_control;
vector[N] y;
real<upper=0> mu_df_inf;
real<lower=0> sigma_df_inf;
real<lower=0> mu_tau;
real<lower=0> sigma_tau;
}
parameters {
    real<upper=0> df_inf;
    real<lower=0> tau;
    real<lower=0> sigma_0;
    real<lower=0> sigma_1;
    real<lower=0> sigma_2;
}
model {
    vector[N] sig;
    df_inf ~ cauchy(mu_df_inf, sigma_df_inf);
    tau ~ cauchy(mu_tau, sigma_tau);
    sig <- sigma_0 + sigma_1 * t + sigma_2 * t .* t;
    y_control ~ normal(0, sig);
    y ~ normal(df_inf*(t + tau*(exp(-t/tau)-1)), sig);
}
""",
'exp_sq_no_control':
"""
data {
int<lower=0> N;
vector[N] t;
vector[N] y;
real<upper=0> mu_df_inf;
real<lower=0> sigma_df_inf;
real<lower=0> mu_tau;
real<lower=0> sigma_tau;
}
parameters {
    real<upper=0> df_inf;
    real<lower=0> tau;
    real<lower=0> sigma_0;
    real<lower=0> sigma_1;
    real<lower=0> sigma_2;
}
model {
    df_inf ~ cauchy(mu_df_inf, sigma_df_inf);
    tau ~ cauchy(mu_tau, sigma_tau);
    y ~ normal(df_inf*(t + tau*(exp(-t/tau)-1)),
        sigma_0 + sigma_1 * t + sigma_2 * t .* t);
}
""",
'exp_sigma':
"""
data {
int<lower=0> N;
vector[N] t;
vector[N] y;
vector<lower=0>[N] sigma;
real<upper=0> mu_df_inf;
real<lower=0> sigma_df_inf;
real<lower=0> mu_tau;
real<lower=0> sigma_tau;
}
parameters {
    real<upper=0> df_inf;
    real<lower=0> tau;
}
model {
    df_inf ~ cauchy(mu_df_inf, sigma_df_inf);
    tau ~ cauchy(mu_tau, sigma_tau);
    y ~ normal(df_inf*(t + tau*(exp(-t/tau)-1)),
               sigma);
}
""",
'exp2_sq_nc':
"""
data {
int<lower=0> N;
vector[N] t;
vector[N] y;
real<upper=0> mu_df_inf;
real<lower=0> sigma_df_inf;
vector<lower=0>[2] mu_tau;
vector<lower=0>[2] sigma_tau;
}
parameters {
    real<upper=0> df_inf;
    real<lower=0,upper=1> df_ratio;
    positive_ordered[2] tau;
    real<lower=0> sigma_0;
    real<lower=0> sigma_1;
    real<lower=0> sigma_2;
}
model {
    df_inf ~ cauchy(mu_df_inf, sigma_df_inf);
    tau ~ cauchy(mu_tau, sigma_tau);
    y ~ normal(df_inf * (df_ratio * (t + tau[1]*(exp(-t/tau[1])-1)) +
                (1 - df_ratio) * (t + tau[2]*(exp(-t/tau[2])-1))),
        sigma_0 + sigma_1 * t + sigma_2 * t .* t);
}
""",
'dflive':
"""
data {
int<lower=0> N;
vector[N] t;
vector[N] y;
vector[N] y_err;
int<lower=0> N_neg;
vector[N_neg] y_neg;
vector[N_neg] y_neg_err;
real<upper=0> mu_df_inf;
real<lower=0> sigma_df_inf;
real<lower=0> mu_tau;
real<lower=0> sigma_tau;
real mu_df0;
real<lower=0> sigma_df0;
}
parameters {
    real<upper=0> df_inf;
    real<lower=0> tau;
    real df0;
}
model {
    df_inf ~ normal(mu_df_inf, sigma_df_inf);
    tau ~ normal(mu_tau, sigma_tau);
    df0 ~ normal(mu_df0, sigma_df0);
    y_neg ~ normal(df0, y_neg_err);
    y ~ normal(df0 + df_inf * (1 - exp(-t/tau)), y_err);
}
""",
'dflive_doub':
"""
data {
int<lower=0> N;
vector[N] t;
vector[N] y;
vector[N] y_err;
int<lower=0> N_neg;
vector[N_neg] y_neg;
vector[N_neg] y_neg_err;
real<upper=0> mu_df_inf;
real<lower=0> sigma_df_inf;
vector<lower=0>[2] mu_tau;
vector<lower=0>[2] sigma_tau;
real mu_df0;
real<lower=0> sigma_df0;
}
parameters {
    real<upper=0> df_inf;
    real<lower=0,upper=1> ratio;
    positive_ordered[2] tau;
    real df0;
}
model {
    df_inf ~ normal(mu_df_inf, sigma_df_inf);
    tau ~ normal(mu_tau, sigma_tau);
    df0 ~ normal(mu_df0, sigma_df0);
    y_neg ~ normal(df0, y_neg_err);
    y ~ normal(df0 + df_inf * (
                        ratio * (1 - exp(-t/tau[1])) +
                        (1 - ratio) * (1-exp(-t/tau[2]))
                        )
                , y_err);
}
""",
'dflive_stretched':
"""
data {
int<lower=0> N;
vector[N] t;
vector[N] y;
vector[N] y_err;
int<lower=0> N_neg;
vector[N_neg] y_neg;
vector[N_neg] y_neg_err;
real<upper=0> mu_df_inf;
real<lower=0> sigma_df_inf;
real<lower=0> mu_tau;
real<lower=0> sigma_tau;
real mu_df0;
real<lower=0> sigma_df0;
}
parameters {
    real<upper=0> df_inf;
    real<lower=0> tau;
    real df0;
    real<lower=0,upper=1> beta;
}
model {
    df_inf ~ normal(mu_df_inf, sigma_df_inf);
    tau ~ normal(mu_tau, sigma_tau);
    df0 ~ normal(mu_df0, sigma_df0);
    y_neg ~ normal(df0, y_neg_err);
    for (i in 1:N) {
        y[i] ~ normal(df0 + df_inf * (1 - exp(-pow(t[i]/tau, beta))), y_err);
    }
}
""",
'dflive_sigma':
"""
data {
int<lower=0> N;
vector[N] t;
vector[N] y;
real<upper=0> mu_df_inf;
real<lower=0> sigma_df_inf;
real<lower=0> mu_tau;
real<lower=0> sigma_tau;
real mu_df0;
real<lower=0> sigma_df0;
real<lower=0> mu_sigma;
real<lower=0> sigma_sigma;
}
parameters {
    real<upper=0> df_inf;
    real<lower=0> tau;
    real df0;
    real<lower=0> sigma; 
}
model {
    df_inf ~ cauchy(mu_df_inf, sigma_df_inf);
    tau ~ cauchy(mu_tau, sigma_tau);
    df0 ~ cauchy(mu_df0, sigma_df0);
    sigma ~ normal(mu_sigma, sigma_sigma);
    y ~ normal(df0 + df_inf * (1 - exp(-t/tau)), sigma);
}
""",
'stretched_exp_sq_nc':
"""
data {
int<lower=0> N;
vector[N] t;
vector[N] y;
real<upper=0> mu_df_inf;
real<lower=0> sigma_df_inf;
real<lower=0> mu_tau;
real<lower=0> sigma_tau;
}
parameters {
    real<upper=0> df_inf;
    real<lower=0> tau;
    real<lower=0> sigma_0;
    real<lower=0> sigma_1;
    real<lower=0> sigma_2;
    real<lower=0,upper=1> beta;
}
model {
    df_inf ~ normal(mu_df_inf, sigma_df_inf);
    tau ~ normal(mu_tau, sigma_tau);
    for (i in 1:N) {
    y[i] ~ normal(df_inf * (t[i] +
               tau * (-tgamma(1/beta) +
                      gamma_p(1/beta, pow(t[i] / tau, beta))
                     )
                   ),
        sigma_0 + sigma_1 * t[i] + sigma_2 * t[i] * t[i]
        );
    }

}
"""
}

default_priors = {
    'df': {
        'mu_df_inf': -150,
        'sigma_df_inf': 100,
        'mu_sigma_df': 0,
        'sigma_sigma_df': 50,
        'mu_tau': 0,
        'sigma_tau': 0.5,
        'mu_sigma': 0,
        'sigma_sigma': 0.25,
    },
    'df2': {
        'mu_df_inf': -150,
        'sigma_df_inf': 100,
        'mu_sigma_df': 0,
        'sigma_sigma_df': 50,
        'mu_tau': 0,
        'sigma_tau': 0.5,
        'mu_sigma': 0,
        'sigma_sigma': 0.1,
        'mu_sigma0': 0.25
    },
    'noise_only': {
        'mu_sigma': 0,
        'sigma_sigma': 0.02,
        'mu_sigma0': 0.08
    },
    'noise_square': {},
    'dfsq': {
        'mu_df_inf': -150,
        'sigma_df_inf': 100,
        'mu_sigma_df': 0,
        'sigma_sigma_df': 50,
        'mu_tau': 0,
        'sigma_tau': 0.5,
    },
    'dfsq_no_control': {
        'mu_df_inf': -150,
        'sigma_df_inf': 100,
        'mu_sigma_df': 0,
        'sigma_sigma_df': 50,
        'mu_tau': 0,
        'sigma_tau': 0.5,
    },
    'exp_sq': {
    'mu_tau': 0.5,
    'sigma_tau': 1,
    'mu_df_inf': -20,
    'sigma_df_inf': 15,
    },
    'exp_sq_no_control': {
    'mu_tau': 0.5,
    'sigma_tau': 1,
    'mu_df_inf': -20,
    'sigma_df_inf': 15,
    },
    'exp_sigma': {
    'mu_tau': 0.5,
    'sigma_tau': 1,
    'mu_df_inf': -20,
    'sigma_df_inf': 15,
    },
    'stretched_exp_sq_nc': {
    'mu_tau': 0.5,
    'sigma_tau': 1,
    'mu_df_inf': -20,
    'sigma_df_inf': 15,
    },
    'exp2_sq_nc':{
    'mu_tau': np.array([0.5, 0.5]),
    'sigma_tau': np.array([1, 1]),
    'mu_df_inf': -20,
    'sigma_df_inf': 15,
    },
    'dflive': {
    'mu_tau': 0.5,
    'sigma_tau': 1,
    'mu_df_inf': -20,
    'sigma_df_inf': 15,
    'mu_df0': -150,
    'sigma_df0': 150
    },
    'dflive_doub': {
    'mu_df_inf': -20,
    'sigma_df_inf': 10,
    'mu_df0': 0,
    'sigma_df0': 5,
    'mu_tau': np.array([0, 1.]),
    'sigma_tau': np.array([2.5, 5.]),
    },
    'dflive_stretched': {
    'mu_tau': 0.5,
    'sigma_tau': 1,
    'mu_df_inf': -20,
    'sigma_df_inf': 15,
    'mu_df0': -150,
    'sigma_df0': 150
    },
    'dflive_sigma': {
    'mu_tau': 0.5,
    'sigma_tau': 1,
    'mu_df_inf': -20,
    'sigma_df_inf': 15,
    'mu_df0': -150,
    'sigma_df0': 150,
    'mu_sigma': 0.,
    'sigma_sigma': 20.,
    }
}

model_fit_err = {
    'dfsq': ('integrate ddf', 'quadratic'),
    'dfsq_no_control': ('integrate ddf', 'quadratic'),
    'exp': ('single exponential', 'cumsum'),
    'exp_sq': ('single exponential', 'quadratic'),
     'exp_sq_no_control': ('single exponential', 'quadratic'),
     'exp2_sq_nc': ('double exponential', 'quadratic')
}

def update_models(model_code_dict=model_code_dict,
                  stanmodel_pkl_file=model_code_dict_fname,
                  recompile_all=False,
                  global_model=True):
    """Update stan models if model code has changed. Otherwise load from disk.

    Setting recompile_all=True forces all models to be recompiled."""
    existing_dict = {}
    if not recompile_all:
        try:
            existing_dict = pickle.load(open(stanmodel_pkl_file, 'rb'))
        except IOError:
            pass

    if global_model:
        global models
    models = update_models_dict(model_code_dict, existing_dict)

    pickle.dump(models, open(stanmodel_pkl_file, 'wb'))

    return models

update_models()

pickle.dump(model_code_dict, open(model_code_dict_fname, 'wb'))

# This should probably be a class

@memodict
def get_model(model_name):
    return pickle.load(open(model_pkl_file(model_name), 'rb'))

def get_priors(model_name, data, **priors):
    """Return a copy of data, updated with priors kwargs,
    and default priors associated with the model."""
    updated_data = copy.copy(data)
    updated_priors = copy.copy(default_priors[model_name])
    updated_priors.update(priors)
    updated_data.update(updated_priors)
    return updated_data


def run(model_name, data, chains=4, iter=2000, **priors):
    """Run the model associated with model_name"""
    sm = models[model_name][1]
    updated_data = get_priors(model_name, data, **priors)

    return sm.sampling(data=updated_data, chains=chains, iter=iter)


def save(gr, model_name, model_code, out, compress=True):
    if compress:
        kwargs = {'compression': "gzip", 'compression_opts':9, 'shuffle':True}
    else:
        kwargs = {}

    gr['model_name'] = model_name
    gr['model_code'] = model_code
    gr['timestamp'] = datetime.datetime.isoformat(datetime.datetime.now())

    summary = out.summary()
    params = out.extract(permuted=True)

    gr.create_dataset('summary', summary['summary'].shape, **kwargs)
    gr['summary'][:] = summary['summary']

    gr['summary_colnames'] = np.array(summary['summary_colnames'], dtype=np.str)
    gr['summary_rownames'] = np.array(summary['summary_rownames'], dtype=np.str)
    param_gr = gr.create_group('params')
    for key, val in params.items():
        param_gr.create_dataset(key, val.shape, **kwargs)
        param_gr[key][:] = val

    gr['parameters'] = np.array(params.keys(), dtype=np.str)

    data_gr = gr.create_group('data')
    for key, val in out.data.items():
        try:
            data_gr.create_dataset(key, val.shape, **kwargs)
            data_gr[key][:] = val
        except (AttributeError, TypeError):
            data_gr[key] = val


class PhasekickModel(object):
    """A class to store phasekick model data."""
    def __init__(self, model_name, data_or_fname, model=None, priors=None):
        self.model_name = model_name
        self.model_code = models[model_name]

        if model is None:
            self.sm = get_model(model_name)
        else:
            self.sm = model

        if isinstance(data_or_fname, string_types):
            self.data = df2dict(data_or_fname)
            self.data_fname = data_or_fname
        else:
            self.data = data_or_fname

        self.default_priors = default_priors[model_name]
        if priors is None:
            self.priors = copy.copy(self.default_priors)
        else:
            self.priors = priors

    def run(self, chains=4, iter=2000, priors=None, **priors_kwargs):
        if priors is not None:
            self.priors = priors
        self.priors.update(priors_kwargs)
        updated_data = copy.copy(self.data)
        updated_data.update(self.priors)
        self.out = self.sm.sampling(data=updated_data, chains=chains, iter=iter)
        return self.out

    def save(self, gr_or_fname, compress=True):
        if isinstance(gr_or_fname, string_types):
            with h5py.File(gr_or_fname, 'w') as gr:
                save(gr, self.model_name, self.model_code, self.out,
                     compress=compress)
        else:
            save(gr_or_fname, self.model_name, self.model_code, self.out,
                 compress=compress)


def cdf(x, loc, scale):
    """The cumulative probability function of the normal distribution."""
    return sp.stats.norm.cdf(x, loc=loc, scale=scale)

def fit_residuals(sorted_residuals):
    """Return the result of fitting the residuals to the normal cdf."""
    size = sorted_residuals.size
    y = np.arange(1, 1 + size) / size

    popt, _ = curve_fit(cdf, sorted_residuals, y)

    return popt

def gr2datadict(gr):
    return {key: val.value for key, val in gr['data'].items() if not isinstance(val.value, np.ndarray)}


def gr2summary_str(gr, ndigits=2):
    summary = [
        gr['model_name'].value,
        gr.file.filename,
        prnDict(gr2datadict(gr), braces=False),
        _array_to_table(gr['summary'][:], gr['summary_rownames'][:], 
                gr['summary_colnames'][:], ndigits)
    ]
    return '\n\n'.join(summary)

class PlotStanModels(object):
    def __init__(self, name, grs_or_fnames):
        self.name = name
        self.filenames = []
        self.grs = []
        for gr_or_fname in grs_or_fnames:
            if isinstance(gr_or_fname, string_types):
                self.filenames.append(gr_or_fname)
                gr = h5py.File(gr_or_fname, 'r')
            else:
                self.filenames.append(None)
                gr = gr_or_fname

            self.grs.append(gr)

        self.data = {key: val.value for key, val in gr['data'].items()}

        self.y = self.data['y']
        self.y_control = self.data['y_control']
        self.t = self.data['t']

        self._y_fit = [get_y_fit(gr) for gr in self.grs]
        self.y_fit = [percentile_func(_y_fit) for _y_fit in self._y_fit]

        self._sigma = [get_sigma(gr) for gr in self.grs]
        self.sigma = [percentile_func(_sigma) for _sigma in self._sigma]

        self.residuals = [self.y - y_fit(50) for y_fit in self.y_fit]
        self.reduced_residuals = [residual / sigma(50) for residual, sigma in
                                  zip(self.residuals, self.sigma)]

        self.sorted_residuals = [np.sort(residuals) for residuals in
                                 self.reduced_residuals]

        self.popt_reduced_residuals = [fit_residuals(residuals) for residuals
                                       in self.sorted_residuals]

        self.summary_strs = [gr2summary_str(gr) for gr in self.grs]

        self.data_strs = [{key: val.value for key, val in gr['data'].items() if not isinstance(val.value, np.ndarray)} for gr in self.grs]

        self.data2print = [prnDict(d, braces=False) for d in self.data_strs]

        self._df = [get_df(gr) for gr in self.grs]
        self.df = [percentile_func(_df) for _df in self._df]

    def plot_df(self, figax=None, alpha=0.2, rcParams={}, fname=None):
        fig, ax = plt.subplots()
        for df in self.df:
            lines = ax.plot(self.t, df(50))
            ax.fill_between(self.t, df(15), df(85), color=lines[0].get_c(), alpha=alpha)

        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Frequency shift [Hz]")

        if fname is not None:
            fig.tight_layout()
            fig.savefig(fname)

        return fig, ax

    def plot_phi(self, figax=None, alpha=0.4, rcParams={}, fname=None):
        fig, ax = plt.subplots()
        ax.plot(self.t, self.y, '.')
        ax.plot(self.t, self.y_control, '.')
        for y_fit, sigma in zip(self.y_fit, self.sigma):
            lines = ax.plot(self.t, y_fit(50))
            sig = sigma(50)
            ax.fill_between(self.t, y_fit(50) - sig, y_fit(50) + sig,
                            color=lines[0].get_c(), alpha=alpha, zorder=10)

        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Phase shift [mcyc.]")

        if fname is not None:
            fig.tight_layout()
            fig.savefig(fname)

        return fig, ax


    def plot_residuals(self, figax=None, fname=None):
        fig, ax = plt.subplots()
        for residual in self.residuals:
            ax.plot(self.t, residual, '.')

        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Residual [mcyc.]")

        if fname is not None:
            fig.tight_layout()
            fig.savefig(fname)

        return fig, ax

    def plot_reduced_residuals(self, figax=None, fname=None):
        fig, ax = plt.subplots()
        for residual in self.reduced_residuals:
            ax.plot(self.t, residual, '.')

        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Reduced resid.")

        if fname is not None:
            fig.tight_layout()
            fig.savefig(fname)

        return fig, ax

    def plot_cdf_residuals(self, figax=None, fname=None):
        N = self.sorted_residuals[0].size
        y = np.arange(N) / N
        fig, ax = plt.subplots()
        for popt, sorted_residuals in zip(self.popt_reduced_residuals, 
                                          self.sorted_residuals):
            lines = ax.plot(sorted_residuals, y, '.')
            ax.plot(sorted_residuals, cdf(sorted_residuals, *popt), color=lines[0].get_c())

        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Reduced resid.")
        ax.set_ylabel("Cumulative probabilty")

        if fname is not None:
            fig.tight_layout()
            fig.savefig(fname)

        return fig, ax

    def plot_filtered_reduced_residuals(self, N=2, bandwidth=0.02, fname=None):
        fig, ax = plt.subplots()
        b, a = signal.butter(N, bandwidth)
        for reduced_residuals in self.reduced_residuals:
            lines = ax.plot(self.t, reduced_residuals, alpha=0.15, zorder=1)
            ax.plot(self.t, signal.lfilter(b, a, reduced_residuals), color=lines[0].get_c(), linewidth=2, zorder=10)

        ax.axhline(color='k')
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin/2, ymax/2)

        if fname is not None:
            fig.tight_layout()
            fig.savefig(fname)

        return fig, ax

    def report(self, outfile=None, outdir=None):
        if outfile is None:
            outfile = self.name

        basename = os.path.splitext(outfile)[0]

        if outdir is not None and not os.path.exists(outdir):
            os.mkdir(outdir)

        if outdir is None:
            outdir=''

        html_fname = os.path.join(outdir, basename+'.html')


        phi_fname = os.path.join(outdir, self.name+'-phi.png')
        df_fname = os.path.join(outdir, self.name+'-df.png')
        red_resid_fname = os.path.join(outdir, self.name+'-red_resid.png')
        resid_fname = os.path.join(outdir, self.name+'-resid.png')
        cdf_resid_fname = os.path.join(outdir, self.name+'-cdf_resid.png')
        filt_resid_fname = os.path.join(outdir, self.name+'-filt_resid.png')

        self.plot_phi(fname=phi_fname)
        self.plot_df(fname=df_fname)
        self.plot_cdf_residuals(fname=cdf_resid_fname)
        self.plot_filtered_reduced_residuals(fname=filt_resid_fname)
        self.plot_reduced_residuals(fname=red_resid_fname)
        self.plot_residuals(fname=resid_fname)

        locs = np.round(np.array([popt[0] for popt in self.popt_reduced_residuals]), 2)
        scales = np.round(np.array([popt[1] for popt in self.popt_reduced_residuals]), 2)

        indented_summary_strs = []
        for string in self.summary_strs:
            indented_str = []
            string_lines = string.split('\n')
            for line in string_lines:
                indented_str.append('    '+line)
            indented_summary_strs.append('\n'.join(indented_str))

        body = """\
======================
Phasekick model report
======================


Model summaries
===============

::
    
    {summary_strs}

Phase
-----

.. image:: {phi_plot}


Frequency shift
---------------

.. image:: {df_plot}

Residuals
---------

.. image:: {resid_plot}

Reduced residuals
-----------------

.. image:: {red_resid_plot}

Filtered reduced residuals
--------------------------

.. image:: {filt_resid_plot}

Reduced residuals cumulative distribution
-----------------------------------------

.. image:: {cdf_resid_plot}

::
    
    locs: {locs}

    scales: {scales}


""".format(summary_strs="\n    \n".join(indented_summary_strs),
                phi_plot=phi_fname, df_plot=df_fname,
                resid_plot=resid_fname, red_resid_plot=red_resid_fname,
                filt_resid_plot=filt_resid_fname,
                cdf_resid_plot=cdf_resid_fname,
                locs=locs, scales=scales
            )

        image_dependent_html = docutils.core.publish_string(body, writer_name='html')
        self_contained_html = unicode(img2uri(image_dependent_html), 'utf8')

        with io.open(html_fname, 'w', encoding='utf8') as f:
            f.write(self_contained_html)


        for fname in [phi_fname, df_fname, red_resid_fname, resid_fname,
                      cdf_resid_fname, filt_resid_fname]:
            try:
                os.remove(fname)
            except:
                pass


def sigma_sq(fh):
    return (np.outer(fh['params/sigma_0'][:], np.ones(fh['data/N'].value)) +
            np.outer(fh['params/sigma_1'][:], fh['data/t'][:]) + 
            np.outer(fh['params/sigma_2'][:], fh['data/t'][:]**2)
            )


def sigma_cumsum(fh):
    return np.cumsum(fh['params/sigma'][:], axis=1)


def get_sigma(fh):
    err_type = get_err_type(fh)
    if err_type == 'quadratic':
        return sigma_sq(fh)
    elif err_type == 'cumsum':
        return sigma_cumsum
    else:
        raise ValueError("{} must be 'quadratic' or 'cumsum'".format(err_type))


def ddf2df(ddf, t):
    dt = np.r_[t[0], np.diff(t)]
    return np.cumsum(ddf * dt, axis=1)


def df2dphi(df, t):
    dt = np.r_[t[0], np.diff(t)]
    return np.cumsum(df * dt, axis=1)


def fh2df(fh):
    return ddf2df(fh['params/ddf'][:], fh['data/t'][:])


def fh2dphi(fh):
    df = fh2df(fh)
    return df2dphi(df, fh['data/t'][:])

def exp2df(t, df, tau):
    return (np.outer(df, np.ones(t.size))
            * (1 - np.exp(-np.outer(np.ones(tau.size), t) /
            np.outer(tau, np.ones(t.size)))))

def exp2dphi(t, df, tau):
    return (np.outer(df, t) +
            np.outer(df * tau, np.ones(t.size))
            * (np.exp(-np.outer(np.ones(tau.size), t) /
            np.outer(tau, np.ones(t.size)))-1))


def fh_exp2dphi(fh):
    t = fh['data/t'][:]
    df = fh['params/df_inf'][:]
    tau = fh['params/tau'][:]
    return exp2dphi(t, df, tau)


def fh_exp2df(fh):
    t = fh['data/t'][:]
    df = fh['params/df_inf'][:]
    tau = fh['params/tau'][:]

    return exp2df(t, df, tau)


def fh_exp_doub2df(fh, t=None):
    if t is None:
        t = fh['data/t'][:]
    df_inf = fh['params/df_inf'][:]
    ratio = fh['params/df_ratio'][:]
    tau = fh['params/tau'][:]

    df = df_inf * np.c_[ratio, 1-ratio].T

    out = np.zeros((df_inf.size, t.size))
    for _df, _tau in zip(df, tau.T):
        out += exp2df(t, _df, _tau)

    return out

def fh_exp_doub2dphi(fh, t=None):
    if t is None:
        t = fh['data/t'][:]
    df_inf = fh['params/df_inf'][:]
    ratio = fh['params/df_ratio'][:]
    tau = fh['params/tau'][:]

    df = df_inf * np.c_[ratio, 1-ratio].T

    out = np.zeros((df_inf.size, t.size))
    for _df, _tau in zip(df, tau.T):
        out += exp2dphi(t, _df, _tau)

    return out

def get_fit_type(fh):
    return model_fit_err[fh['model_name'].value][0]


def get_err_type(fh):
    return model_fit_err[fh['model_name'].value][1]


def get_y_fit(fh):
    fit_type = get_fit_type(fh)
    if fit_type == 'single exponential':
        return fh_exp2dphi(fh)
    elif fit_type == 'double exponential':
        return fh_exp_doub2dphi(fh)
    elif fit_type == 'integrate ddf':
        return fh2dphi(fh)


def get_sigma(fh):
    err_type = get_err_type(fh)
    if err_type == 'quadratic':
        return sigma_sq(fh)
    elif err_type == 'cumsum':
        return sigma_cumsum(fh)

def get_df(fh):
    fit_type = get_fit_type(fh)
    if fit_type == 'single exponential':
        return fh_exp2df(fh)
    elif fit_type == 'double exponential':
        return fh_exp_doub2df(fh)
    elif fit_type == 'integrate ddf':
        return fh2df(fh)


def plot_phi(fh, figax=None, rcParams={}):
    y = fh['data/y'][:]
    t = fh['data/t'][:]
    y_control = fh['data/y_control'][:]
    y_fit = np.percentile(get_y_fit(fh), 50, axis=0)
    print(y_fit)
    if figax is None:
        fig, ax = plt.subplots()

    with mpl.rc_context(rcParams):
        ax.plot(t, y_control, 'g.')
        ax.plot(t, y, 'b.')
        ax.plot(t, y_fit, 'm-', linewidth=2)

    return fig, ax
