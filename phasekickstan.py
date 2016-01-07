"""
Stan routines for working up phasekick datasets.
These routines compile and store useful models to disk, then provide utility
functions for working with the models;
"""
from __future__ import division, absolute_import, print_function
import cPickle as pickle
import os
import copy
import datetime
import pandas as pd
import numpy as np
import pystan
import phasekick


def df2dict(filename):
    """Extract a csv file into the dictionary of data required by stan models."""
    df = pd.read_csv(filename, index_col=[0, 1])
    df = df.sort_values('tp')
    phi_data = df.xs('data')['dphi_corrected [cyc]'].values*1e3
    tp = df.xs('data')['tp'].values*1e3
    phi_control = df.xs('control')['dphi_corrected [cyc]'].values*1e3
    return {'N': tp.size, 't':tp, 'y_control': phi_control, 'y': phi_data}


def update_models_dict(model_code_dict, existing_models={}, test=False):
    """Compile outdated stan models and return in a dictionary for pickling.

    Models are recompiled (which can take 10-20 seconds) if the model code has
    changed, or the model_name is n"""
    updated_models = {}
    for model_name, model_code in model_code_dict.items():
        if model_name not in existing_models or model_code != existing_models[model_name][0]:
            # test = True bypasses time-intensive compilation,
            # so this function can be tested quickly.
            if test:
                updated_models[model_name] = (model_code, hash(model_code))
            else:
                updated_models[model_name] = (model_code,
                                 pystan.StanModel(model_code=model_code,
                                                  model_name=model_name))
        else:
            updated_models[model_name] = existing_models[model_name]

    return updated_models



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
    real sig;
    sig <- 0;
    sigma[1] ~ cauchy(mu_sigma0, sigma_sigma);
    tail(sigma, N-1) ~ cauchy(mu_sigma, sigma_sigma);
}
"""}

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
    'sigma_sigma': 0.1,
    'mu_sigma0': 0.25
    },
}

directory = os.path.split(__file__)[0]
stanmodel_pkl_file = os.path.join(directory, 'stan_models.pkl')


def update_models(model_code_dict=model_code_dict,
                  stanmodel_pkl_file=stanmodel_pkl_file,
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

# This should probably be a class

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
        except AttributeError:
            data_gr[key] = val

