"""
utils.py

Handles the utilities used in the calculations
"""

# imports
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import tensorflow_probability as tfp
tfd = tfp.distributions
import copy
from tensorflow_probability import edward2 as ed

# NOTE: this is very weird. not sure why you need to manually log it back.

class LogNormal:
    """ Wrapper for log normal distribution."""
    def __new__(self, loc, scale):
        log_normal = tfd.TransformedDistribution(
        distribution = tfd.Normal(loc = loc, scale = scale),
        bijector = tfp.bijectors.Exp(),
        name = "LogNormal"
        )
        return log_normal

class MultivariateLogNormal:
    """ Wrapper for multivariate log normal distribution."""
    def __new__(self, loc, covariance_matrix):
        log_normal = tfd.TransformedDistribution(
        distribution = tfd.MultivariateNormalFullCovariance(loc = loc, covariance_matrix = covariance_matrix),
        bijector = tfp.bijectors.Exp(),
        name = "MultivariateLogNormal"
        )
        return log_normal

class MultivariateLogNormalDiag:
    """ Wrapper for multivariate log normal distribution with diagonal scale matrix."""
    def __new__(self, loc, scale_diag):
        log_normal = tfd.TransformedDistribution(
        distribution = tfd.MultivariateNormalDiag(loc = loc, scale_diag = scale_diag),
        bijector = tfp.bijectors.Exp(),
        name = "MultivariateLogNormalDiag"
        )
        return log_normal
