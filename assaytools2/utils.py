"""
utils.py

Handles the utilities used in the calculations
"""

# imports
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import copy
from tensorflow_probability import edward2 as ed

# NOTE: this is very weird. not sure why you need to manually log it back.

class LogNormal:
    def __new__(self, loc, scale):
        log_normal = tf.contrib.distributions.TransformedDistribution(
        distribution = tf.contrib.distributions.Normal(loc = loc, scale = scale),
        bijector = tf.contrib.distributions.bijectors.Exp(),
        name = "LogNormal"
        )
        return log_normal

class MultivariateLogNormal:
    def __new__(self, loc, covariance_matrix):
        log_normal = tf.contrib.distributions.TransformedDistribution(
        distribution = tf.contrib.distributions.MultivariateNormalFullCovariance(loc = loc, covariance_matrix = covariance_matrix),
        bijector = tf.contrib.distributions.bijectors.Exp(),
        name = "MultivariateLogNormal"
        )
        return log_normal
