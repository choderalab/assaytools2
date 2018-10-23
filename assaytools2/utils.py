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

class LogNormal:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def __call__(self):
        log_normal = tf.contrib.distributions.TransformedDistribution(
        distribution = tf.contrib.distributions.Normal(loc = self.loc, scale = self.scale),
        bijector = tf.contrib.distributions.bijectors.Exp(),
        name = "LogNormal"
        )
        return log_normal

class MultivariateLogNormal:
    def __init__(self, loc, covariance_matrix):
        self.loc = loc
        self.covariance_matrix = covariance_matrix

    def __call__(self):
        log_normal = tf.contrib.distributions.TransformedDistribution(
        distribution = tf.contrib.distributions.MultivariateNormalFullCovariance(loc = self.loc, covariance_matrix = self.covariance_matrix),
        bijector = tf.contrib.distributions.bijectors.Exp(),
        name = "MultivariateLogNormal"
        )
        return log_normal
