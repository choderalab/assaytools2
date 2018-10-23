"""
titration.py

Bayesian modelling of titration volumes and concentrations

"""

# imports
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import copy
from tensorflow_probability import edward2 as ed
from utils import *


class Solution:
    """
    A Solution object contains the information about the solution, i.e. species,
    concentrations, and so on.
    """
    # TODO: use this class to further record physical constants.
    def __init__(self, conc_p = 0, conc_l = 0, conc_r = 0,
                d_conc_p = 0, d_conc_l = 0, d_conc_r = 0,
                concs = None, d_concs = None):
        if concs == None:
            concs = [conc_p, conc_l, conc_r]

        if d_concs == None:
            d_concs = [d_conc_p, d_conc_l, d_conc_r]

        self.concs = np.array([conc_p, conc_l, conc_r], dtype=np.float32)
        self.d_concs = np.array([d_conc_p, d_conc_l, d_conc_r], dtype=np.float32)


class SingleWell:
    """
    A SingWell object contains all the information about the volumes, concentrations of
    species, as well as the covariance thereof, in a certain time series.

    """

    def __init__(self, analytical=True):
        self.analytical = analytical
        if analytical == True:
            self.vols = np.zeros((1, ), dtype=np.float32)
            self.vols_cov = np.zeros((1, 1), dtype=np.float32)
            self.concs = np.zeros((3, 1), dtype=np.float32)
            self.concs_cov = np.zeros((3, 1, 1), dtype=np.float32)

            # to simplify, we enforce that, the titration all come from the same source,
            # i.e., the uncertainty of the same species should stay the same.
            # TODO: generalize this to get rid of the restrictions

            self.fixed_params = {'d_vol' : None}

        elif analytical == False:
            raise NotImplementedError

    def inject(self,
                solution = None,
                vol = 0,
                d_vol = 0):
        """
        Models one titration, with:
            certain species,
            volume $V$,
            uncertainty of volume $dV$,
            concentration of that species $c$,
            uncertainty of the concentration $dc$.

        The values should be read from the solution; the errors are determined
        by uncertainty in the purity, instrument error, etc.

        Following things happen:

        1. The expected volume of the cell increased by $V$. this is modelled by
            appending value $V$ at the end of volume tensor.
        2. Uncertainty introduced to the volume. This is modelled by expanding another
            column and another row at the end of covariance matrix, and filling
            it with $dV$.
        3. The expected concentration of the certain species becomes:
            $$ \frac{c_0 V_0 + cV}{V_0 + V}$$
        4. Error introduced to the concentration. This is modelled by expanding another
            column and another row at the end of covariance matrix, and filling
            it with $$ \sigma^2(c)E(\frac{V}{V + V_0})$$

        """

        if self.analytical == True:
            # update expected volumes
            self.vols = np.append(self.vols, self.vols[-1] + vol)

            # construct the covariance matrix
            vols_cov_1d = np.diag(self.vols_cov)
            vols_cov_1d = np.append(vols_cov_1d, d_vol)
            self.vols_cov = np.tile(vols_cov_1d, (vols_cov_1d.shape[0], 1))

            # update concentration
            current_vol = self.vols[-1]
            current_concs = np.expand_dims(self.concs[:, -1], 0)
            new_concs = np.true_divide(current_concs * current_vol + vol * solution.concs,
                                        current_vol + vol).transpose()
            self.concs = np.concatenate([self.concs, new_concs], axis=1)

            # update the cov matrix of concentrations
            # TODO: make this more general
            current_conc_vars = np.array([np.diag(self.concs_cov[idx, :, :]) for idx in range(self.concs_cov.shape[0])])
            current_conc_vars = np.concatenate([current_conc_vars, np.expand_dims(solution.d_concs, axis=1)], axis=1)
            volume_scaling = np.expand_dims(self.vols, axis=1).dot(np.expand_dims(np.power(self.vols, -1), axis=1).transpose())
            self.concs_cov = np.apply_along_axis(lambda x: np.tile(x, (self.vols.shape[0], 1)), 1, current_conc_vars) * \
                            (np.triu(volume_scaling) + np.triu(volume_scaling, 1).transpose())
            self.concs_cov[:, 0, 0] = np.zeros(self.concs_cov.shape[0])


    def sample(self, n_samples = 1):
        """
        Draw samples from the defined distribution.

        Parameters
        ----------
        n_samples : the number of samples to be drawn.

        Returns
        -------
        vols : the volumes of the well, with time axis.
        concs: concentrations of species in the well, with time axis.
        """
        if self.analytical == True:
            vols_ = LogNormal(loc=np.array(self.vols, dtype=np.float64),
                        scale=np.array(self.vols_cov, dtype=np.float64))

            concs_l_ = MultivariateLogNormal(loc=np.array(self.concs[1, 2:], dtype=np.float64),
                        covariance_matrix=np.array(self.concs_cov[1, 2:, 2:], dtype=np.float64))

            return vols_().sample(n_samples), concs_l_().sample(n_samples)
            # return vols_.distribution.sample(n_samples)
            # concs_p_ = ed.MultivariateLogNormal(loc=self.concs[0, :], covariance_matrix=self.concs_cov[0, :, :])
            # return vols_.distribution.sample(n_samples), concs_p_.distributions.sample(n_samples)
