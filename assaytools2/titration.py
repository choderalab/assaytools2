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
    """A Solution object contains the information about the solution, i.e. species,
    concentrations, and so forth.

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
    """A SingWell object contains all the information about the volumes, concentrations of
    species, as well as the covariance thereof, in a certain time series.

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, analytical=True, path_length=1.0):
        self.analytical = analytical
        self.path_length = path_length
        if analytical == True:
            self.vols = np.zeros((1, ), dtype=np.float32)
            self.vols_cov = np.zeros((1, 1), dtype=np.float32)
            self.concs = np.zeros((3, 1), dtype=np.float32)
            self.concs_cov = np.zeros((3, 1, 1), dtype=np.float32)

            # to simplify, we enforce that, the titration all come from the same source,
            # i.e., the uncertainty of the same species should stay the same.
            # TODO: generalize this to get rid of the restrictions

            self.finished_select_non_zero = False

        elif analytical == False:
            self.sampled = False
            self.finished_select_non_zero = False
            self.ind_vols = np.zeros((1, ), dtype=np.float32)
            self.ind_concs = np.zeros((3, 1), dtype=np.float32)
            self.ind_d_vols = np.zeros((1, ), dtype=np.float32)
            self.ind_d_concs = np.zeros((3, 1), dtype=np.float32)

    def inject(self,
                solution = None,
                vol = 0,
                d_vol = 0):
        """Models one titration, with:
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

        Parameters
        ----------
        solution :
             (Default value = None)
        vol :
             (Default value = 0)
        d_vol :
             (Default value = 0)

        Returns
        -------

        """

        if self.analytical == True:
            # update expected volumes
            self.vols = np.append(self.vols, self.vols[-1] + vol)

            # construct the covariance matrix
            vols_cov_1d = np.diag(self.vols_cov)
            vols_cov_1d = np.append(vols_cov_1d, d_vol)
            self.vols_cov = np.tile(vols_cov_1d, (vols_cov_1d.shape[0], 1))

            # update concentration
            current_vol = self.vols[-2]
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

        elif self.analytical == False:
            # keep track of the independently added vols and concs.
            self.ind_vols = np.append(self.ind_vols, vol)
            self.ind_concs = np.concatenate([self.ind_concs, np.expand_dims(solution.concs, axis=1)], axis=1)
            self.ind_d_vols = np.append(self.ind_d_vols, d_vol)
            self.ind_d_concs = np.concatenate([self.ind_d_concs, np.expand_dims(solution.d_concs, axis=1)], axis=1)


    @property
    def vols_rv(self):
        """ Random variable describing the volume of the cell."""
        if (self.analytical == False) and (self.sampled == False):
            self.sample()
        if (self.analytical == True) and (self.finished_select_non_zero == False):
            self.select_non_zero()

        return copy.deepcopy(tfd.Normal(loc=np.array(self.vols, dtype=np.float32),
                    scale=np.array(self.vols_cov[:, :], dtype=np.float32)))

    @property
    def concs_l_rv(self):
        """ Random variable describing the ligand concentrations of the cell."""
        if (self.analytical == False) and (self.sampled == False):
            self.sample()
        if (self.analytical == True) and (self.finished_select_non_zero == False):
            self.select_non_zero()

        return copy.deepcopy(tfd.MultivariateNormalFullCovariance(
                    loc=np.array(self.concs[1, :], dtype=np.float32),
                    covariance_matrix=np.array(self.concs_cov[1, :, :], dtype=np.float32)))

    @property
    def concs_p_rv(self):
        """ Random variable describing the protein concentrations of the cell."""
        if (self.analytical == False) and (self.sampled == False):
            self.sample()
        if (self.analytical == True) and (self.finished_select_non_zero == False):
            self.select_non_zero()

        return copy.deepcopy(tfd.MultivariateNormalFullCovariance(
                    loc=np.array(self.concs[0, :], dtype=np.float32),
                    covariance_matrix=np.array(self.concs_cov[0, :, :], dtype=np.float32)))

    @property
    def concs_r_rv(self):
        """ Random variable describing the receptor concentrations of the cell."""
        if (self.analytical == False) and (self.sampled == False):
            self.sample()
        if (self.analytical == True) and (self.finished_select_non_zero == False):
            self.select_non_zero()
        return copy.deepcopy(tfd.MultivariateNormalFullCovariance(loc=np.array(self.concs[2, :], dtype=np.float32),
                    covariance_matrix=np.array(self.concs_cov[2, :, :], dtype=np.float32)))

    def sample(self, n_samples = 1):
        """ Get a sample from an analytical cell.
        Throws an error if the cell is not analytical.

        Parameters
        ----------
        n_samples : number of samples obtained.
             (Default value = 1)

        """
        if self.analytical:
            import warning
            warning.warn("This is an analytical cell. No need to sample.")

        elif self.analytical == False:
            # define the random variables
            ind_vols_rv = MultivariateLogNormalDiag(self.ind_vols,
                                                    self.ind_d_vols)
            ind_concs_p_rv = MultivariateLogNormalDiag(self.ind_concs[0, :],
                                                       self.ind_d_concs[0, :])
            ind_concs_l_rv = MultivariateLogNormalDiag(self.ind_concs[1, :],
                                                       self.ind_d_concs[1, :])
            ind_concs_r_rv = MultivariateLogNormalDiag(self.ind_concs[2, :],
                                                       self.ind_d_concs[2, :])

            # sample independent volumes and concentrations from
            ind_vols = np.log(ind_vols_rv.sample(n_samples).numpy())
            ind_concs_p = np.log(ind_concs_p_rv.sample(n_samples).numpy())
            ind_concs_l = np.log(ind_concs_l_rv.sample(n_samples).numpy())
            ind_concs_r = np.log(ind_concs_r_rv.sample(n_samples).numpy())

            # the cumulative volume is the sum of all the volumes previously
            vols = np.tril(np.ones((ind_vols.shape[1], ind_vols.shape[1]))).dot(ind_vols.transpose()).transpose()

            # use quantity instead
            q_p = np.tril(np.ones((ind_vols.shape[1], ind_vols.shape[1]))).dot((ind_vols * ind_concs_p).transpose()).transpose()
            q_l = np.tril(np.ones((ind_vols.shape[1], ind_vols.shape[1]))).dot((ind_vols * ind_concs_l).transpose()).transpose()
            q_r = np.tril(np.ones((ind_vols.shape[1], ind_vols.shape[1]))).dot((ind_vols * ind_concs_r).transpose()).transpose()

            # calculate the concentrations from quantity
            concs_p = np.nan_to_num(np.true_divide(q_p, vols))
            concs_l = np.nan_to_num(np.true_divide(q_l, vols))
            concs_r = np.nan_to_num(np.true_divide(q_r, vols))

            # now take average and give mean and variance
            self.vols = np.average(vols.transpose(), axis=1).flatten()
            self.concs = np.concatenate([
                         np.expand_dims(np.average(concs_p.transpose(), axis=1), axis=0),
                         np.expand_dims(np.average(concs_l.transpose(), axis=1), axis=0),
                         np.expand_dims(np.average(concs_r.transpose(), axis=1), axis=0)], axis=0)

            # update volumes
            self.vols_cov = np.cov(vols.transpose())
            self.concs_cov = np.concatenate([
                             np.expand_dims(np.cov(concs_p.transpose()), axis=0),
                             np.expand_dims(np.cov(concs_l.transpose()), axis=0),
                             np.expand_dims(np.cov(concs_r.transpose()), axis=0)], axis=0)

            # mark the object as sampled
            self.sampled = True


    def select_non_zero(self, n_zeros):
        """Select strictly nonzero volumes and concentrations for analysis.

        Parameters
        ----------
        n_zeros : number of void observations.


        Returns
        -------

        """

        self.vols = self.vols[n_zeros:]
        self.concs = self.concs[:, n_zeros:]
        self.vols_cov = self.vols_cov[n_zeros:, :][:, n_zeros:]
        self.concs_cov = self.concs_cov[:, n_zeros:, :][:, :, n_zeros:]
        self.finished_select_non_zero = True
