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
from .utils import *

@tf.contrib.eager.defun
def cov(X):
    """ Covariance of an 2D array.

    $ K_{XX} = Cov(X, X) = E(XX^T) - E(X)E^T(X) $

    Parameters
    ----------
    X : tf.Tensor

    Returns
    -------
    cov : the covariance matrix
    """
    X = tf.expand_dims(X, 1)
    cov_ = tf.reduce_mean(tf.matmul(X, tf.transpose(X, [0, 2, 1])), 0)\
        - tf.matmul(
                tf.reduce_mean(X, 0),
                tf.transpose(tf.reduce_mean(X, [0, 2, 1])))

    return cov_

class Solution:
    """A Solution object contains the information about the solution, i.e. species,
    concentrations, and so forth.

    Attributes
    ----------
    concs : list or np.ndarray, shape = (3, ), concentrations of the solution
    d_concs : list or np.ndarray, shape = (3, ), uncertainty of concentrations
        of the solution

    """
    # TODO: use this class to further record physical constants.
    def __init__(self, conc_p = 0, conc_l = 0, conc_r = 0,
                d_conc_p = 0, d_conc_l = 0, d_conc_r = 0,
                concs = None, d_concs = None):

        if concs == None:
            concs = [conc_p, conc_l, conc_r]

        if d_concs == None:
            d_concs = [d_conc_p, d_conc_l, d_conc_r]

        self.concs = tf.constant(concs, dtype=tf.float32)
        self.d_concs = tf.constant(d_concs, dtype=tf.float32)


class Plate:
    """A Plate object contains all the information about the volumes,
    concentrations of species, as well as the covariance thereof,
    in a certain time series.

    Attributes
    ----------
    n_wells : int
        number of cells in the plate.
    path_length : float
        length of the path of the cells.
    sampled : Boolean
        whether this plate is sampled or not.
    finished_selected_non_zero : Boolean
        to keep record of whether the process of select nonzero volumes are
        finished, to ensure that Cholesky decomposition works.
    ind_vols : tf.Tensor
        volumes of liquid in cells in each step.
    ind_concs : tf.Tensor
        concentrations of species in each step.
    ind_d_vols : tf.Tensor
        uncertainty associated with ind_vols.
    ind_d_concs : tf.Tensor
        uncertainty associated with ind_concs.

    """

    def __init__(self, n_wells: int, path_length: float = 1.0) -> None:
        # generic properties of the plate
        self.n_wells = n_wells
        self.path_length = path_length

        # flags for status
        self.sampled = False
        self.finished_select_non_zero = False

        # matrices to keep track of the quantities in the plate

        # individual volumes at each time step
        # (time, n_wells)
        self.ind_vols = tf.zeros((1, n_wells), dtype=tf.float32)

        # individual concentrations at each time step
        # (time, 3, n_wells)
        self.ind_concs = tf.zeros((1, 3, n_wells), dtype=tf.float32)

        # uncertainty associated with ind_vols
        # (time, n_wells)
        self.ind_d_vols = tf.zeros((1, n_wells), dtype=tf.float32)

        # uncertainty associated with ind_concs
        # (time, 3, n_wells)
        self.ind_d_concs = tf.zeros((1, 3, n_wells), dtype=tf.float32)

    def inject(
            self,
            solution = None,
            well_idx: int = 0,
            vol: float = 0.0,
            d_vol: float = 0.0) -> None:

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
        2. Uncertainty introduced to the volume. This is modelled by expanding
            another column and another row at the end of covariance matrix,
            and filling it with $dV$.
        3. The expected concentration of the certain species becomes:
            $$ \frac{c_0 V_0 + cV}{V_0 + V}$$
        4. Error introduced to the concentration. This is modelled by expanding
            another column and another row at the end of covariance matrix,
            and filling it with $$ \sigma^2(c)E(\frac{V}{V + V_0})$$

        Parameters
        ----------
        solution :
             the solution object to be injected into the plate.
        cells : list
             indicies of cells
        vols :
             volumes of the injection
        d_vol :
             uncertainty associated with val

        """
        # assert that the place of injection is within the plate
        assert well_idx < self.n_wells

        # handle ind_vols
        new_ind_vols = tf.Variable(
            tf.zeros((1, self.n_wells), dtype=tf.float32))
        new_ind_vols[0, well_idx].assign(vol + new_ind_vols[0, well_idx])
        self.ind_vols = tf.concat((self.ind_vols, new_ind_vols),
            axis=0)

        # handle ind_concs
        new_ind_concs = tf.Variable(
            tf.zeros((1, 3, self.n_wells), dtype=tf.float32))
        new_ind_concs[0, :, well_idx].assign(solution.concs
            + new_ind_concs[0, :, well_idx])
        self.ind_concs = tf.concat((self.ind_concs, new_ind_concs),
            axis=0)

        # handle ind_d_vols
        new_ind_d_vols = tf.Variable(
            tf.zeros((1, self.n_wells), dtype=tf.float32))
        new_ind_d_vols[0, well_idx].assign(d_vol + new_ind_d_vols[0, well_idx])
        self.ind_d_vols = tf.concat((self.ind_d_vols, new_ind_d_vols),
            axis=0)

        # handle ind_d_concs
        new_ind_d_concs = tf.Variable(
            tf.zeros((1, 3, self.n_wells), dtype=tf.float32))
        new_ind_d_concs[0, :, well_idx].assign(solution.d_concs
            + new_ind_d_concs[0, :, well_idx])
        self.ind_d_concs = tf.concat((self.ind_d_concs, new_ind_d_concs),
            axis=0)

    def sample(self, n_samples: int = 1) -> None:
        """ Sample independent volumes and concentrations and compute the
        cumulative volumes and concentrations.

        Parameters
        ----------
        n_samples : int
            the number of samples

        """
        time = int(self.ind_vols.shape[0])
        n_wells = int(self.ind_vols.shape[1])

        # ========
        # sampling
        # ========

        # unravel time series
        ind_vols_fl = tf.reshape(self.ind_vols, [-1])
        ind_concs_fl = tf.reshape(self.ind_concs, [-1])
        ind_d_vols_fl = tf.reshape(self.ind_d_vols, [-1])
        ind_d_concs_fl = tf.reshape(self.ind_d_concs, [-1])

        # put them into random variables
        ind_vols_fl_rv = MultivariateLogNormalDiag(ind_vols_fl,
            ind_d_vols_fl)
        ind_concs_fl_rv = MultivariateLogNormalDiag(ind_concs_fl,
            ind_d_concs_fl)

        # sample!
        # NOTE: here we need to manually log it back
        ind_vols_fl_sampled = tf.log(
            ind_vols_fl_rv.sample(n_samples))
        ind_concs_fl_sampled = np.log(
            ind_concs_fl_rv.sample(n_samples))

        # drop ref, in case it is used under high performace context
        del ind_vols_fl
        del ind_concs_fl
        del ind_d_vols_fl
        del ind_d_concs_fl
        del ind_vols_fl_rv
        del ind_concs_fl_rv

        # reshape back
        # NOTE: although this is not deterministically stable,
        #       it was validated to work
        # (n_samples, time, n_wells)
        ind_vols_sampled = tf.reshape(
            ind_vols_fl_sampled,
            (n_samples, time, n_wells))
        # (n_samples, time, 3, n_wells)
        ind_concs_sampled = tf.reshape(
            ind_concs_fl_sampled,
            (n_samples,
            time, 3, n_wells))

        # ================
        # calculating mean
        # ================

        # the cumulative volume is the sum of all the volumes previously
        # build a lower triangular identity matrix
        # shape = (time, time)
        tril_ones = tf.tile(
            tf.expand_dims(tf.linalg.band_part(tf.eye(time), -1, 0), 0),
            [n_samples, 1, 1])

        # calculate the cumulative volume, sampled
        # (n_samples, time, n_wells)
        vols_sampled = tf.matmul(tril_ones, ind_vols_sampled)
        # (time, n_wells)
        self.vols = tf.math.reduce_mean(vols_sampled, 0)

        # handle quantities
        # (n_samples, time, 3, n_wells)
        ind_qs = tf.multiply(
            tf.tile(tf.expand_dims(ind_vols_sampled, 2), [1, 1, 3, 1]),
            ind_concs_sampled)

        # we want to implement the following:
        # qs = tf.matmul(tril_ones, ind_qs)
        # but this is not supported by TensorFlow
        # (n_samples, time, 3, n_wells)
        qs = tf.Variable(
            tf.zeros([n_samples, time, 3, n_wells], dtype=tf.float32))
        idx = tf.constant(0)

        def loop_body(idx):
            qs[:, :, idx, :].assign(tf.matmul(tril_ones, ind_qs[:, :, idx, :]))
        tf.while_loop(
            lambda idx: tf.less(idx, 3),
            lambda idx: loop_body(idx),
            [idx])
        
        # average to calculate the concentrations
        # (n_samples, time, 3, n_wells)
        concs_sampled = tf.div(
            qs,
            tf.tile(tf.expand_dims(vols_sampled, 2), 3))
        # (time, 3, n_wells)
        self.concs = tf.math.reduce_mean(concs_sampled, 0)

        # ======================
        # calculating covariance
        # ======================
        # (time, time, n_wells)
        vols_cov = tf.Variable(
            tf.zeros((time, time, n_wells), dtype=tf.float32)) # initialize
        # NOTE: tf.while_loop is actually paralled
        idx = tf.constant(0)
        def loop_body(idx):
            vols_cov[:, :, idx].assign(cov(vols_sampled[:, :, idx]))
            return idx + 1
        tf.while_loop(
            lambda idx: tf.less(idx, n_wells),
            lambda idx: loop_body(idx),
            [idx])
        self.vols_cov = vols_cov

        # (time, time, 3, n_wells)
        concs_cov = tf.Variable(
            tf.zeros((time, time, 3, n_wells), dtype=tf.float32))
        idx0 = tf.constant(0)
        idx1 = tf.constant(0)
        def loop_body(idx0, idx1):
            concs_cov[:, :, idx0, idx1].assign(
                cov(concs_sampled[:, :, idx0, idx1]))
            return idx0 + 1, idx1 + 1
        tf.while_loop(
            lambda idx0, idx1: tf.logical_and(
                tf.less(idx0, 3),
                tf.less(idx1, n_wells)),
            lambda idx0, idx1 : loop_body(idx0, idx1),
            [idx0, idx1])
        self.concs_cov = concs_cov
