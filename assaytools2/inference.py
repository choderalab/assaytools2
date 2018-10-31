# imports
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import copy
from tensorflow_probability import edward2 as ed
from utils import *
import titration
from bindingmodels import TwoComponentBindingModel

def make_model(well_complex, well_ligand,
               fi_complex, fi_ligand):

    """
    Build a tfp model for an assay that consists of N wells of protein:ligand
    at various concentrations and an additional N wells of ligand in buffer,
    with the ligand at the same concentrations as the corresponding protein:ligand wells.

    Parameters:
    well_complex : a titration.SingleWell object, models the well in which the fluorescense
                   of the complex is measured.
    well_ligand : a titration.SingleWell object, models the well in which the fluorescense
                    of the ligand is measured.
    fi_complex : np.array, shape = (N, ). the fluorescense intensity of the complex.
    fi_ligand : np.array, shape = (N, ). the fluorescense intensity of the ligand.


    """

    n_wells = fi_complex.shape[0]

    # define max and min fluorescense intensity
    fi_max = np.max([np.max(fi_complex), np.max(fi_ligand)])
    fi_min = np.min([np.min(fi_complex), np.min(fi_ligand)])

    # grab the path_length from the wells
    assert well_complex.path_length == well_ligand.path_length
    path_length = tf.constant(well_complex.path_length, dtype=tf.float32)

    # grab the concentrations rv from the complex well
    concs_p_complex_rv = well_complex.concs_p_rv
    concs_l_complex_rv = well_complex.concs_l_rv

    # grab the concentrations rv from the ligand plate
    concs_l_ligand_rv = well_ligand.concs_l_rv

    # the guesses, to be used as initial state
    delta_g_guess = tf.constant(-12.5, dtype=tf.float32) # kT # use the value from ChEMBL
    concs_p_complex_guess = tf.constant(well_complex.concs[0, :], dtype=tf.float32)
    concs_l_complex_guess = tf.constant(well_complex.concs[1, :], dtype=tf.float32)
    concs_l_ligand_guess = tf.constant(well_ligand.concs[1, :], dtype=tf.float32)
    fi_pl_guess = tf.constant(np.true_divide(fi_max - fi_min,
                                np.min([np.max(concs_p_complex_guess), np.max(concs_l_complex_guess)])), dtype=tf.float32)
    fi_p_guess = tf.constant(fi_min, dtype=tf.float32)
    fi_l_guess = tf.constant(np.true_divide(fi_max-fi_min, np.max(concs_l_complex_guess)), dtype=tf.float32)
    fi_plate_guess = tf.constant(fi_min, dtype=tf.float32)
    fi_buffer_guess = tf.constant(np.true_divide(fi_min, path_length), dtype=tf.float32)
    fi_complex_guess = tf.constant(fi_complex, dtype=tf.float32)
    fi_ligand_guess = tf.constant(fi_ligand, dtype=tf.float32)
    jeffrey_log_sigma_complex_guess = tf.constant(np.tile([0.5 * (fi_max-10)], (n_wells,)), dtype=tf.float32)
    jeffrey_log_sigma_ligand_guess = tf.constant(np.tile([0.5 * (fi_max-10)], (n_wells,)), dtype=tf.float32)

    #======================================================================
    # Define a whole bunch of rv
    #======================================================================

    # TODO: not sure which way is faster:
    # define rv inside or outside this function?



    # define free energy prior
    delta_g_rv = tfd.Uniform(low=tf.log(tf.constant(1e-15, dtype=tf.float32)), high=tf.constant(0.0, dtype=tf.float32))

    # define fluorescense intensity prior
    fi_plate_rv = tfd.Uniform(low=tf.constant(0.0, dtype=tf.float32), high=tf.constant(fi_max, dtype=tf.float32))
    fi_buffer_rv = tfd.Uniform(low=tf.constant(0.0, dtype=tf.float32), high=tf.constant(np.true_divide(fi_max, path_length), dtype=tf.float32))

    fi_pl_rv = tfd.Uniform(low=tf.constant(0.0, dtype=tf.float32), high=tf.constant(2*np.max([np.max(np.true_divide(fi_max, concs_p_complex_guess)),
                                                  np.max(np.true_divide(fi_max, concs_l_complex_guess))]), dtype=tf.float32))
    fi_p_rv = tfd.Uniform(low=tf.constant(0.0, dtype=tf.float32), high=tf.constant(2*np.max(np.true_divide(fi_max, concs_p_complex_guess)), dtype=tf.float32))
    fi_l_rv = tfd.Uniform(low=tf.constant(0.0, dtype=tf.float32), high=tf.constant(2*np.max(np.true_divide(fi_max, concs_l_complex_guess)), dtype=tf.float32))

    jeffrey_log_sigma_rv = tfd.Uniform(low=tf.constant(np.tile([-10.0], (n_wells,)), dtype=tf.float32), high=tf.constant(np.tile([np.log(fi_max)], (n_wells,)), dtype=tf.float32))


    # define the joint_log_prob function to be used in MCMC
    def joint_log_prob(delta_g, # primary parameters to INFER
                       fi_plate, fi_buffer,
                       fi_pl, fi_p, fi_l, # fluorescence intesensity to INFER
                       fi_complex, fi_ligand,
                       concs_p_complex, concs_l_complex, # primary parameters to INFER
                       concs_l_ligand, # primary parameters to INFER
                       jeffrey_log_sigma_complex, jeffrey_log_sigma_ligand): # TODO: figure out a way to get rid of this


        #======================================================================
        # Calculate the relationships between the observed values,
        # the values to infer, and the constants.
        #======================================================================

        # using a binding model, get the true concentrations of protein, ligand,
        # and protein-ligand complex


        concs_p_, concs_l_, concs_pl_ = TwoComponentBindingModel.equilibrium_concentrations_tf(
            delta_g, concs_p_complex, concs_l_complex)

        # predict observed fluorescence intensity
        fi_complex_ = fi_p * concs_p_ + fi_l * concs_l_ + fi_pl * concs_pl_ + path_length * fi_buffer + fi_plate
        fi_ligand_ = fi_l * concs_l_ligand + path_length * fi_buffer + fi_plate


        # make this rv inside the function, since it changes with jeffery_log_sigma
        fi_complex_rv = tfd.Normal(loc=tf.constant(fi_complex, dtype=tf.float32), scale=tf.square(tf.exp(jeffrey_log_sigma_complex)))
        fi_ligand_rv = tfd.Normal(loc=tf.constant(fi_ligand, dtype=tf.float32), scale=tf.square(tf.exp(jeffrey_log_sigma_ligand)))


        #======================================================================
        # Sum up the log_prob.
        #======================================================================
        # NOTE: this is very weird. the LogNormal offered by tfp
        # is actually just normal distribution but with transformation before input
        # so you have to transfer again yourself



        log_prob = (delta_g_rv.log_prob(delta_g) # initialize a log_prob
                 + fi_plate_rv.log_prob(fi_plate)
                 + fi_buffer_rv.log_prob(fi_buffer)
                 + fi_pl_rv.log_prob(fi_pl)
                 + fi_p_rv.log_prob(fi_p)
                 + fi_l_rv.log_prob(fi_l)
                 + tf.reduce_sum(fi_complex_rv.log_prob(fi_complex_))
                 + tf.reduce_sum(fi_ligand_rv.log_prob(fi_ligand_))
                 + concs_p_complex_rv.log_prob(tf.log(concs_p_complex))
                 + concs_l_complex_rv.log_prob(tf.log(concs_l_complex))
                 + concs_l_ligand_rv.log_prob(tf.log(concs_l_ligand))
                 + tf.reduce_sum(jeffrey_log_sigma_rv.log_prob(jeffrey_log_sigma_complex))
                 + tf.reduce_sum(jeffrey_log_sigma_rv.log_prob(jeffrey_log_sigma_ligand)))




        return log_prob


    # put the log_prob function and initial guesses into a mcmc chain
    chain_states, kernel_results = tfp.mcmc.sample_chain(
        num_results=int(10e3),
        num_burnin_steps=int(1e3),
        parallel_iterations=10,
        current_state = [delta_g_guess,
                           fi_plate_guess, fi_buffer_guess,
                           fi_pl_guess, fi_p_guess, fi_l_guess,
                           fi_complex_guess, fi_ligand_guess,
                           concs_p_complex_guess, concs_l_complex_guess,
                           concs_l_ligand_guess,
                           jeffrey_log_sigma_complex_guess, jeffrey_log_sigma_ligand_guess],

        kernel=tfp.mcmc.TransformedTransitionKernel(
            bijector = [tfp.bijectors.Identity() for dummy_idx in range(13)],
            inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=joint_log_prob,
            num_leapfrog_steps=2,
            step_size=tf.Variable(1.),
            step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy()
            )))

    return chain_states, kernel_results
