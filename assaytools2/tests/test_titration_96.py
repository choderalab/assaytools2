"""
test_titration_96.py
"""

import pytest

from ..titration_96 import *
import numpy
import numpy.testing as npt

# ========
# test cov
# ========
def test_cov():
    X = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    npt.assert_almost_equal(
        cov(X),
        [[1.0, 1.0], [1.0, 1.0]])

# =============
# test Solution
# =============
def test_create_empty_solution():
    """ empty solution
    """
    empty_solution = Solution()
    npt.assert_almost_equal(empty_solution.concs.numpy(), 0.0)
    npt.assert_almost_equal(empty_solution.d_concs.numpy(), 0.0)

def test_simple_solution():
    """ simple solution
    """
    simple_solution = Solution(
        conc_l=1.0, d_conc_l=0.05,
        conc_p=2.0, d_conc_p=0.01)

    npt.assert_almost_equal(
        simple_solution.concs, [2.0, 1.0, 0.0])

    npt.assert_almost_equal(
        simple_solution.d_concs, [0.01, 0.05, 0.0])

# ==========
# test Plate
# ==========
@pytest.fixture
def ligand_stock():
    return Solution(conc_l=1.0, d_conc_l=0.05)

@pytest.fixture
def protein_stock():
    return Solution(conc_p=2.0, d_conc_p=0.01)

@pytest.fixture
def complex_stock():
    return Solution(
        conc_l=1.0, d_conc_l=0.05,
        conc_p=2.0, d_conc_p=0.01)

def test_empty_plate():
    empty_plate = Plate(96)
    npt.assert_almost_equal(empty_plate.n_wells, 96)

@pytest.fixture
def plate():
    return Plate(96)

def test_inject(plate, ligand_stock, protein_stock, complex_stock):
    """ inject some random stuff into different wells on our plate
    """
    # inject only ligand once
    plate.inject(ligand_stock, 0, 1.0, 0.02)
    npt.assert_almost_equal(
        plate.ind_vols[1, 0], 1.0)
    npt.assert_almost_equal(
        plate.ind_d_vols[1, 0], 0.02)
    npt.assert_almost_equal(
        plate.ind_concs[1, :, 0], [0.0, 1.0, 0.0])
    npt.assert_almost_equal(
        plate.ind_d_concs[1, :, 0], [0.0, 0.05, 0.0])

    # and twice
    plate.inject(ligand_stock, 1, 0.5, 0.02)
    # in the newly-created time series,
    # the independent volume at the 1st well should be altered
    npt.assert_almost_equal(
        plate.ind_vols[2, 1], 0.5)
    npt.assert_almost_equal(
        plate.ind_d_vols[2, 1], 0.02)
    npt.assert_almost_equal(
        plate.ind_concs[2, :, 1], [0.0, 1.0, 0.0])
    npt.assert_almost_equal(
        plate.ind_d_concs[2, :, 1], [0.0, 0.05, 0.00])

    # but ind_vols[1, 0] should remain unchanged
    npt.assert_almost_equal(
        plate.ind_vols[1, 0], 1.0)
    npt.assert_almost_equal(
        plate.ind_d_vols[1, 0], 0.02)
    npt.assert_almost_equal(
        plate.ind_concs[1, :, 0], [0.0, 1.0, 0.0])
    npt.assert_almost_equal(
        plate.ind_d_concs[1, :, 0], [0.0, 0.05, 0.00])

    # and once in the old well
    plate.inject(ligand_stock, 0, 0.5, 0.02)
    npt.assert_almost_equal(
        plate.ind_vols[3, 0], 0.5)
    npt.assert_almost_equal(
        plate.ind_d_vols[3, 0], 0.02)
    npt.assert_almost_equal(
        plate.ind_concs[3, :, 0], [0.0, 1.0, 0.0])
    npt.assert_almost_equal(
        plate.ind_d_concs[3, :, 0], [0.0, 0.05, 0.00])

    # inject protein in the first well
    plate.inject(protein_stock, 0, 0.5, 0.02)
    npt.assert_almost_equal(
        plate.ind_vols[4, 0], 0.5)
    npt.assert_almost_equal(
        plate.ind_d_vols[4, 0], 0.02)
    npt.assert_almost_equal(
        plate.ind_concs[4, :, 0], [2.0, 0.0, 0.0])
    npt.assert_almost_equal(
        plate.ind_d_concs[4, :, 0], [0.01, 0.0, 0.0])

    # inject complex solution in the first well
    plate.inject(complex_stock, 0, 0.5, 0.02)
    npt.assert_almost_equal(
        plate.ind_vols[5, 0], 0.5)
    npt.assert_almost_equal(
        plate.ind_d_vols[5, 0], 0.02)
    npt.assert_almost_equal(
        plate.ind_concs[5, :, 0], [2.0, 1.0, 0.0])
    npt.assert_almost_equal(
        plate.ind_d_concs[5, :, 0], [0.01, 0.05, 0.0])

def test_sample(plate, ligand_stock, protein_stock, complex_stock):
    """ Test the sampling process.
    """

    plate.inject(ligand_stock, 0, 1.0, 0.02)
    plate.sample()
