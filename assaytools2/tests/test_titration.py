# test_titration.py

# =======
# imports
# =======
import pytest
import titration

def test_building_solution():
    protein_stock = Solution(conc_p = 1e-5, d_conc_p = 1e-3)
    protein_stock = Solution(conc_p = 0, d_conc_p = 0)
    protein_stock = Solution(conc_p = 1, d_conc_0 = 1)

    
