
# Ignore this block -- it's for the documentation build
try:
    import os, sys
    sys.path.insert(1, os.path.abspath(r"D:\Users\Ganel\psi4conda\Scripts\ "))
except ImportError:
    pass

# This is the important part
import psi4



# Set the molecule and basis set
molecule = psi4.geometry("""
    Ag 0 0 0
""")
psi4.set_options({'basis': 'sto-3g'})

# Set the X-ray source energy
source_energy = 17.4 # keV, for Mo K-alpha X-ray
psi4.set_options({'reference': 'uhf', 'scf_type': 'pk', 'df_basis_scf': 'sto-3g', 'e_convergence': 1e-8})

# Perform the SCF calculation
scf_e, wfn = psi4.energy('scf', return_wfn=True)

# Extract the density matrix
D = psi4.core.Matrix.from_array(psi4.core.get_variable('Da'))

# Calculate the screening coefficient
eps_x = 1.0 / (1.0 + (4.0 / 3.0) * psi4.molecule.nuclear_repulsion_energy() * D.vector_dot(D) * (source_energy**2) / (2*psi4.constants.hartree2ev*psi4.constants.c**2) )

# Print the result
print("Screening coefficient for Ag with Mo K-alpha X-ray source: ", eps_x)