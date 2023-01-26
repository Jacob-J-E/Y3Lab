import psi4

psi4.set_memory('2 GB')
psi4.set_num_threads(2)
psi4.set_output_file('output.dat', False)

# Define the silver atom using psi4.geometry
silver = psi4.geometry("""
Ag 0.0 0.0 0.0
""")

# Perform a DFT calculation
psi4.set_options({'basis': 'sto-3g', 'scf_type': 'pk', 'dft_spherical_points': 110, 'dft_radial_points': 50})
psi4_energy, psi4_wavefunction = psi4.energy('b3lyp', molecule=silver, return_wfn=True)

# Extract the screening coefficient
screening_coefficient = psi4_wavefunction.V_potential()[0][0][0]
print('Screening coefficient of silver:', screening_coefficient)