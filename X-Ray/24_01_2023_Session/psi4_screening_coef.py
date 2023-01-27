import psi4

psi4.set_memory('500 MB')

h2o = psi4.geometry("""
O 3
C
H 1 {0}
H 1 {0} 2 {1}
""")

psi4.energy('scf/cc-pvdz')