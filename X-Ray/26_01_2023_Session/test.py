import xraydb

ka1 = 0
kb1 = 0
for name, line in xraydb.xray_lines('Ag', 'K').items():
    if name == 'Ka1':
        ka1 = line.energy
    elif name == 'Kb1':
        kb1 = line.energy


print(f'{ka1},{kb1}')