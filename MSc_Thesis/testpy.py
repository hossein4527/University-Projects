import os
li = []
with os.scandir('/home/hossein4527/MEGAsync/python/Complex projects/Ising/Final/Lattice_size 4/') as entries:
    for entry in entries:
        li.append(entry.name)
h = 'hello'
print(h[-3:])