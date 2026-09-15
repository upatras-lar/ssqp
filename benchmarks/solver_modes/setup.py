from os import mkdir
from os.path import isdir

directories = [
    "trajgen/data",
    "trajgen/data/results",
    "trajgen/data/plots",
    "mpc/data",
    "mpc/data/plots",
]

for d in directories:
    if not isdir(d):
        mkdir(d)
