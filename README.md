# Davout
Repository to store my workhorses and research-themed park. Github link
to the repository and source files: https://github.com/Matheus-Janczkowski/Davout

1. Link to the booklet on installation of a miscelaneous of software: https://www.overleaf.com/read/wbxhncmtnmkm#0eb73b

2. Link to the booklet on python programming: https://www.overleaf.com/read/hcmfzzsrhndj#00fdb1

3. Link to the booklet on writing in LaTeX: https://www.overleaf.com/read/sdrvfrpdjhft#66d6f9

4. Link for the presentation template: https://www.overleaf.com/read/sbgxdphxswmm#6d7d64

# Citation
https://doi.org/10.5281/zenodo.18806245

# What does Davout do?
Workhorse for a research project that encompasses finite element analy-
sis, machine learning, and multiscale analysis. Additionally, a set of 
tools is provided.

This package contains extensive implementation of hyperelastic problems 
in large strains using FEniCS and GMSH for mesh generation. A consistent 
and general purpose implementation of ANN models using tensorflow is al-
so available. 

This suite performs:
1. Geometry and mesh generation
2. Finite element analysis
3. ANN models definition and training
4. Post-processing automation using ParaView
5. Generation of figures, collages, and slides using own graphical tools
6. LaTeX writing and formating using an ensemble of commands

# Installation using pip
pip install Davout

# Philosophy and aims
Davout aims to be a unified software to accompany researchers in compu-
tational mechanics. Davout sits on a profound love for python and open
software.

Davout stands on the shoulders of other massively mighty python packages,
such as FEniCS, TensorFlow, GMSH, ParaView, and matplotlib.

# Installation using installer file
Download the zip file, unzip it and move it to a suitable directory. 
Open the Davout folder, where setup.py is located. Open this path in
terminal (using a virtual environment) and run the following command

python davout_installer.py

# Installation using command
Download the repository, unzip the file, put it in a suitable place for you.
Activate a python virtual environment (follow instruction in the booklet 1. 
to create a virtual environment if you don't have one), go into the directory 
where the files are located through the virtual environment terminal. Then, 
type in terminal (instead of python you might need to explicitely type in
the version, like python3):

python setup.py bdist_wheel sdist

pip install .

To test the installation:

python

import Davout