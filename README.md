# pcat-de
Probabilistic Cataloger (in the presence of) Diffuse Emission (PCAT-DE). This code builds off the probabilistic cataloging code Lion (Portillo+2017), and the implementation of PCAT-DE is described in Feder+(2022). 

Certain datasets may require a consistent treatment of both point-like and diffuse structured signals in order to maximize the information content (and recover appropriate uncertainties) of any individual component. The code incorporates both explicit spatial templates and a set of non-parametric Fourier component templates into PCAT's forward model.

The majority of the code is implemented in Python, with the exception of the image model and likelihood evaluations which use Intel's Math Kernel Library (MKL) in C.

Documentation (will soon) be located at pcat-de.readthedocs.io.

