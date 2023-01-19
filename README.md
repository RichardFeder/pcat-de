# pcat-de
Probabilistic Cataloger (in the presence of) Diffuse Emission (PCAT-DE). This code builds off the probabilistic cataloging code [Lion](https://github.com/stephenportillo/lion) [(Portillo+2017)](https://iopscience.iop.org/article/10.3847/1538-3881/aa8565/pdf), and the implementation of PCAT-DE is described in Feder+(2022). 

Certain datasets may require a consistent treatment of both point-like and diffuse structured signals in order to maximize the information content (and recover appropriate uncertainties) of the individual components. This code incorporates both explicit spatial templates and a set of non-parametric Fourier component templates as additional ingredients in PCAT's forward model.

The majority of the code is implemented in Python, with the exception of the image model/likelihood evaluations which are written in C. The C code optionally uses Intel's [Math Kernel Library](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-mkl-for-dpcpp/top.html) (MKL), but OpenBLAS and BLAS are available options as well if running on a non-Intel processor.

Documentation can be found on [Read the Docs](https://pcat-de.readthedocs.io).

