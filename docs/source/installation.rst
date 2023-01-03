Installation
-------------

To install PCAT-DE you can clone the latest version on `Github <https://github.com/RichardFeder/pcat-de>`_. It should soon be possible to install PCAT-DE directly from pip, i.e.::
    
    pip install pcat-de
or by downloading the latest release and running::

    python setup.py install

Intel Math Kernel Library (MKL)
+++++++++++++++++++++++++++++++

MKL matrix multiplication routines can be used on Intel processors. Instructions for setting up MKL can be found `here <https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-mkl-for-dpcpp/top.html>`_.

OpenBLAS
++++++++

OpenBLAS is supported on AMD and Apple Silicon processors. To use OpenBLAS:

1) Install OpenBLAS through conda or pip:: 

    conda install openblas
or directly from Github::

      git clone https://github.com/xianyi/OpenBLAS

2) Make the library with ‘make’ from within OpenBLAS directory. It should automatically detect the processor for installation
3) Install the library::

    make PREFIX="desired directory" install
4) Compile the library with::

    gcc -shared -o pcat-lion-openblas.so -fPIC pcat-lion-openblas.c -L"desired directory path" -lopenblas
``-L[path]`` looks for the installed library in its path, and the ``-lopenblas`` searches for anything starting with “lib” that has “openblas” in it.
