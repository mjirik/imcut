# Compiler notes

For installation with `pip` on windows you will need compiler.

* Download and install [miniconda](http://conda.pydata.org/miniconda.html)
* Download and install [C++ Compiler for Windows](https://wiki.python.org/moin/WindowsCompilers) 
    * Python 2.7: [MS Visual C++ compiler for Python 2.7](http://aka.ms/vcpython27)
    * Python 3.6: [Microsoft Build Tools for Visual Studio 2017](
    https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017)
    
        Check "Python development" and "Desktop Application C++ development" during install. 
        You may remove all submodules. Keep just VC++ 2017 tools.
    
* Install depenencies
    ```bash
    conda install -c simpleitk -c mjirik -c conda-forge --file requirements_conda.txt
    ```
    
* Install `pygco`
    ```bash
    pip install pygco
    ```
* [Compilers for Windows](https://wiki.python.org/moin/WindowsCompilers)

* 


# Installation Instructions

Installing requires you to have installed:

  * numpy (http://www.numpy.org)
  * scipy (http://scipy.org)
  * scikit-learn (http://scikit-learn.org)
  * Cython - C-extension for Python (http://cython.org)
  * pyqt - Python bindings for Qt application framework
    (http://www.riverbankcomputing.com/software/pyqt)
  * git - distributed version control system (http://git-scm.com)
  * pygco - Graphcuts for Python (https://github.com/amueller/gco_python)
  * pydicom - package for working with DICOM files
    (http://code.google.com/p/pydicom)

On Linux, ask the package manager of your distribution.

On Windows, packages numpy, scipy, scikit-learn, Cython, pydicom are part of
Python(x,y) distribution (http://code.google.com/p/pythonxy).


# Pygco install

Pygco has to be installed manually, in gitbash write:

Linux:

  git clone https://github.com/amueller/gco_python.git
  cd gco_python
  make
  python setup.py install

Windows:

   git clone https://github.com/amueller/gco_python.git
   cd gco_python
   mkdir gco_src && cd gco_src
   curl -O http://vision.csd.uwo.ca/code/gco-v3.0.zip
   unzip gco-v3.0.zip
   cd ..
   curl -O https://raw2.github.com/mjirik/pyseg_base/master/distr/gco_python.pyx.patch
   patch gco_python.pyx < gco_python.pyx.patch
   python setup.py build_ext -i --compiler=mingw32
   python.exe setup.py build --compiler=mingw32
   python.exe setup.py install --skip-build

Mac:

   git clone https://github.com/amueller/gco_python.git
   cd gco_python
   mkdir gco_src
   cd gco_src
   curl -O http://vision.csd.uwo.ca/code/gco-v3.0.zip
   unzip gco-v3.0.zip
   curl -O https://raw.github.com/mjirik/pyseg_base/master/distr/energy.patch
   patch energy.h < energy.patch
   cd ..
   sudo python setup.py install



Install PYSEG_BASE:

  pip install imcut

  or:

  git clone https://github.com/mjirik/pyseg_base.git


Sample DICOM data can be found at:

  * http://www.mathworks.com/matlabcentral/fileexchange/2762-dicom-example-files?download=true
  * http://www.osirix-viewer.com/datasets/
