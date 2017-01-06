# Installing Warp with mpi4py on MIRA

This document describes how to install Warp on the MIRA cluster at
the ALCF facility, Argonne National Laboratory.
The installation requires the use of a special python interpreter "scalable-python"
and needs to be done by compiling the sources fo every package. Scalable-python
ensures very fast import times even on 100,000s of cores.

## Setting up the environnement

On MIRA, environment variables have to be set up in your `.soft` file located
in your `$HOME` directory. These environment variables will only be used for
compiling the different modules on the compute node. At run time, if you want
to use some of them, you have to pass them explicitly to the command `runjob` (see
  ALCF documentation for more info). Copy paste these lines to your `.soft` file:   


```
+cmake
+bgqtoolchain-gcc447
+mpiwrapper-mpich3-gcc.legacy
+mpiwrapper-xl.legacy
# User defined environment variables
PATH+=$HOME/scalable-python/installdir/bin/
@default
```

Then run `resoft` to enable those changes (or alternatively
log out from MIRA, and then log in again).

## Installing the scalable-python interpreter

- In your `$HOME` directory: clone the scalable-python repo: `git clone https://github.com/CSC-IT-Center-for-Science/scalable-python.git`
- `cd scalable-python/` and create a file `install_mira` with those lines:

```
#!/bin/bash
export CC=powerpc64-bgq-linux-gcc
export CXX=powerpc64-bgq-linux-g++
export MPICC=mpicc
export CCSHARED=-fPIC
export CFLAGS='-I/soft/libraries/alcf/20130312/xl/ZLIB/include'
export LDFLAGS='-L/soft/libraries/alcf/20130312/xl/ZLIB/lib'
export LINKFORSHARED='-Wl,--allow-multiple-definition -Xlinker -export-dynamic -dynamic'
export MPI_LINKFORSHARED='-Xlinker -export-dynamic -dynamic'
./configure --prefix=$HOME/scalable-python/installdir --with-zlib --enable-mpi --disable-ipv6 2>&1 | tee mira-conf
make 2>&1 | tee mira-make
make mpi 2>&1 | tee mira-make-mpi
make install 2>&1 | tee mira-inst
make install-mpi 2>&1 | tee mira-inst-mpi
```
- Finally type `chmod 700 install_mira; ./install_mira` to install the python
scalable interpreter.

## Install setuptools python module

- In your `$HOME` directory download the setuptools package by typing the following command:
`wget --no-check-certificate
https://pypi.python.org/packages/f7/94/eee867605a99ac113c4108534ad7c292ed48bf1d06dfe7b63daa51e49987/setuptools-28.0.0.tar.gz#md5=9b23df90e1510c7353a5cf07873dcd22`,

- Untar/zip the .tar.gz archive by typing: `tar -xzvf setuptools-28.0.0.tar.gz` and `cd setuptools-28.0.0/`
- create a file `install_mira` with those lines:

```
#!/bin/bash
export CC=powerpc64-bgq-linux-gcc
export PYTHONHOME=$HOME/scalable-python/installdir/
export PYTHON=${PYTHONHOME}/bin/python
buildir=build

rm -rf ${builddir}
${PYTHON} setup.py install 2>&1 | tee setuptools.log.mira
```
- Finally type `chmod 700 install_mira; ./install_mira` to install the python
package in site-packages

## Install the argparse python module

- In your `$HOME` directory download the argparse package by typing the following command:
`wget --no-check-certificate
https://pypi.python.org/packages/18/dd/e617cfc3f6210ae183374cd9f6a26b20514bbb5a792af97949c5aacddf0f/argparse-1.4.0.tar.gz#md5=08062d2ceb6596fcbc5a7e725b53746f`,

- Untar/zip the .tar.gz archive by typing: `tar -xzvf argparse-1.4.0.tar.gz` and `cd argparse-1.4.0/`
- create a file `install_mira` with those lines:

```
#!/bin/bash
export CC=powerpc64-bgq-linux-gcc
export PYTHONHOME=$HOME/scalable-python/installdir/
export PYTHON=${PYTHONHOME}/bin/python
buildir=build

rm -rf ${builddir}
${PYTHON} setup.py install  2>&1 | tee argparse.log.mira
```
- Finally type `chmod 700 install_mira; ./install_mira` to install the python
package in site-packages


## Install the numpy python module

- In your `$HOME` directory download the numpy package by typing the following command:
`wget --no-check-certificate
https://pypi.python.org/packages/3d/82/a8e9227167dca4301d4d7a61977a50d12cd98c277eb9035d7b78bc8b4a1f/numpy-1.10.2.tar.gz#md5=816518282f1617636aaf26e7cd9b127b`,

- Untar/zip the .tar.gz archive by typing: `tar -xzvf numpy-1.10.2.tar.gz` and `cd numpy-1.10.2/`
- create a file `install_mira` with those lines:

```
#!/bin/bash
export CC=powerpc64-bgq-linux-gcc
export BASECFLAGS="-m64 -fno-strict-aliasing"
export F77=powerpc64-bgq-linux-gfortran
export FFLAGS="-m64 -fdefault-integer-8"
export PYTHONHOME=$HOME/scalable-python/installdir/
export PYTHON=${PYTHONHOME}/bin/python
buildir=build

rm -rf ${builddir}

${PYTHON} setup.py config
${PYTHON} setup.py build
${PYTHON} setup.py install  2>&1 | tee numpy.log.mira
```
- Finally type `chmod 700 install_mira; ./install_mira` to install the python
package in site-packages

## Install Forthon

- In your `$HOME` directory :`git clone https://github.com/dpgrote/Forthon.git`
then `cd Forthon/`
- create a file `install_mira` with those lines:

```
#!/bin/bash
export CC=powerpc64-bgq-linux-gcc
export MPICC=mpicc
export PYTHONHOME=$HOME/scalable-python/installdir/
export PYTHON=${PYTHONHOME}/bin/python
buildir=build
rm -rf ${builddir}
${PYTHON} setup.py install 2>&1 | tee forthon.log.mira
```

- NB: for the moment, before installing Forthon, you need to comment the following lines in `setup.py`
(Forthon is assuming python2.7 by default and these lines are not compatible with python2.6):
```
try:
# --- In python3, check_output returns a byte string that needs to be decoded to get the string.
# --- The decode method is mostly harmless in python2.
#bcommithash = subprocess.check_output('git log -n 1 --
pretty=%h',stderr=subprocess.STDOUT,shell=True).strip()
# commithash = bcommithash.decode()
commithash = 'f3ecb8e'
except subprocess.CalledProcessError:
# --- This version was obtained from a non-git distrobution. Use the
# --- saved commit hash from the release.
# --- This is automatically
```

- Finally type `chmod 700 install_mira; ./install_mira` to install the python
package in site-packages

## Install mpi4py

- In your `$HOME` directory download the mpi4py package by typing the following command:
`wget --no-check-certificate
https://pypi.python.org/packages/ee/b8/f443e1de0b6495479fc73c5863b7b5272a4ece5122e3589db6cd3bb57eeb/mpi4py-2.0.0.tar.gz#md5=4f7d8126d7367c239fd67615680990e3`,

- Untar/zip the .tar.gz archive by typing: `tar -xzvf mpi4py-2.0.0.tar.gz` and `cd mpi4py-2.0.0/`
- create a file `install_mira` with those lines:

```
#!/bin/bash
export CC=mpicc
export MPICC=mpicc
export LDSHARED="mpicc -shared"
#export LD_LIBRARY_PATH=/bgsys/drivers/ppcfloor/gnu-linux/lib64
export PYTHONHOME=$HOME/scalable-python/installdir/
export PYTHON=${PYTHONHOME}/bin/python
buildir=build

rm -rf ${builddir}
${PYTHON} setup.py install 2>&1 | tee mpi4py.log.mira
```
- Finally type `chmod 700 install_mira; ./install_mira` to install the python
package in site-packages

## Install six

- In your `$HOME` directory download the six package by typing the following command:
`wget --no-check-certificate
https://pypi.python.org/packages/b3/b2/238e2590826bfdd113244a40d9d3eb26918bd798fc187e2360a8367068db/six-1.10.0.tar.gz#md5=34eed507548117b2ab523ab14b2f8b55`,

- Untar/zip the .tar.gz archive by typing: `tar -xzvf six-1.10.0.tar.gz` and `cd six-1.10.0/`
- create a file `install_mira` with those lines:

```
#!/bin/bash
export CC=powerpc64-bgq-linux-gcc
export MPICC=mpicc
export PYTHONHOME=$HOME/scalable-python/installdir/
export PYTHON=${PYTHONHOME}/bin/python
buildir=build

rm -rf ${builddir}
${PYTHON} setup.py install  2>&1 | tee six.log.mira
```
- Finally type `chmod 700 install_mira; ./install_mira` to install the python
package in site-packages

## Install nose

- In your `$HOME` directory download the nose package by typing the following command:
`wget --no-check-certificate
https://pypi.python.org/packages/58/a5/0dc93c3ec33f4e281849523a5a913fa1eea9a3068acfa754d44d88107a44/nose-1.3.7.tar.gz#md5=4d3ad0ff07b61373d2cefc89c5d0b20b`,

- Untar/zip the .tar.gz archive by typing: `tar -xzvf nose-1.3.7.tar.gz` and `cd nose-1.3.7/`
- create a file `install_mira` with those lines:

```
#!/bin/bash
export CC=powerpc64-bgq-linux-gcc
export MPICC=mpicc
export PYTHONHOME=$HOME/scalable-python/installdir/
export PYTHON=${PYTHONHOME}/bin/python
buildir=build

rm -rf ${builddir}
${PYTHON} setup.py install  2>&1 | tee nose.log.mira
```
- Finally type `chmod 700 install_mira; ./install_mira` to install the python
package in site-packages

## Install pkgconfig

- In your `$HOME` directory download the pkgconfig package by typing the following command:
`wget --no-check-certificate
https://pypi.python.org/packages/9d/ba/80910bbed2b4e646a6adab4474d2e506744c260c7002a0e6b41ef8750d8d/pkgconfig-1.2.2.tar.gz#md5=81a8f6ef3371831d081e03db39e09683`,

- Untar/zip the .tar.gz archive by typing: `tar -xzvf pkgconfig-1.2.2.tar.gz` and `cd pkgconfig-1.2.2/`
- create a file `install_mira` with those lines:

```
#!/bin/bash
export CC=powerpc64-bgq-linux-gcc
export MPICC=mpicc
export PYTHONHOME=$HOME/scalable-python/installdir/
export PYTHON=${PYTHONHOME}/bin/python
buildir=build

rm -rf ${builddir}
${PYTHON} setup.py install  2>&1 | tee pkgconfig.log.mira
```
- Finally type `chmod 700 install_mira; ./install_mira` to install the python
package in site-packages

## Install Cython

- In your `$HOME` directory download the Cython package by typing the following command:
`wget --no-check-certificate
https://pypi.python.org/packages/2f/ae/0bb6ca970b949d97ca622641532d4a26395322172adaf645149ebef664eb/Cython-0.25.1.tar.gz#md5=3c1541c15ba511645684a4eaca2cec0f`,

- Untar/zip the .tar.gz archive by typing: `tar -xzvf Cython-0.25.1.tar.gz` and `cd Cython-0.25.1/`
- create a file `install_mira` with those lines:

```
#!/bin/bash
export CC=powerpc64-bgq-linux-gcc
export MPICC=mpicc
export PYTHONHOME=$HOME/scalable-python/installdir/
export PYTHON=${PYTHONHOME}/bin/python
buildir=build

rm -rf ${builddir}
${PYTHON} setup.py install  2>&1 | tee cython.log.mira
```
- Finally type `chmod 700 install_mira; ./install_mira` to install the python
package in site-packages

## Install h5py

- In your `$HOME` directory download the mpi4py package by typing the following command:
`wget --no-check-certificate
https://pypi.python.org/packages/22/82/64dada5382a60471f85f16eb7d01cc1a9620aea855cd665609adf6fdbb0d/h5py-2.6.0.tar.gz#md5=ec476211bd1de3f5ac150544189b0bf4`,

- Untar/zip the .tar.gz archive by typing: `tar -xzvf h5py-2.6.0.tar.gz` and `cd h5py-2.6.0/`
- create a file `install_mira` with those lines:

```
#!/bin/bash
export CC=mpicc
export LDSHARED="mpicc -shared"
export HDF5_VERSION=10.1.0
export HDF5_DIR=/soft/libraries/hdf5/1.10.0/cnk-gcc/current/
export PYTHONHOME=$HOME/scalable-python/installdir/
export PYTHON=${PYTHONHOME}/bin/python
buildir=build

rm -rf ${builddir}
${PYTHON} setup.py build_ext --library-dirs="/soft/libraries/alcf/20130312/gcc/ZLIB/lib/" --libraries=z  2>&1 | tee h5py.log.mira
${PYTHON} setup.py install
```
- Finally type `chmod 700 install_mira; ./install_mira` to install the python
package in site-packages



## Installing Warp itself

- In your `$HOME` directory :`git clone https://bitbucket.org/berkeleylab/warp.git`
then `cd warp/pywarp90`
- create a file `Makefile.local.pympi` with this line:

```
FCOMP= -F gfortran --fcompexec mpif90 --fargs "-fPIC" --cargs "-fPIC"
```
- Before compiling: `export CC=mpicc; export LDSHARED="mpicc -shared -lmpifort"`
- To compile parallel version: `make pinstall`

## Running simulations

To build a batch submission script, Follow instructions at: `https://www.alcf.anl.gov/user-guides/running-jobs`. To run python scripts at large
scale you have to use your $HOME/scalable-python/installdir/bin/python_mpi
interpreter.
