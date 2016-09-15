# Installing Warp with mpi4py on Cori

This document describes how to install Warp on the Cori cluster.

## Using Shifter

Because of difficulties with the compilation of Warp on Cori, it is
currently advised to use
[Shifter](http://www.nersc.gov/research-and-development/user-defined-images/)
to run Warp on Cori. Shifter handles Linux container (similar to
Docker), and allows to easily port codes from one
architecture to another.

Shifter should **not** use the installation of Warp which is on the
system itself. Therefore, if you previously compiled and installed
Warp in `$SCRATCH/warp_install`, please remove this directory to avoid
errors.
```
rm -rf $SCRATCH/warp_install
```

In addition, make sure that your `$PATH` and `$PYTHONPATH` variables
are unmodified, and that no module is loaded.
In particular, if you used to modify these variables
in `.bashrc.ext`, please remove or comment out the corresponding
lines, as in the following example (note that the lines are commented out):

```
# if [ "$NERSC_HOST" == "cori" ]
# then
#  module swap PrgEnv-intel PrgEnv-gnu
#	module load python/2.7-anaconda
#	module load mpi4py
#  export PATH=$SCRATCH/warp_install/bin:$PATH
#	export PYTHONPATH=$SCRATCH/warp_install/lib/python:$PYTHONPATH
# fi
```


## Running simulations

In order to run a simulation, create a new directory,
copy your Warp input script to this directory, and rename this script
to `warp_script.py`. (The folder `scripts/examples/` of the
[Warp repository](https://bitbucket.org/berkeleylab/warp/src) contains
several examples of input scripts.)

Then create a submission script named `submission_script`. Here is an
example of a typical submission script.
```
#!/bin/bash -l
#SBATCH --job-name=test_simulation
#SBATCH --time=00:30:00
#SBATCH -n 32
#SBATCH --partition=debug
#SBATCH -e test_simulation.err
#SBATCH -o test_simulation.out
#SBATCH --image=docker:rlehe/warp:latest
#SBATCH --volume=<your$SCRATCH>:/home/warp_user/run

export mydir="$SCRATCH/test_simulation"

rm -fr $mydir
mkdir -p $mydir

cd $SLURM_SUBMIT_DIR

cp ./* $mydir/.
cd $mydir

shifter cd test_simulation
shifter mpirun -np 32 -launcher ssh python -i warp_script.py -p 2 1 16
```
Note that the options `--image=docker:rlehe/warp:latest`, `--
volume=<your$SCRATCH>:/home/warp_user/run` and `-launcher ssh` are essential
and should be copied exactly (**do not** replace `warp_user` or
`rlehe` by your username), with the exception of `<your$SCRATCH>`,
which should be replaced by the full path to your SCRATCH directory.

Then submit the simulation by typing `sbatch submission_script`.  The
progress of the simulation can be seen by typing ```squeue -u `whoami` ```. 
