When picsar is compiled using full mode. (See picsar documentation), warp can take advantage
of additional features from picsar.
This includes Hybrid PSATD solver for Maxwell's equations and picsar absorbing boundary conditions.

Hybrid PSATD allows to solve Maxwell's equations using distributed memory FFT (right now picsar supports
p3dfft and FFTW_MPI) across multiple mpi subdomains in order to reduce the memory footprint and the computational time of the simulation. 

The idea is to divide the mpi tasks between different mpi groups, each mpi group forms a cartesian subdomain.
Then perform distributed memory fft across each group and excange the guardcells between the groups.
This will result in better memory footprint and less data redundancy as well as better performance.

Note that when using full_pxr = True, PMLS  computations are done in picsar.
Picsar PMLS are inside the simulation domain you set, (the first and last np_pml cells are set to be pml cells)
You might take that into consideration when designing you simulation box.


When using this mode, you need to turn full_pxr flag in your EM3DPXR to True.
Then you need to specify additional parameters:


em=EM3DPXR{.
	   .
	   .
           'full_pxr': False,
           'fftw_hybrid':False,
           'fftw_with_mpi':False,
           'fftw_mpi_transpose':False,
           'p3dfft_flag':False,
           'p3dfft_stride':False,
           'nb_group_x':0,
           'nb_group_y':0,
           'nb_group_z':0,
           'nyg_group':0,
           'nzg_group':0,
           'nx_pml':8,
           'ny_pml':8,
           'nz_pml':8,
           'g_spectral':False,
           }

when full_pxr is True this will allows matrix initializations for PSATD and PMLS compuations to be done in picsar.
To use hybrid PSATD you need to set fftw_with_mpi = True and fftw_hybrid = True, if fftw_hyrid = False, then picsar will solve Maxwell 
using whole domain ffts. Note that this mode is buggy and is has not been tested extensively. 
So you might want to set fftw_with_mpi = True, and fftw_hybrid = True to avoid inconvenient bugs.

Then you need to choose between between the fft library that performs the fft computations.
Right now you can choose between FFTW_MPI and p3dfft.
To choose p3dfft you need to set p3dfft_flag = True, otherwise warp will use FFTW_MPI by default.

Whether you chose fftw_mpi or p3dfft you might want to set fftw_mpi_transpose = True p3dfft_stride = True respectively.
This will allow to preform faster ffts since this will allow to get rid of one data transposition during the fft computation.
Note that when using p3dfft_stride = True, P3DFFT library needs to be compiled using --enable-stride1 
Also, when p3dfft_stride=False, you need to recompile p3dfft  WITHOUT --enable-stride1.


But we recommand to always use --enable-stride1 since this results in better performance.


The main difference between these two libraries is that fftw_mpi can gather mpi_tasks along one direction(the z axis)
while p3dfft can gather mpis along y and z axes.



Note that p3dfft CANNOT be used when doing 2d simulations since this library only performs 3D ffts.
Plus, fftw_mpi_transpose must be set to False when performing 2d simulations.


The next thing you need to set are nb_group_z and nb_group_y. Note that you can't choose nb_group_x as this parameter is only  defined for future implementations.

nb_group_y is the number of groups along the y direction.
nb_group_z is the number of groups along the z direction.

If you are using fftw_mpi you might nb_group_y since this will be set automatically to nprocy

Finally, you need to choose nyg_group = (number of guardcells along y for the groups)
and nzg_group = (number of guardcells along z  for the groups)

Note that when using the hybrid mode, you can set nzg_group, nyg_group(P3DFFT only) high enough to get very small truncation error and reduce nyguards, nzguards to the minimum value (nyguard+1,nzguard+1)
to avoid data redundancy
Note that nxg_group is set to nxguards automatically since task gathering along x axis is not supported.


If you are doing simulations using periodic boundaries you might want to set g_spectral = False, it will reduce the memory footprint of the EM solver.

In general you don't need to change the values of nx_pml, ny_pml ... since the default value brings good absorbtion rate.









