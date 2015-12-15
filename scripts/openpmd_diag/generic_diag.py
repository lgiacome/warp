"""
This file defines the generic class OpenPMDDiagnostic.

This class is a common class from which both ParticleDiagnostic
and FieldDiagnostic inherit
"""
import os
import datetime
import shutil
from dateutil.tz import tzlocal
import numpy as np

# Dictionaries of correspondance for openPMD
from data_dict import unit_dimension_dict

class OpenPMDDiagnostic(object) :
    """
    Generic class that contains methods which are common
    to both FieldDiagnostic and ParticleDiagnostic
    """

    def __init__(self, period, top, w3d, comm_world,
                 lparallel_output=False, write_dir=None ) :
        """
        General setup of the diagnostic

        Parameters
        ----------
        period : int
            The period of the diagnostics, in number of timesteps.
            (i.e. the diagnostics are written whenever the number
            of iterations is divisible by `period`)

        top : the object representing the `top` package in Warp
            Contains information on the time.

        w3d : the object representing the `w3d` package in Warp
            Contains the dimensions of the grid.
        
        comm_world : a communicator object
            Either an mpi4py or a pyMPI object, or None (single-proc)

        lparallel_output : boolean, optional
            Switch to set output mode (parallel or gathering)
            If "True" : Parallel output
            
        write_dir : string, optional
            The POSIX path to the directory where the results are
            to be written. If none is provided, this will be the path
            of the current working directory
        """    
        # Get the rank of this processor
        if comm_world is not None :
            self.rank = comm_world.rank
        else :
            self.rank = 0

        # Register the arguments
        self.top = top
        self.w3d = w3d
        self.period = period
        self.comm_world = comm_world
        self.lparallel_output = lparallel_output
            
        # Get the directory in which to write the data
        if write_dir is None :
            self.write_dir = os.getcwd()
        else :
            self.write_dir = os.path.abspath(write_dir)

        # Create a few addiditional directories within self.write_dir
        self.create_dir("")
        self.create_dir("diags")
        # If the directory hdf5 exists, remove it. (Otherwise the code
        # may crash if the directory hdf5 contains preexisting files.)
        self.create_dir("diags/hdf5", remove_existing=True)

    def write( self ) :
        """
        Check if the data should be written at this iteration
        (based on self.period) and if yes, write it.

        The variable top.it should be defined in the Python
        environment, and should represent the total number of
        timesteps in the simulation.
        """        
        # Check if the fields should be written at this iteration
        if self.top.it % self.period == 0 :

            # Write the hdf5 file if needed
            self.write_hdf5( self.top.it )
        
    def create_dir( self, dir_path, remove_existing=False ) :
        """
        Check whether the directory exists, and if not create it.
    
        Parameter
        ---------
        dir_path : string
           Relative path from the directory where the diagnostics
           are written
        """
        # The following operations are done only by the first processor.
        if self.rank == 0 :
            
            # Get the full path
            full_path = os.path.join( self.write_dir, dir_path )
        
            # Check wether it exists, and create it if needed
            if os.path.exists(full_path) and remove_existing==True:
                shutil.rmtree(full_path)
            if os.path.exists(full_path) == False :
                try:
                    os.makedirs(full_path)
                except OSError :
                    pass

    def setup_openpmd_file( self, f, iteration, time ) :
        """
        Sets the attributes of the hdf5 file, that comply with OpenPMD
    
        Parameter
        ---------
        f : an h5py.File object

        iteration: int
            The present iteration

        time: float
            The physical time of the present iteration
        """
        # Set the attributes of the HDF5 file
    
        # General attributes
        f.attrs["openPMD"] = np.string_("1.0.0")
        f.attrs["openPMDextension"] = np.uint32(1)
        f.attrs["software"] = np.string_("warp")
        f.attrs["softwareVersion"] = np.string_("4")
        today = datetime.datetime.now()
        f.attrs["date"] = np.string_(
            datetime.datetime.now(tzlocal()).strftime('%Y-%m-%d %H:%M:%S %z'))
        f.attrs["meshesPath"] = np.string_("fields/")
        f.attrs["particlesPath"] = np.string_("particles/")
        f.attrs["iterationEncoding"] = np.string_("fileBased")
        f.attrs["iterationFormat"] =  np.string_("data%T.h5")

        # Setup the basePath
        base_path = "/data/%d/" %iteration
        f.attrs["basePath"] = np.string_(base_path)
        bp = f.require_group( base_path )
        bp.attrs["time"] = time
        bp.attrs["dt"] = iteratiion
        bp.attrs["timeUnitSI"] = 1.
        
    def setup_openpmd_record( self, dset, quantity ) :
        """
        Sets the attributes of a record, that comply with OpenPMD
    
        Parameter
        ---------
        dset : an h5py.Dataset or h5py.Group object
        
        quantity : string
           The name of the record considered
        """
        dset.attrs["unitDimension"] = unit_dimension_dict[quantity]
        # timeOffset is set to zero (approximation)
        dset.attrs["timeOffset"] = 0.        
        
    def setup_openpmd_component( self, dset ) :
        """
        Sets the attributes of a component, that comply with OpenPMD
    
        Parameter
        ---------
        dset : an h5py.Dataset or h5py.Group object
        """
        dset.attrs["unitSI"] = 1.

    
