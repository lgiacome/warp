"""
This file defines the class BoostedParticleDiagnostic

Major features:
- The class reuses the existing methods of FieldDiagnostic
  as much as possible, through class inheritance
- The class implements memory buffering of the slices, so as
  not to write to disk at every timestep

Remaining questions:
- Should one use the IO collectives when only a few proc modify a given file?
- Should we just have the proc writing directly to the file ?
  Should we gather on the first proc ?
- Is it better to write all the attributes of the openPMD file
  with only one proc ?
"""
import os
import numpy as np
import time
import h5py
from scipy.constants import c
from particle_diag import ParticleDiagnostic
from data_dict import z_offset_dict
from parallel import gatherarray
import pdb

class BoostedParticleDiagnostic(ParticleDiagnostic):
	def __init__(self, zmin_lab, zmax_lab, v_lab, dt_snapshots_lab,
				 Ntot_snapshots_lab, gamma_boost, period, em, top, w3d,
				 comm_world=None, particle_data=["position", "momentum", "weighting"],
				 select=None, write_dir=None, lparallel_output=False,
				 species = {"electrons": None}):
		
		"""
		Initialize diagnostics that retrieve the data in the lab frame,
			as a series of snapshot (one file per snapshot),
			within a virtual moving window defined by zmin_lab, zmax_lab, v_lab.
					 
			Parameters
			----------
			zmin_lab, zmax_lab: floats (meters)
				Positions of the minimum and maximum of the virtual moving window,
				*in the lab frame*, at t=0

			v_lab: float (m.s^-1)
				Speed of the moving window *in the lab frame*

			dt_snapshots_lab: float (seconds)
				Time interval *in the lab frame* between two successive snapshots

			Ntot_snapshots_lab: int
				Total number of snapshots that this diagnostic will produce

			period: int
				Number of iterations for which the data is accumulated in memory,
				before finally writing it to the disk. 
				
			See the documentation of ParticleDiagnostic for the other parameters

		"""
		
		if write_dir is None:
			write_dir='lab_diags'
		
		#initialize Particle diagnostic normal attributes
		ParticleDiagnostic.__init__(self, period, top, w3d, comm_world,
			species, particle_data, select, write_dir, lparallel_output)

		# Register the boost quantities
		self.gamma_boost    = gamma_boost
		self.inv_gamma_boost= 1./gamma_boost
		self.beta_boost     = np.sqrt( 1. - self.inv_gamma_boost**2 )
		self.inv_beta_boost = 1./self.beta_boost

		# Find the z resolution and size of the diagnostic *in the lab frame*
		# (Needed to initialize metadata in the openPMD file)
		dz_lab              = c*self.top.dt * self.inv_beta_boost*self.inv_gamma_boost
		Nz                  = int( (zmax_lab - zmin_lab)/dz_lab )
		self.inv_dz_lab     = 1./dz_lab
		
		# Create the list of LabSnapshot objects
		self.snapshots      = []
		self.species        = species

		# Record the time it takes
		measured_start      = time.clock()
		print('\nInitializing the lab-frame diagnostics: %d files...' %(
			Ntot_snapshots_lab) )
		# Loop through the lab snapshots and create the corresponding files
		for i in range( Ntot_snapshots_lab ):
			t_lab   = i * dt_snapshots_lab/Ntot_snapshots_lab
			snapshot= LabSnapshot( t_lab,
									zmin_lab + v_lab*t_lab,
									top.dt,
									zmax_lab + v_lab*t_lab,
									self.write_dir, i ,self.species)
			self.snapshots.append( snapshot )
			# Initialize a corresponding empty file
			self.create_file_empty_slice( snapshot.filename, i,
				snapshot.t_lab, self.top.dt )

		# Print a message that records the time for initialization
		measured_end= time.clock()
		print('Time taken for initialization of the files: %.5f s' %(
			measured_end-measured_start) )

		self.ParticleCatcher= ParticleCatcher(self.gamma_boost, self.beta_boost,top)
		self.ParticleCatcher.initialize_pevious_instant()

	def write( self ):
		
		"""
		Redefines the method write of the parent class ParticleDiagnostic

		Should be registered with installafterstep in Warp
		"""
		# At each timestep, store a slice of the particles in memory buffers 

		self.store_snapshot_slices()
		#pdb.set_trace()
		# Every self.period, write the buffered slices to disk 
		#print snapshot.buffered_slices
		
		if self.top.it % self.period == 0:
			self.flush_to_disk()
		
	def store_snapshot_slices( self ):
		"""
		Store slices of the particles in the memory buffers of the
		corresponding lab snapshots
		"""

		# Loop through the labsnapshots
		for snapshot in self.snapshots:

			# Update the positions of the output slice of this snapshot
			# in the lab and boosted frame (current_z_lab and current_z_boost)
			snapshot.update_current_output_positions( self.top.time,
							self.inv_gamma_boost, self.inv_beta_boost )
			self.ParticleCatcher.zboost	  = snapshot.current_z_boost
			self.ParticleCatcher.zboost_prev= snapshot.prev_z_boost
			self.ParticleCatcher.zlab     = snapshot.current_z_lab
			self.ParticleCatcher.zlab_prev = snapshot.prev_z_lab
			# For this snapshot:
			# - check if the output position *in the boosted frame*
			#   is in the current local domain
			# - check if the output position *in the lab frame*
			#   is within the lab-frame boundaries of the current snapshot
			for species_name in self.species_dict:
				species     =self.species_dict[species_name]
				slice_array = self.ParticleCatcher.extract_slice(species)
				snapshot.register_slice( slice_array,species_name)
		return 

	def flush_to_disk( self):
		"""
		Writes the buffered slices of particles to the disk

		Erase the buffered slices of the LabSnapshot objects
		"""
		# Loop through the labsnapshots and flush the data
		
		for snapshot in self.snapshots:
			
			# Compact the successive slices that have been buffered
			# over time into a single array
			for species_name in self.species_dict:
				particle_array = snapshot.compact_slices(species_name)
				
				# Erase the memory buffers
				snapshot.buffered_slices[species_name] = []
				# Write this array to disk (if this snapshot has new slices)
				if len(particle_array[0])!=0:
					
					self.write_slices( particle_array, species_name,
						snapshot, self.ParticleCatcher.particle_to_index )

	def write_boosted_dataset(self, species_grp,  path, data, quantity):
		dset = species_grp[path]
		index=dset.shape[0]
		dset.resize(index+len(data),axis=0)
		dset [index:] = data

	def write_slices( self, particle_array, species_name, snapshot, p2i ): 
		"""
		For one given snapshot, write the slices of the
		different fields to an openPMD file

		Parameters
		----------
		particle_array: array of reals
			Array of shape
			- (10, num_part, nslices) 

		iz_min, iz_max: integers
			The indices between which the slices will be written
			iz_min is inclusice and iz_max is exclusive

		snapshot: a LabSnaphot object

		p2i: dict
			Dictionary of correspondance between the particle quantities
			and the integer index in the particle_array
		"""
		# Open the file without parallel I/O in this implementation
		
		f			= self.open_file( snapshot.filename )
		
		particle_path= "/data/%d/particles/%s" %(snapshot.iteration,species_name)
		species_grp = f[particle_path]
		# Loop over the different quantities that should be written
		for particle_var in self.particle_data:
			# Scalar field
			if particle_var == "position":
				for coord in ["x","y","z"]:
					quantity= coord
					path	= "%s/%s" %(particle_var, quantity)
					data	= particle_array[ p2i[ quantity ] ]
					self.write_boosted_dataset( species_grp, path,data,quantity)
	 
			elif particle_var == "momentum":
				for coord in ["x","y","z"]:
					quantity= "u%s" %coord
					path	= "%s/%s" %(particle_var,quantity)
					data	= particle_array[ p2i[ quantity ] ]
					self.write_boosted_dataset( species_grp ,path,data,quantity)
				
			elif particle_var == "weighting":
			   quantity= "w"
			   path	= 'weighting'
			   data	= particle_array[ p2i[ quantity ] ]
			   self.write_boosted_dataset( species_grp, path,data,quantity)
			

		# Close the file
		f.close()


class LabSnapshot:
	"""
	Class that stores data relative to one given snapshot
	in the lab frame (i.e. one given *time* in the lab frame)
	"""
	def __init__(self, t_lab, zmin_lab, dt, zmax_lab, write_dir, i, species_dict):
		"""
		Initialize a LabSnapshot 

		Parameters
		----------
		t_lab: float (seconds)
			Time of this snapshot *in the lab frame*
			
		zmin_lab, zmax_lab: floats
			Longitudinal limits of this snapshot

		write_dir: string
			Absolute path to the directory where the data for
			this snapshot is to be written

		i: int
		   Number of the file where this snapshot is to be written
		"""
		# Deduce the name of the filename where this snapshot writes
		self.filename       = os.path.join( write_dir, 'hdf5/data%08d.h5' %i)
		self.iteration      = i
		self.dt             = dt
		# Time and boundaries in the lab frame (constants quantities)
		self.zmin_lab       = zmin_lab
		self.zmax_lab       = zmax_lab
		self.t_lab          = t_lab

		# Positions where the fields are to be registered
		# (Change at every iteration)
		self.current_z_lab  = 0
		self.current_z_boost= 0

		# Buffered particle slice and corresponding array index in z

		self.buffered_slices= {}
		
		for species in species_dict:
		
			self.buffered_slices[species]=[]
			


	def update_current_output_positions( self, t_boost, inv_gamma, inv_beta ):
		"""
		Update the positions of output for this snapshot, so that
		if corresponds to the time t_boost in the boosted frame

		Parameters
		----------
		t_boost: float (seconds)
			Time of the current iteration, in the boosted frame

		inv_gamma, inv_beta: floats
			Inverse of the Lorentz factor of the boost, and inverse
			of the corresponding beta.
		"""
		t_lab = self.t_lab
	
		t_boost_diff = t_boost -self.dt
		# This implements the Lorentz transformation formulas,
		# for a snapshot having a fixed t_lab
		self.current_z_boost = ( t_lab*inv_gamma - t_boost )*c*inv_beta
		self.prev_z_boost   =  ( t_lab*inv_gamma - t_boost_diff )*c*inv_beta
		self.delta_t_boost = (self.prev_z_boost-self.current_z_boost)/(c*inv_beta)
		

		self.current_z_lab = ( t_lab - t_boost*inv_gamma )*c*inv_beta
		self.prev_z_lab  = (t_lab - t_boost_diff*inv_gamma)*c*inv_beta 

	def register_slice( self, slice_array, species):
		"""
		Store the slice of fields represented by slice_array
		and also store the z index at which this slice should be
		written in the final lab frame array

		Parameters
		----------
		slice_array: array of reals
			An array of packed fields that corresponds to one slice,
			as given by the SliceHandler object

		inv_dz_lab: float
			Inverse of the grid spacing in z, *in the lab frame*
		"""

		# Store the values and the index
		
		self.buffered_slices[species].append( slice_array)


	def compact_slices(self,species):
		"""
		Compact the successive slices that have been buffered
		over time into a single array, and return the indices
		at which this array should be written.

		Returns
		-------
		paticle_array: an array of reals of shape
		- (7, numPart, nslices) regardless of the dimension
		In the above nslices is the number of buffered slices

		iz_min, iz_max: integers
		The indices between which the slices should be written
		(iz_min is inclusive, iz_max is exclusive)

		Returns None if the slices are empty
		"""
		

		##problem here with stacking 
		particle_array = np.concatenate( self.buffered_slices[species], axis=1 )

		return particle_array

class ParticleCatcher:
	"""
	Class that extracts, Lorentz-transforms and writes particles
	"""
	def __init__( self, gamma_boost, beta_boost, top):
		"""
		Initialize the ParticleHandler object

		Parameters
		----------
		gamma_boost, beta_boost: float
			The Lorentz factor of the boost and the corresponding beta
		
		"""
		# Store the arguments
		self.gamma_boost = gamma_boost
		self.beta_boost  = beta_boost
		self.zboost      = 0.0
		self.zboost_prev = 0.0
		self.zlab        = 0.0
		self.zlab_prev   = 0.0
		self.top         =top
		# Create a dictionary that contains the correspondance
		# between the field names and array index
		self.particle_to_index = {'x':0, 'y':1, 'z':2, 'ux':3,
				'uy':4, 'uz':5, 'w':6, 'gamma':7}
			   
	def particle_getter(self,species):
		"""
		Returns the quantities of the particle at each time step

		Returns
		-------
		species : a warp Species object
		an array of quantities that are ready to be written in the buffer
		"""
		
		current_x   	=	self.get_quantity(species, "x" )
		current_y   	=	self.get_quantity(species, "y" )
		current_z   	=	self.get_quantity(species, "z" )
		current_ux   	=	self.get_quantity(species, "ux" )
		current_uy   	=	self.get_quantity(species, "uy" )
		current_uz   	=	self.get_quantity(species, "uz" )
		current_weights =	self.get_quantity(species, "w" )

		##the previous quantities

		previous_x	 = 	self.get_previous_quantity(species, "x" )
		previous_y	 = 	self.get_previous_quantity(species, "y" )
		previous_z	 = 	self.get_previous_quantity(species, "z" )
		previous_ux	= 	self.get_previous_quantity(species, "ux" )
		previous_uy	= 	self.get_previous_quantity(species, "uy" )
		previous_uz	= 	self.get_previous_quantity(species, "uz" )

		#an array for mapping purpose

		z_array = np.arange(len(current_z))

		# we track the particles that cross z_boost and 
		# particles that are confined between z_prev_boost and z_boost 
		
		ii=np.compress((((current_z>=self.zboost) & (previous_z<= self.zboost_prev)) |
			((current_z<=self.zboost) & (previous_z>= self.zboost_prev))),z_array)
		#print "zboost, zboostprev", self.zboost, self.zboost_prev
		
		## particle quantities that satisfy the aforementioned condition
		self.x_captured=np.take(current_x,ii)
		self.y_captured=np.take(current_y,ii)
		self.z_captured=np.take(current_z,ii)
		self.ux_captured=np.take(current_ux,ii)
		self.uy_captured=np.take(current_uy,ii)
		self.uz_captured=np.take(current_uz,ii)
		self.w_captured=np.take(current_weights,ii)
		self.gamma_captured = np.sqrt(1.+(self.ux_captured**2+\
			self.uy_captured**2+self.ux_captured**2)/c**2)

		self.x_prev_captured=np.take(previous_x,ii)
		self.y_prev_captured=np.take(previous_y,ii)
		self.z_prev_captured=np.take(previous_z,ii)
		self.ux_prev_captured=np.take(previous_ux,ii)
		self.uy_prev_captured=np.take(previous_uy,ii)
		self.uz_prev_captured=np.take(previous_uz,ii)
		self.gamma_prev_captured = np.sqrt(1.+(self.ux_prev_captured**2+\
			self.uy_prev_captured**2+self.ux_prev_captured**2)/c**2)
		num_part = len(ii)
		return num_part

	def transform_particles_to_lab_frame(self):
		uzfrm=-self.beta_boost*self.gamma_boost*c
		len_z=np.shape(self.z_captured)[0]
		
		self.top.setu_in_uzboosted_frame3d(len_z, 
				self.ux_captured,
				self.uy_captured, 
				self.uz_captured, 
				self.gamma_captured,
				uzfrm, self.gamma_boost)

		len_prev_z=np.shape(self.z_prev_captured)[0]
		self.top.setu_in_uzboosted_frame3d(len_prev_z, 
				self.ux_prev_captured,
				self.uy_prev_captured, 
				self.uz_prev_captured, 
				self.gamma_prev_captured,
				uzfrm, self.gamma_boost)

		self.z_captured	  = (self.z_captured-self.zboost) + self.zlab 
		self.z_prev_captured = (self.z_prev_captured-self.zboost_prev) + self.zlab_prev
		self.t		   = self.gamma_boost*self.top.time*np.ones(len_z) \
							-uzfrm*self.z_captured/c**2
		self.t_prev	   = self.gamma_boost*self.top.time*np.ones(len_prev_z) \
						-uzfrm*self.z_prev_captured/c**2

	def collapse_to_mid_point(self):
		"""
		collapse the particle quantities to the mid point between t_prev and t_current

		"""
		t_mid=np.mean(self.t+self.t_prev)
		self.x_captured	 = self.x_prev_captured*(self.t-t_mid)/(self.t-self.t_prev)\
							+self.x_captured*(t_mid-self.t)/(self.t-self.t_prev)
		self.y_captured	 = self.y_prev_captured*(self.t-t_mid)/(self.t-self.t_prev)\
							+self.y_captured*(t_mid-self.t)/(self.t-self.t_prev)
		self.z_captured	 = self.z_prev_captured*(self.t-t_mid)/(self.t-self.t_prev)\
							+self.z_captured*(t_mid-self.t)/(self.t-self.t_prev)
		self.ux_captured = self.ux_prev_captured*(self.t-t_mid)/(self.t-self.t_prev)\
							+self.ux_captured*(t_mid-self.t)/(self.t-self.t_prev)
		self.uy_captured = self.uy_prev_captured*(self.t-t_mid)/(self.t-self.t_prev)\
							+self.uy_captured*(t_mid-self.t)/(self.t-self.t_prev)
		self.uz_captured = self.uz_prev_captured*(self.t-t_mid)/(self.t-self.t_prev)\
							+self.uz_captured*(t_mid-self.t)/(self.t-self.t_prev)
		self.gamma_captured	 = self.gamma_prev_captured*(self.t-t_mid)/\
		(self.t-self.t_prev)+self.gamma_captured*(t_mid-self.t)/(self.t-self.t_prev)

	def gather_array(self, quantity):
		ar=np.zeros(np.shape(self.x_captured)[0])
		if quantity=="x" :
			ar=np.array(gatherarray(self.x_captured))
		elif quantity=="y":
			ar=np.array(gatherarray(self.y_captured))
		elif quantity=="z":
			ar=np.array(gatherarray(self.z_captured))
		elif quantity=="ux":
			ar=np.array(gatherarray(self.ux_captured))
		elif quantity=="uy":
			ar=np.array(gatherarray(self.uy_captured))
		elif quantity=="uz":
			ar=np.array(gatherarray(self.uz_captured))
		elif quantity=="w":
			ar=np.array(gatherarray(self.w_captured))
		elif quantity=="gamma":
			ar=np.array(gatherarray(self.gamma_captured))
		return ar

	def extract_slice(self, species):
		"""
		
		Extract a slice of the particles at z_boost, using interpolation in z

		See the docstring of extract_slice for the parameters.

		Returns
		-------
		An array that packs together the slices of the different particles.
			The shape of this arrays is:
			 (8, num_part,)  

		"""
		p2i     =self.particle_to_index

		num_part=self.particle_getter(species)

		# Transform the particles from boosted frame back to lab frame
		self.transform_particles_to_lab_frame()
		# Collapse the particle quantities using interpolation to
		# the the midpoint of t and t_prev
		self.collapse_to_mid_point()

		slice_array = np.empty( (8, num_part,) )
                                                                                            
		for quantity in self.particle_to_index.keys():
			 # Here typical values for `quantity` are e.g. 'z', 'ux', 'gamma'
			 slice_array[ p2i[quantity], ... ] = self.gather_array(quantity)
		return slice_array


	def get_quantity( self, species, quantity ) :
		"""
		Get a given quantity

		Parameters
		----------
		species : a Species object
			Contains the species object to get the particle data from

		quantity : string
			Describes which quantity is queried
			Either "x", "y", "z", "ux", "uy", "uz" or "w"
		"""
		# Extract the chosen quantities

		if quantity == "x" :
			quantity_array = species.getx(gather=False).flatten()
		elif quantity == "y" :
			quantity_array = species.gety(gather=False).flatten()
		elif quantity == "z" :
			quantity_array = species.getz(gather=False).flatten()
		elif quantity == "ux" :
			quantity_array = species.getux(gather=False).flatten()
		elif quantity == "uy" :
			quantity_array = species.getuy(gather=False).flatten()
		elif quantity == "uz" :
			quantity_array = species.getuz(gather=False).flatten()
		elif quantity == "w" :
			quantity_array = species.getweights(gather=False).flatten()

		return( quantity_array )

	def get_previous_quantity( self, species, quantity ) :
		"""
		Get a given quantity

		Parameters
		----------
		species : a Species object
			Contains the species object to get the particle data from

		quantity : string
			Describes which quantity is queried
			Either "x", "y", "z", "ux", "uy" or "uz" 
		"""
		# Extract the chosen quantities
		if quantity == "x" :
			quantity_array = species.getpid(id=self.top.xoldpid-1, gather=0,bcast=0).flatten()
		elif quantity == "y" :
			quantity_array = species.getpid(id=self.top.yoldpid-1, gather=0,bcast=0).flatten()
		elif quantity == "z" :
			quantity_array = species.getpid(id=self.top.zoldpid-1, gather=0,bcast=0).flatten()
		elif quantity == "ux" :
			quantity_array = species.getpid(id=self.top.uxoldpid-1, gather=0,bcast=0).flatten()
		elif quantity == "uy" :
			quantity_array = species.getpid(id=self.top.uyoldpid-1, gather=0,bcast=0).flatten()
		elif quantity == "uz" :
			quantity_array = species.getpid(id=self.top.uzoldpid-1, gather=0,bcast=0).flatten()
		
		return( quantity_array )

	def initialize_pevious_instant(self):
		if not self.top.xoldpid:self.top.xoldpid=self.top.nextpid()
		if not self.top.yoldpid:self.top.yoldpid=self.top.nextpid()
		if not self.top.zoldpid:self.top.zoldpid=self.top.nextpid()
		if not self.top.uxoldpid:self.top.uxoldpid=self.top.nextpid()
		if not self.top.uyoldpid:self.top.uyoldpid=self.top.nextpid()
		if not self.top.uzoldpid:self.top.uzoldpid=self.top.nextpid()
