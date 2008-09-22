"""Class for doing 3 D electromagnetic solver """
from warp import *
import operator

try:
  import psyco
except ImportError:
  pass

class EM3D(SubcycledPoissonSolver):
  
#  __w3dinputs__ = ['solvergeom','nx','ny','nzlocal','nz',
#                   'xmmin','xmmax','ymmin','ymmax','zmmin','zmmax',
#                   'zmminlocal','zmmaxlocal',
#                   'bound0','boundnz','boundxy','l2symtry','l4symtry',
#                   'solvergeom']
  __em3dinputs__ = []
#  __em3dinputs__ = ['l_onegrid','l_copyfields','l_moving_window',
#                    'tmin_moving_main_window','l_smoothdensity',
#                    'l_elaser_out_plane','ndelta_t',
#                   'ntamp_scatter','ntamp_gather']
#  __topinputs__ = ['pbound0','pboundnz','pboundxy',
#                   'my_index','nslaves','lfsautodecomp','zslave','lautodecomp',
#                   'debug']
  __flaginputs__ = {'l_apply_pml':true,'nbndx':10,'nbndy':10,'nbndz':10,
                    'nxguard':1,'nyguard':1,'nzguard':1,
                    'l_particles_weight':false,'l_usecoeffs':false,
                    'l_pushf':false,'l_pushpot':false,'l_verbose':false,
                    'laser_amplitude':1.,'laser_profile':None,
                    'laser_gauss_width':None,'laser_angle':0.,
                    'laser_wavelength':None,'laser_wavenumber':None,
                    'laser_frequency':None,'laser_source':2,
                    'laser_focus':None,'laser_focus_velocity':0.,
                    'nfield_subcycle':1,'density_1d':0,
                    'autoset_timestep':true,'dtcoef':1.,#0.99,
                    'deposit_energy_density':false,'refinement':None}

  def __init__(self,**kw):
    self.grid_overlap = 1
    FieldSolver.__init__(self,kwdict=kw)
#    top.allspecl = true
    top.lcallfetchb = true
    top.lgridqnt = true
    self.zgridprv=top.zgrid
    # --- Save input parameters
#    self.processdefaultsfrompackage(EM3D.__w3dinputs__,w3d,kw)
#    self.processdefaultsfrompackage(EM3D.__topinputs__,top,kw)
    self.processdefaultsfrompackage(EM3D.__em3dinputs__,em3d,kw)
    self.processdefaultsfromdict(EM3D.__flaginputs__,kw)

    # --- bounds is special since it will sometimes be set from the
    # --- variables bound0, boundnz, boundxy, l2symtry, and l4symtry
    if 'bounds' not in self.__dict__:
      if 'bounds' in kw:
        self.bounds = kw['bounds']
      else:
        self.bounds = zeros(6,'l')
        self.bounds[0] = self.boundxy
        self.bounds[1] = self.boundxy
        self.bounds[2] = self.boundxy
        self.bounds[3] = self.boundxy
        self.bounds[4] = self.bound0
        self.bounds[5] = self.boundnz
        if self.l2symtry:
          self.bounds[2] = neumann
          if self.boundxy == periodic: self.bounds[3] = neumann
          if self.forcesymmetries: self.ymmin = 0.
        elif self.l4symtry:
          self.bounds[0] = neumann
          self.bounds[2] = neumann
          if self.boundxy == periodic: self.bounds[1] = neumann
          if self.boundxy == periodic: self.bounds[3] = neumann
          if self.forcesymmetries: self.xmmin = 0.
          if self.forcesymmetries: self.ymmin = 0.

    # --- If there are any remaning keyword arguments, raise an error.
    assert len(kw.keys()) == 0,"Bad keyword arguemnts %s"%kw.keys()

    # --- Calculate mesh sizes
    self.dx = (self.xmmax - self.xmmin)/self.nx
    self.xmesh = self.xmmin + arange(0,self.nx+1)*self.dx
    self.dy = (self.ymmax - self.ymmin)/self.ny
    self.ymesh = self.ymmin + arange(0,self.ny+1)*self.dy
    self.dz = (self.zmmax - self.zmmin)/self.nz
    self.zmesh = self.zmmin + arange(0,self.nz+1)*self.dz
#    self.zmeshlocal = self.zmminlocal + arange(0,self.nzlocal+1)*self.dz

    # --- set time step as a fraction of Courant condition
    # --- also set self.nfield_subcycle if top.dt over Courant condition times dtcoef
    if self.autoset_timestep:
      dt=self.dtcoef/(clight*sqrt(1./self.dx**2+1./self.dy**2+1./self.dz**2))
      if top.dt==0.:
        top.dt=dt
      else:
        if top.dt<>dt:
          self.nfield_subcycle=int(top.dt/dt)+1
    self.dtinit = top.dt
    
    # --- Create field and source arrays and other arrays.
    self.allocatefieldarrays()
    self.initializeconductors()

    # --- Handle laser inputs
    self.setuplaser()

    installbeforeloadrho(self.solve2ndhalf)
    print 'l_pushf',self.l_pushf
    if self.refinement is not None:
      ref=self.refinement
      self.field_coarse = self.__class__(
                               nx=self.nx/ref[0],dx=self.dx*ref[0],
                               ny=self.ny/ref[1],dy=self.dy*ref[1],
                               nz=self.nz/ref[2],dz=self.dz*ref[2],
                               nzlocal=self.nzlocal/ref[2],
                               xmmin=self.xmmin,xmmax=self.xmmax,
                               ymmin=self.ymmin,ymmax=self.ymmax,
                               zmmin=self.zmmin,zmmax=self.zmmax,
                               zmminlocal=self.zmminlocal,zmmaxlocal=self.zmmaxlocal,
                               bounds=self.bounds,
                               **self.kw)
      self.block_coarse = self.field_coarse.block

  def processdefaultsfrompackage(self,defaults,package,kw):
    for name in defaults:
      if name not in self.__dict__:
        #self.__dict__[name] = kw.pop(name,getattr(w3d,name)) # Python2.3
        self.__dict__[name] = kw.get(name,getattr(package,name))
      if kw.has_key(name): del kw[name]

  def processdefaultsfromdict(self,dict,kw):
    for name,defvalue in dict.iteritems():
      if name not in self.__dict__:
        #self.__dict__[name] = kw.pop(name,getattr(top,name)) # Python2.3
        self.__dict__[name] = kw.get(name,defvalue)
      if kw.has_key(name): del kw[name]

  def getpdims(self):
    # --- Returns the dimensions of the arrays used by the particles

    # --- If there are any relativistic groups, then turn on the code
    # --- which uses the selfe array.
    if max(abs(top.fselfb)) > 0.:
      # --- This is probably redundant, but it shouldn't hurt.
      # --- This forces all species to use the precalculated E field
      # --- if any have the B correction.
      top.efetch = 3
      # --- Number of fields (E and B)
      nfields = 2
    else:
      # --- Number of fields (E only)
      nfields = 1

    if sometrue(top.efetch == 3):
      return ((1+self.nxp,1+self.nyp,1+self.nzp),
              (3,1+self.nxp,1+self.nyp,1+self.nzp,nfields),
              (1+self.nxp+2*self.nxguard,1+self.nyp+2*self.nyguard,1+self.nzp+2*self.nzguard))
    else:
      return ((1+self.nxp,1+self.nyp,1+self.nzp),
              (1+self.nxp+2*self.nxguard,1+self.nyp+2*self.nyguard,1+self.nzp+2*self.nzguard))

  def getdims(self):
    # --- Returns the dimensions of the arrays used by the field solver
    return ((1+self.nx,1+self.ny,1+self.nzlocal),
            (1+self.nx+2*self.nxguard,1+self.ny+2*self.nyguard,1+self.nzlocal+2*self.nzguard))

  def allocatefieldarrays(self):
    self.block = pyinit_3dem_block(self.nx, 
                                   self.ny, 
                                   self.nzlocal,
                                   self.nbndx, 
                                   self.nbndy, 
                                   self.nbndz, 
                                   self.nxguard, 
                                   self.nyguard, 
                                   self.nzguard, 
                                   top.dt, 
                                   self.dx, 
                                   self.dy, 
                                   self.dz, 
                                   clight, 
                                   mu0, 
                                   self.xmmin, 
                                   self.ymmin, 
                                   self.zmminlocal, 
                                   self.bounds[0], 
                                   self.bounds[2], 
                                   self.bounds[4], 
                                   self.bounds[1], 
                                   self.bounds[3], 
                                   self.bounds[5],
                                   self.deposit_energy_density,
                                   self.refinement,
                                   self.l_pushf)
    self.fields = self.block.core.yf
    
  def addpatch(self,ixpatch,iypatch,nxpatch,nypatch,rap):
    # --- Initializes refinement patch
    em3d.l_onegrid = false
    self.l_onegrid = em3d.l_onegrid
    self.fpatches.append([EM3D_FIELDtype(),EM3D_FIELDtype()])
    self.fpatchcoarse = self.fpatches[-1][0]
    self.fpatchfine   = self.fpatches[-1][1]
    self.fpatchfine.rap = rap
    xlb = xrb = ylb = yrb = absorb
    if ixpatch==0:xlb = self.bounds[0]
    if iypatch==0:ylb = self.bounds[1]
    if ixpatch+nxpatch==self.nx:xrb = self.bounds[2]
    if iypatch+nypatch==self.ny:yrb = self.bounds[3]
    init_fields(self.fpatchcoarse,nxpatch,nypatch,self.nbndx,self.nbndx,top.dt/self.nfield_subcycle,
                self.dx,self.dy,clight,mu0,
                self.xmmin+ixpatch*self.dx,self.ymmin+iypatch*self.dy,1,
                xlb,ylb,xrb,yrb)
    init_fields(self.fpatchfine,nxpatch*rap,nypatch*rap,self.nbndx,self.nbndx,top.dt/self.nfield_subcycle,
                self.dx/rap,self.dy/rap,clight,mu0,
                self.xmmin+ixpatch*self.dx,self.ymmin+iypatch*self.dy,rap,
                xlb,ylb,xrb,yrb)
    self.fpatchcoarse.js = self.laser_source
    self.fpatchfine.js = self.laser_source*rap
    self.fpatchfine.xminpatch_scatter = self.fpatchfine.xmin+em3d.ntamp_scatter*rap*self.fpatchfine.dx
    self.fpatchfine.xmaxpatch_scatter = self.fpatchfine.xmax-em3d.ntamp_scatter*rap*self.fpatchfine.dx
    self.fpatchfine.yminpatch_scatter = self.fpatchfine.ymin+em3d.ntamp_scatter*rap*self.fpatchfine.dy
    self.fpatchfine.ymaxpatch_scatter = self.fpatchfine.ymax-em3d.ntamp_scatter*rap*self.fpatchfine.dy
    self.fpatchfine.xminpatch_gather = self.fpatchfine.xmin+em3d.ntamp_gather*rap*self.fpatchfine.dx
    self.fpatchfine.xmaxpatch_gather = self.fpatchfine.xmax-em3d.ntamp_gather*rap*self.fpatchfine.dx
    self.fpatchfine.yminpatch_gather = self.fpatchfine.ymin+em3d.ntamp_gather*rap*self.fpatchfine.dy
    self.fpatchfine.ymaxpatch_gather = self.fpatchfine.ymax-em3d.ntamp_gather*rap*self.fpatchfine.dy
    self.fpatchfine.nxfsum = self.fpatchfine.nx
    self.fpatchfine.nyfsum = self.fpatchfine.ny
    self.fpatchfine.gchange()
    self.setuplaser_profile(self.fpatchcoarse)
    self.setuplaser_profile(self.fpatchfine)

  def setuplaser(self):
    if self.laser_profile is not None:
      if self.laser_frequency is None:
        if self.laser_wavenumber is not None:
          self.laser_wavelength = clight*self.laser_wavenumber
        elif self.laser_wavelength is not None:
          self.laser_wavelength = 2.*pi*clight/self.laser_wavelength
      assert self.laser_frequency is not None,\
             "One of the frequency, wavenumber, or wavelength must be given"

    # --- Check if laser_amplitude is a function, table, or constant
    self.laser_amplitude_func = None
    self.laser_amplitude_table = None
    if operator.isSequenceType(self.laser_amplitude):
      assert len(self.laser_amplitude.shape) == 2 and \
             self.laser_amplitude.shape[1] == 2,\
             "The laser_amplitude table is not formatted properly"
      self.laser_amplitude_table = self.laser_amplitude
      self.laser_amplitude_table_i = -1
    elif callable(self.laser_amplitude):
      self.laser_amplitude_func = self.laser_amplitude
      
    self.setuplaser_profile(self.fields)

  def setuplaser_profile(self,f):
    # --- Check if laser_profile has a type, is a function, or a table
    self.laser_profile_func = None
    # --- disable laser emission on processors id>0 
    if me>0:return
    if self.laser_profile == 'gaussian':
      assert self.laser_gauss_width is not None,\
             "For a gaussian laser, the width must be specified using laser_gauss_width"
#      xx = arange(self.nx+4)*f.dx+f.xmin*f.dx - 0.5*f.nx*f.dx
#      self.laser_profile = exp(-(xx/self.laser_gauss_width)**2/2.)
      yy = arange(f.ny+3)*f.dy+f.ymin*f.dy - 0.5*f.ny*f.dy
      self.laser_profile = exp(-(yy/self.laser_gauss_width)**2/2.)
    elif operator.isSequenceType(self.laser_profile):
      assert len(self.laser_profile) == f.ny+3,"The specified profile must be of length ny+3"
    elif callable(self.laser_profile):
      self.laser_profile_func = self.laser_profile
    
  def fetchfieldfrompositions(self,x,y,z,ex,ey,ez,bx,by,bz,js=0,pgroup=None):
    # --- This is called by fetchfield from fieldsolver.py
    n = len(x)
    if n == 0: return
    nox = top.depos_order[0,w3d.jsfsapi]
    noy = top.depos_order[1,w3d.jsfsapi]
    noz = top.depos_order[2,w3d.jsfsapi]
#    if (not (self.getconductorobject(top.fselfb[iselfb]).lcorrectede or
#    else:
    f = self.block.core.yf
    # --- fetch e
    if top.efetch[w3d.jsfsapi] in [1,3,5]:
      getf3d_linear(n,x,y,z,ex,ey,ez,
                    f.xmin,f.ymin,f.zmin,
                    f.dx,f.dy,f.dz,
                    f.nx,f.ny,f.nz,
                    f.nxguard,f.nyguard,f.nzguard,
                    f.Exp,f.Eyp,f.Ezp)
    elif top.efetch[w3d.jsfsapi]==4:
      gete3d_n_energy_conserving(n,x,y,z,ex,ey,ez,
                    f.xmin,f.ymin,f.zmin,
                    f.dx,f.dy,f.dz,
                    f.nx,f.ny,f.nz,
                    f.nxguard,f.nyguard,f.nzguard,
                    nox,noy,noz,
                    f.Exp,f.Eyp,f.Ezp,
                    w3d.l4symtry)
    # --- fetch b
    if top.efetch[w3d.jsfsapi] in [1,3,5]:
      getf3d_linear(n,x,y,z,bx,by,bz,
                    f.xmin,f.ymin,f.zmin,
                    f.dx,f.dy,f.dz,
                    f.nx,f.ny,f.nz,
                    f.nxguard,f.nyguard,f.nzguard,
                    f.Bxp,f.Byp,f.Bzp)
    elif top.efetch[w3d.jsfsapi]==4:
      getb3d_n_energy_conserving(n,x,y,z,bx,by,bz,
                    f.xmin,f.ymin,f.zmin,
                    f.dx,f.dy,f.dz,
                    f.nx,f.ny,f.nz,
                    f.nxguard,f.nyguard,f.nzguard,
                    nox,noy,noz,
                    f.Bxp,f.Byp,f.Bzp,
                    w3d.l4symtry)

  def fetchphifrompositions(self,x,z,phi):
    pass

  def setsourcep(self,js,pgroup,zgrid):
    if self.l_verbose:print 'setsourcep, species ',js
    n  = pgroup.nps[js]
    if n == 0: return
    i  = pgroup.ins[js] - 1
    x  = pgroup.xp[i:i+n]
    y  = pgroup.yp[i:i+n]
    z  = pgroup.zp[i:i+n]
    ux = pgroup.uxp[i:i+n]
    uy = pgroup.uyp[i:i+n]
    uz = pgroup.uzp[i:i+n]
    gaminv = pgroup.gaminv[i:i+n]
    q  = pgroup.sq[js]
    w  = pgroup.sw[js]*pgroup.dtscale[js]
    if top.wpid==0:
      wfact = zeros((0,), 'd')
    else:
      wfact = top.pgroup.pid[i:i+n,top.wpid-1]
    w3d.jsfsapi=js
    self.setsourcepatposition(x,y,z,ux,uy,uz,gaminv,wfact,zgrid,q,w)

  def setsourcepatposition(self,x,y,z,ux,uy,uz,gaminv,wfact,zgrid,q,w):
    n = len(x)
    if n == 0: return
    # --- call routine performing current deposition
    f = self.block.core.yf
    js = w3d.jsfsapi
    if top.wpid==0:
      wfact = ones((1,),'d')
      l_particles_weight = false
    else:
      l_particles_weight = true
    if not self.deposit_energy_density:
          depose_jxjyjz_esirkepov_n(self.fields.J,n,
                                            x,y,z,ux,uy,uz,
                                            gaminv,wfact,q*w,
                                            f.xmin,f.ymin,f.zmin,
                                            top.dt*top.pgroup.ndts[js],
                                            f.dx,f.dy,f.dz,
                                            f.nx,f.ny,f.nz,
                                            f.nxguard,f.nyguard,f.nzguard,
                                            top.depos_order[0,js],
                                            top.depos_order[1,js],
                                            top.depos_order[2,js],
                                            l_particles_weight,w3d.l4symtry)
    else:
          depose_jxjyjz_pxpypz_esirkepov_linear_serial(self.fields.J,self.fields.Mp,n,
                                            x,y,z,ux,uy,uz,
                                            gaminv,wfact,q*w,m,
                                            f.xmin,f.ymin,f.zmin,
                                            top.dt*top.pgroup.ndts[js],
                                            f.dx,f.dy,f.dz,
                                            f.nx,f.ny,f.nz,
                                            f.nxguard,f.nyguard,f.nzguard,
                                            l_particles_weight,
                                            top.lrelativ)
    if self.l_pushf:
      depose_rho_n(self.fields.Rho,n,
                               x,y,z,
                               wfact,q*w,
                               f.xmin,f.ymin,f.zmin,
                               f.dx,f.dy,f.dz,
                               f.nx,f.ny,f.nz,
                               f.nxguard,f.nyguard,f.nzguard,
                               top.depos_order[0,js],
                               top.depos_order[1,js],
                               top.depos_order[2,js],
                               l_particles_weight,w3d.l4symtry)

  def allocatedataarrays(self):
    if self.l_verbose:print 'allocatedataarrays'
    # --- reallocate Jarray if needed
    if self.fields.ntimes<>top.nsndts:
      self.fields.ntimes=top.nsndts
      self.fields.gchange()

  def zerosourcep(self):
    if self.l_verbose:print 'zerosourcep',self

    # --- copy rho to rhoold if needed
    if self.l_pushf and self.nfield_subcycle>1:
      self.fields.Rhoold = self.fields.Rho.copy()

    # --- zero proper portion of Jarray
    for indts in range(top.nsndts):
      if top.ldts[indts]:
        self.fields.Jarray[:,:,:,:,indts] = 0.
        if self.l_pushf:
          self.fields.Rhoarray[:,:,:,indts] = 0.
        if self.deposit_energy_density:
          self.fields.Mp[...] = 0.
  
  def setsourcepforparticles(self,isourcepndtscopies,indts,iselfb):
    if self.l_verbose:print 'setsourcepforparticles'
    # --- point J array to proper Jarray slice
    self.fields.J = self.fields.Jarray[:,:,:,:,indts]
    if self.l_pushf: self.fields.Rho = self.fields.Rhoarray[:,:,:,indts]

  def finalizesourcep(self):
    if self.sourcepfinalized: return
    self.sourcepfinalized = 1
    if self.l_verbose:print 'finalizesourcep blabla'
    # --- add slices
    if top.nsndts>1:
      for i in range(top.nsndts-2,-1,-1):
        if top.ldts[indts]:
          add_current_slice_3d(self.fields,indts+1)
          if self.l_pushf:add_rho_slice_3d(self.fields,indts+1)
    self.aftersetsourcep()
          
    # --- apply boundary condition on current
    if not self.density_1d:
      self.apply_current_bc()
    if self.l_pushf:self.apply_rho_bc()

    # --- point J to first slice of Jarray
    self.fields.J = self.fields.Jarray[:,:,:,:,0]
    if self.l_pushf:self.fields.Rho = self.fields.Rhoarray[:,:,:,0]
    
    # --- get 1-D density
    if self.density_1d:
      for i in range(3):
       J = sum(sum(self.fields.J[:,:,:,i],0),0)
       for ii in range(shape(self.fields.J[:,:,:,i])[1]):
        for jj in range(shape(self.fields.J[:,:,:,i])[0]):
         self.fields.J[ii,jj,:,i] = J

    # --- smooth current density 
#    if self.l_smoothdensity:self.smoothdensity()
    if self.l_verbose:print 'finalizesourcep done'
#    print 'sum',sum(self.fields.Mp[...,-1]*w3d.dz), \
#          sum(emass*(1./where(top.pgroup.gaminv==0.,1.,top.pgroup.gaminv)-1.)*top.pgroup.sw[0]*clight**2), \
#          sum(self.fields.J[...,-1]**2)*(top.dt/eps0)**2*eps0*w3d.dz/2
#    print 'max',maxnd(self.fields.Mp[...,-1]*w3d.dz), \
#          maxnd(emass*(1./where(top.pgroup.gaminv==0.,1.,top.pgroup.gaminv)-1.)*top.pgroup.sw[0]*clight**2), \
#          maxnd(self.fields.J[...,-1]**2)*(top.dt/eps0)**2*eps0*w3d.dz/2


  def apply_rho_bc(self):
    # --- point Rho to first slice of Rhoarray
    self.fields.Rho = self.fields.Rhoarray[:,:,:,0]
    em3d_applybc_rho(self.block.core.yf, 
                     self.block.xlbnd,
                     self.block.xrbnd,
                     self.block.ylbnd,
                     self.block.yrbnd,
                     self.block.zlbnd,
                     self.block.zrbnd)
    em3d_exchange_rho(self.block)

  def apply_current_bc(self):
    # --- point J to first slice of Jarray
    self.fields.J = self.fields.Jarray[:,:,:,:,0]
    em3d_applybc_j(self.block.core.yf, 
                   self.block.xlbnd,
                   self.block.xrbnd,
                   self.block.ylbnd,
                   self.block.yrbnd,
                   self.block.zlbnd,
                   self.block.zrbnd)
    em3d_exchange_j(self.block)

  def smoothdensity(self):
    smooth2d_lindman(self.fields.J[:,:,0],self.fields.nx,self.fields.ny)
    smooth2d_lindman(self.fields.J[:,:,1],self.fields.nx,self.fields.ny)
    smooth2d_lindman(self.fields.J[:,:,2],self.fields.nx,self.fields.ny)

  def fetche(self,*args,**kw):
    SubcycledPoissonSolver.fetchfield(self,*args,**kw)

  def loadrho(self,lzero=None,pgroups=None,**kw):
    if self.l_verbose:print 'loadrho',self
    SubcycledPoissonSolver.loadsource(self,lzero,pgroups,**kw)
    
  def fetchphi(self):
    pass

  def fetcha(self):
    pass

  def initializeconductors(self):
    # --- Create the attributes for holding information about conductors
    # --- and conductor objects.
    # --- Note that a conductor object will be created for each value of
    # --- fselfb. This is needed since fselfb effects how the coarsening
    # --- is done, and different conductor data sets are needed for
    # --- different coarsenings.

    # --- This stores the ConductorType objects. Note that the objects are
    # --- not actually created until getconductorobject is called.
    self.conductorobjects = {}

    # --- This stores the conductors that have been installed in each
    # --- of the conductor objects.
    self.installedconductorlists = {}

    # --- This is a list of conductors that have been added.
    # --- New conductors are not actually installed until the data is needed,
    # --- when getconductorobject is called.
    # --- Each element of this list contains all of the input to the
    # --- installconductor method.
    self.conductordatalist = []
#    self.mgmaxlevels=0
#    self.mglevels=0

  def installconductor(self,conductor,
                            xmin=None,xmax=None,
                            ymin=None,ymax=None,
                            zmin=None,zmax=None,
                            dfill=top.largepos):
    # --- This only adds the conductor to the list. The data is only actually
    # --- installed when it is needed, during a call to getconductorobject.
    self.conductordatalist.append((conductor,xmin,xmax,ymin,ymax,zmin,zmax,dfill))

  def _installconductor(self,conductorobject,installedlist,conductordata,fselfb):
    # --- This does that actual installation of the conductor into the
    # --- conductor object

    # --- Extract the data from conductordata (the arguments to installconductor)
    conductor,xmin,xmax,ymin,ymax,zmin,zmax,dfill = conductordata

    if conductor in installedlist: return
    installedlist.append(conductor)

    if fselfb == 'p':
      zscale = 1.
      nx,ny,nzlocal,nz = self.nxp,self.nyp,self.nzp,self.nz
      xmmin,xmmax = self.xmminp,self.xmmaxp
      ymmin,ymmax = self.ymminp,self.ymmaxp
      zmmin,zmmax = self.zmminp,self.zmmaxp
      mgmaxlevels = 1
    else:
      # --- Get relativistic longitudinal scaling factor
      # --- This is quite ready yet.
      beta = fselfb/clight
      zscale = 1./sqrt((1.-beta)*(1.+beta))
      nx,ny,nzlocal,nz = self.nx,self.ny,self.nzlocal,self.nz
      xmmin,xmmax = self.xmmin,self.xmmax
      ymmin,ymmax = self.ymmin,self.ymmax
      zmmin,zmmax = self.zmmin,self.zmmax
      mgmaxlevels = None

    mgmaxlevels=1
    installconductors(conductor,xmin,xmax,ymin,ymax,zmin,zmax,dfill,
                      self.getzgrid(),
                      nx,ny,nzlocal,nz,
                      xmmin,xmmax,ymmin,ymmax,zmmin,zmmax,
                      zscale,self.l2symtry,self.l4symtry,
                      installrz=0,
                      solvergeom=self.solvergeom,conductors=conductorobject,
                      mgmaxlevels=mgmaxlevels,
                      my_index=self.my_index,nslaves=self.nslaves,
                      izfsslave=self.izfsslave,nzfsslave=self.nzfsslave)

    self.fields.nconds = self.getconductorobject().interior.n
    self.fields.nxcond = self.fields.nx
    self.fields.nycond = self.fields.ny
    self.fields.nzcond = self.fields.nz
    self.fields.gchange()
    self.fields.incond=False
    if self.fields.nconds>0:
      set_incond(self.fields,self.fields.nconds,int(self.getconductorobject().interior.indx))
    if self.block.xlbnd==openbc:self.fields.incond[:3,:,:]=False
    if self.block.xrbnd==openbc:self.fields.incond[-3:,:,:]=False
    if self.block.ylbnd==openbc:self.fields.incond[:,:3,:]=False
    if self.block.yrbnd==openbc:self.fields.incond[:,-3:,:]=False
    if self.block.zlbnd==openbc:self.fields.incond[:,:,:3]=False
    if self.block.zrbnd==openbc:self.fields.incond[:,:,-3:]=False
    
  def hasconductors(self):
    return len(self.conductordatalist) > 0

  def clearconductors(self):
    "Clear out the conductor data"
    for fselfb in top.fselfb:
      if fselfb in self.conductorobjects:
        conductorobject = self.conductorobjects[fselfb]
        conductorobject.interior.n = 0
        conductorobject.evensubgrid.n = 0
        conductorobject.oddsubgrid.n = 0
        self.installedconductorlists[fselfb] = []

  def getconductorobject(self,fselfb=0.):
    "Checks for and installs any conductors not yet installed before returning the object"
    # --- This is the routine that does the creation of the ConductorType
    # --- objects if needed and ensures that all conductors are installed
    # --- into it.

    # --- This method is needed during a restore from a pickle, since this
    # --- object may be restored before the conductors. This delays the
    # --- installation of the conductors until they are really needed.

    # --- There is a special case, fselfb='p', which refers to the conductor
    # --- object that has the data generated relative to the particle domain,
    # --- which can be different from the field domain, especially in parallel.
    if fselfb == 'p':
      # --- In serial, just use a reference to the conductor object for the
      # --- first iselfb group.
      if not lparallel and 'p' not in self.conductorobjects:
        self.conductorobjects['p'] = self.conductorobjects[top.fselfb[0]]
        self.installedconductorlists['p'] = self.installedconductorlists[top.fselfb[0]]
      # --- In parallel, a whole new instance is created (using the
      # --- setdefaults below).
      # --- Check to make sure that the grid the conductor uses is consistent
      # --- with the particle grid. This is needed so that the conductor
      # --- data is updated when particle load balancing is done. If the
      # --- data is not consistent, delete the conductor object so that
      # --- everything is reinstalled.
      try:
        conductorobject = self.conductorobjects['p']
        if (conductorobject.leveliz[0] != self.izpslave[self.my_index] or
            conductorobject.levelnz[0] != self.nzpslave[self.my_index]):
          del self.conductorobjects['p']
          del self.installedconductorlists['p']
      except KeyError:
        # --- 'p' object has not yet been created anyway, so do nothing.
        pass

    conductorobject = self.conductorobjects.setdefault(fselfb,ConductorType())
    installedconductorlist = self.installedconductorlists.setdefault(fselfb,[])

    # --- Now, make sure that the conductors are installed into the object.
    # --- This may be somewhat inefficient, since it loops over all of the
    # --- conductors everytime. This makes the code more robust, though, since
    # --- it ensures that all conductors will be properly installed into
    # --- the conductor object.
    for conductordata in self.conductordatalist:
      self._installconductor(conductorobject,installedconductorlist,
                             conductordata,fselfb)
 
    # --- Return the desired conductor object
    return conductorobject

  def setconductorvoltage(self,voltage,condid=0,discrete=false,
                          setvinject=false):
    'calls setconductorvoltage'
    # --- Loop over all of the selfb groups to that all conductor objects
    # --- are handled.
    for iselfb in range(top.nsselfb):
      conductorobject = self.getconductorobject(top.fselfb[iselfb])
      setconductorvoltage(voltage,condid,discrete,setvinject,
                          conductors=conductorobject)

  def optimizeconvergence(self,resetpasses=1):
    pass

  def move_window_fields(self):
    if (w3d.solvergeom not in [w3d.XZgeom]) or \
       (abs(top.zgrid-self.zgridprv)<0.5*self.dz):return 
#    if not self.l_moving_window or ((top.it%self.ndelta_t)!=0): return
#    if top.time < self.tmin_moving_main_window: return
    move_window_field(self.fields)
    self.zgridprv=top.zgrid
    if not self.l_onegrid:
      self.fpatchfine.xminscatter+=w3d.dz
      self.fpatchfine.xmaxscatter+=w3d.dz
      self.fpatchfine.xmingather+=w3d.dz
      self.fpatchfine.xmaxgather+=w3d.dz

  def add_laser(self,field):
    if self.laser_profile is None: return

    if self.laser_amplitude_func is not None:
      self.laser_amplitude = self.laser_amplitude_func(top.time)
    elif self.laser_amplitude_table is not None:
      if top.time < self.laser_amplitude_table[0,1]:
        self.laser_amplitude = self.laser_amplitude_table[0,0]
      elif top.time >= self.laser_amplitude_table[-1,1]:
        self.laser_amplitude = self.laser_amplitude_table[-1,0]
      else:
        i = self.laser_amplitude_table_i
        while top.time > self.laser_amplitude_table[i+1,1]:
          i = i + 1
        self.laser_amplitude_table_i = i
        ww = ((top.time - self.laser_amplitude_table[i,1])/
           (self.laser_amplitude_table[i+1,1]-self.laser_amplitude_table[i,1]))
        self.laser_amplitude = ((1.-ww)*self.laser_amplitude_table[i,0] +
                                    ww *self.laser_amplitude_table[i+1,0])

    if self.laser_profile_func is not None:
      self.laser_profile = self.laser_profile_func(top.time)
      assert len(self.laser_profile) == field.ny+3,"The specified profile must be of length ny+4"

#    if (self.l_elaser_out_plane):
#      xx = (arange(self.nx+4) - 0.5)*self.fields.dx+self.fields.xmin
#    else:
    xx = (arange(field.ny+3) - 0.5)*field.dy+field.ymin

    if self.laser_frequency is not None:
      if self.laser_focus is not None:
        z0 = self.laser_focus+self.laser_focus_velocity*top.time
        if self.laser_focus>0.:
          phase = (-(sqrt(xx**2+z0**2)-z0)/clight-top.time)*self.laser_frequency
        else:
          phase = ((sqrt(xx**2+z0**2)-z0)/clight-top.time)*self.laser_frequency
      else:
        phase = (xx*sin(self.laser_angle)/clight-top.time)*self.laser_frequency
    else:
      phase = 0.

    self.block.core.yf.Ey[:,:,1] = self.laser_amplitude*self.laser_profile[0]*cos(phase[0])
    return
    if (self.l_elaser_out_plane):
      field.Ez_in = self.laser_amplitude*field.laser_profile[:-1]*cos(phase)
    else:
      field.Bz_in = self.laser_amplitude*field.laser_profile[:-1]*cos(phase)

  def solve2ndhalf(self):
    if self.l_verbose:print 'solve 2nd half',self
    if top.dt<>self.dtinit:raise('Time step has been changed since initialization of EM3D.')
    # --- Set nxl and nyl if using large stencil
#    if(not self.l_onegrid):
#      project_j(self.fields,self.fpatchcoarse,self.fpatchfine)
#    for field in fields:
#      if field.l_uselargestencil and field.nxl<>field.nx:
#        field.nxl=field.nx
#        field.nyl=field.ny
#        field.gchange()
#      self.add_laser(field)
#      grimax(field)
    if top.efetch[0]<>4:node2yee3d(self.block.core.yf)
    dt = top.dt/self.nfield_subcycle
    push_em3d_bf(self.block,dt,2,self.l_pushf,self.l_pushpot)
    if self.l_verbose:print 'solve 2nd half done'
#    yee2node3d(self.block.core.yf)
 #   self.move_window_fields()
 #   for field in fields:
 #     griuni(field)
 #   if not self.l_onegrid:set_substitute_fields(self.fields,self.fpatchcoarse,self.fpatchfine)

  def dosolve(self,iwhich=0,*args):
    if self.l_verbose:print 'solve 1st half'
    if top.dt<>self.dtinit:raise('Time step has been changed since initialization of EM3D.')
    # --- Set nxl and nyl if using large stencil
#    if(not self.l_onegrid):
#      project_j(self.fields,self.fpatchcoarse,self.fpatchfine)
#    for field in fields:
#      if field.l_uselargestencil and field.nxl<>field.nx:
#        field.nxl=field.nx
#        field.nyl=field.ny
#        field.gchange()
#      self.add_laser(field)
#      grimax(field)
#    node2yee3d(self.block.core.yf)
    dt = top.dt/self.nfield_subcycle
    if self.l_pushf and self.nfield_subcycle>1:
      w = 1./self.nfield_subcycle
      self.fields.Rho = (1.-w)*self.fields.Rhoold + w*self.fields.Rhoarray[...,0] 
    push_em3d_eef(self.block,dt,0,self.l_pushf,self.l_pushpot)
    self.add_laser(self.fields)
    for i in range(self.nfield_subcycle-1):
      push_em3d_bf(self.block,dt,0,self.l_pushf,self.l_pushpot)
      if self.l_pushf and self.nfield_subcycle>1:
        w = float(i+2)/self.nfield_subcycle
        self.fields.Rho = (1.-w)*self.fields.Rhoold + w*self.fields.Rhoarray[...,0] 
      push_em3d_eef(self.block,dt,0,self.l_pushf,self.l_pushpot)
    push_em3d_bf(self.block,dt,1,self.l_pushf,self.l_pushpot)
#    push_em3d_eef(self.block,dt,0,self.l_pushf)
#    push_em3d_bf(self.block,dt,1,self.l_pushf)
#    for i in range(self.nfield_subcycle-1):
#      push_em3d_bf(self.block,dt,2,self.l_pushf)
#      push_em3d_eef(self.block,dt,0,self.l_pushf)
#      push_em3d_bf(self.block,dt,1,self.l_pushf)
    if not all(top.efetch==top.efetch[0]):raise('Error:top.efetch must have same value for every species when using EM solver.')
    if top.efetch[0]<>4:yee2node3d(self.block.core.yf)
 #   self.move_window_fields()
 #   for field in fields:
 #     griuni(field)
 #   if not self.l_onegrid:set_substitute_fields(self.fields,self.fpatchcoarse,self.fpatchfine)
    if self.l_verbose:print 'solve 1st half done'
    if self.refinement is not None:self.field_coarse.solve()

  ##########################################################################
  # Define the basic plot commands
  def genericpfem3d(self,data,title,l_transpose=false,direction=2,slice=None,**kw):
    f=self.block.core.yf
    if direction==0:
      if slice is None:
        if self.l4symtry:
          slice=0
        else:
          slice=self.nx/2
      if l_transpose:
        settitles(title,'Z','Y','t = %gs'%(top.time))
        kw.setdefault('xmin',self.zmmin)
        kw.setdefault('xmax',self.zmmax)
        kw.setdefault('ymin',self.ymmin)
        kw.setdefault('ymax',self.ymmax)
      else:
        settitles(title,'Y','Z','t = %gs'%(top.time))
        kw.setdefault('xmin',self.ymmin)
        kw.setdefault('xmax',self.ymmax)
        kw.setdefault('ymin',self.zmmin)
        kw.setdefault('ymax',self.zmmax)
      if l_transpose:
        data=transpose(data[slice,:,:])
      else:
        data=data[slice,:,:]
    if direction==1:
      if slice is None:
        if self.l4symtry:
          slice=0
        else:
          slice=self.ny/2
      if l_transpose:
        settitles(title,'Z','X','t = %gs'%(top.time))
        kw.setdefault('xmin',self.zmmin)
        kw.setdefault('xmax',self.zmmax)
        kw.setdefault('ymin',self.xmmin)
        kw.setdefault('ymax',self.xmmax)
      else:
        settitles(title,'X','Z','t = %gs'%(top.time))
        kw.setdefault('xmin',self.xmmin)
        kw.setdefault('xmax',self.xmmax)
        kw.setdefault('ymin',self.zmmin)
        kw.setdefault('ymax',self.zmmax)
      if l_transpose:
        data=transpose(data[:,slice,:])
      else:
        data=data[:,slice,:]
    if direction==2:
      if slice is None:slice=self.nz/2
      if l_transpose:
        settitles(title,'Y','X','t = %gs'%(top.time))
        kw.setdefault('xmin',self.ymmin)
        kw.setdefault('xmax',self.ymmax)
        kw.setdefault('ymin',self.xmmin)
        kw.setdefault('ymax',self.xmmax)
      else:
        settitles(title,'X','Y','t = %gs'%(top.time))
        kw.setdefault('xmin',self.xmmin)
        kw.setdefault('xmax',self.xmmax)
        kw.setdefault('ymin',self.ymmin)
        kw.setdefault('ymax',self.ymmax)
      if l_transpose:
        data=transpose(data[:,:,slice])
      else:
        data=data[:,:,slice]
    ppgeneric(grid=data,**kw)
      
  def gatherarray(self,g):
    f=self.fields
    if me<npes-1:
      return transpose(gatherarray(transpose(g[f.nxguard:-f.nxguard,f.nyguard:-f.nyguard,f.nzguard:-f.nzguard-1],[2,1,0])),[2,1,0])
    else:
      return transpose(gatherarray(transpose(g[f.nxguard:-f.nxguard,f.nyguard:-f.nyguard,f.nzguard:-f.nzguard],[2,1,0])),[2,1,0])
      
  def pfex(self,**kw):
      self.genericpfem3d(self.gatherarray(self.fields.Ex),'E_x',**kw)

  def pfey(self,**kw):
      self.genericpfem3d(self.gatherarray(self.fields.Ey),'E_y',**kw)

  def pfez(self,**kw):
      self.genericpfem3d(self.gatherarray(self.fields.Ez),'E_z',**kw)

  def pfbx(self,**kw):
      self.genericpfem3d(self.gatherarray(self.fields.Bx),'B_x',**kw)

  def pfby(self,**kw):
      self.genericpfem3d(self.gatherarray(self.fields.By),'B_y',**kw)

  def pfbz(self,**kw):
      self.genericpfem3d(self.gatherarray(self.fields.Bz),'B_z',**kw)

  def pfjx(self,**kw):
      self.genericpfem3d(self.gatherarray(self.fields.J[:,:,:,0]),'J_x',**kw)

  def pfjy(self,**kw):
      self.genericpfem3d(self.gatherarray(self.fields.J[:,:,:,1]),'J_y',**kw)

  def pfjz(self,**kw):
      self.genericpfem3d(self.gatherarray(self.fields.J[:,:,:,2]),'J_z',**kw)

  def pfrho(self,**kw):
      self.genericpfem3d(self.gatherarray(self.fields.Rho),'Rho',**kw)

  def pff(self,**kw):
      self.genericpfem3d(self.gatherarray(self.fields.F),'F',**kw)

  def sezax(self):
    pass

  def sphiax(self):
    pass

  def rhodia(self):
    pass

  def gtlchg(self):
    pass

  def srhoax(self):
    pass

  def getese(self):
    pass

  def getjx(self):
      return self.gatherarray(self.fields.J[:,:,:,0])

  def getjy(self):
      return self.gatherarray(self.fields.J[:,:,:,1])

  def getjz(self):
      return self.gatherarray(self.fields.J[:,:,:,2])

  def getex(self):
      return self.gatherarray(self.fields.Ex)

  def getey(self):
      return self.gatherarray(self.fields.Ey)
        
  def getez(self):
      return self.gatherarray(self.fields.Ez)
        
  def getbx(self):
      return self.gatherarray(self.fields.Bx)

  def getby(self):
      return self.gatherarray(self.fields.By)
        
  def getbz(self):
      return self.gatherarray(self.fields.Bz)
        
  def getrho(self):
      return self.gatherarray(self.fields.Rho)

  def getf(self):
      return self.gatherarray(self.fields.F)

def allocatesf(f):
    f.syf = EM3D_SPLITYEEFIELDtype()
    f.fieldtype = f.syf.fieldtype
    f.proc=me

def allocatef(f):
    f.yf = EM3D_YEEFIELDtype()
    f.fieldtype = f.yf.fieldtype
    f.proc=me

def pyinit_3dem_block(nx, ny, nz, 
                      nbndx, nbndy, nbndz, 
                      nxguard, nyguard, nzguard, 
                      dt, dx, dy, dz, 
                      clight, mu0, 
                      xmin, ymin, zmin, 
                      xlb, ylb, zlb, xrb, yrb, zrb, 
                      deposit_energy_density,
                      refinement,l_pushf):
  
  l_1d_decomposition=true
  
  b = EM3D_BLOCKtype()
  b.core = EM3D_FIELDtype()
  b.core.yf = EM3D_YEEFIELDtype()
  b.core.fieldtype = b.core.yf.fieldtype
  b.core.proc = me

  if npes>1:
    if l_1d_decomposition:
      if me==0:
        zrb = em3d.otherproc
        if zlb==periodic:
          zlb = em3d.otherproc
      elif me==npes-1:
        zlb = em3d.otherproc
        if zrb==periodic:
          zrb = em3d.otherproc
      else:
        zlb=zrb=em3d.otherproc

  b.nx = nx
  b.ny = ny
  b.nz = nz
  b.nbndx = nbndx
  b.nbndy = nbndy
  b.nbndz = nbndz
  b.nxguard = nxguard
  b.nyguard = nyguard
  b.nzguard = nzguard
  b.xmin = xmin
  b.ymin = ymin
  b.zmin = zmin
  b.dx = dx
  b.dy = dy
  b.dz = dz
  b.xmax = xmin+dx*nx
  b.ymax = ymin+dy*ny
  b.zmax = zmin+dz*nz
  b.dxi = 1./dx
  b.dyi = 1./dy
  b.dzi = 1./dz
  b.xlbnd = xlb
  b.xrbnd = xrb
  b.ylbnd = ylb
  b.yrbnd = yrb
  b.zlbnd = zlb
  b.zrbnd = zrb

  f=b.core.yf
  f.nx = nx
  f.ny = ny
  f.nz = nz
  if refinement is None:
    f.nxp = 0
    f.nyp = 0
    f.nzp = 0
  else:
    f.nxp = f.nx
    f.nyp = f.ny
    f.nzp = f.nz
  if not l_pushf:
    f.nxf = 0
    f.nyf = 0
    f.nzf = 0
  else:
    f.nxf = f.nx
    f.nyf = f.ny
    f.nzf = f.nz
  f.nxguard = nxguard
  f.nyguard = nyguard
  f.nzguard = nzguard
  # set min/max of cells positions with FORTRAN indexing
  f.ixmin = 0
  f.iymin = 0
  f.izmin = 0
  f.ixmax = f.nx
  f.iymax = f.ny
  f.izmax = f.nz
  f.ixming = -f.nxguard
  f.iyming = -f.nyguard
  f.izming = -f.nzguard
  f.ixmaxg = f.ixmax+f.nxguard
  f.iymaxg = f.iymax+f.nyguard
  f.izmaxg = f.izmax+f.nzguard
  # set min/max of cells positions with Python indexing
  f.jxmin = f.ixmin-f.ixming
  f.jymin = f.iymin-f.iyming
  f.jzmin = f.izmin-f.izming
  f.jxmax = f.ixmax-f.ixming
  f.jymax = f.iymax-f.iyming
  f.jzmax = f.izmax-f.izming
  f.jxming = 0
  f.jyming = 0
  f.jzming = 0
  f.jxmaxg = f.ixmaxg-f.ixming
  f.jymaxg = f.iymaxg-f.iyming
  f.jzmaxg = f.izmaxg-f.izming
  f.xmin = xmin
  f.ymin = ymin
  f.zmin = zmin
  f.dx = dx
  f.dy = dy
  f.dz = dz
  f.xmax = xmin+dx*nx
  f.ymax = ymin+dy*ny
  f.zmax = zmin+dz*nz
  f.dxi = 1./dx
  f.dyi = 1./dy
  f.dzi = 1./dz
  f.clight = clight
  f.mu0    = mu0

  if deposit_energy_density:
    f.nxmp=f.nx
    f.nymp=f.ny
    f.nzmp=f.nz

  f.gchange()
  
  if refinement is None:
    f.nxp = f.nx
    f.nyp = f.ny
    f.nzp = f.nz
    f.Exp = f.Ex
    f.Eyp = f.Ey
    f.Ezp = f.Ez
    f.Bxp = f.Bx
    f.Byp = f.By
    f.Bzp = f.Bz

  nnx=em3d.nn
  nny=em3d.nn
  nnz=em3d.nn
  smaxx=em3d.s_max_init
  smaxy=em3d.s_max_init
  smaxz=em3d.s_max_init
  sdeltax=em3d.s_delta
  sdeltay=em3d.s_delta
  sdeltaz=em3d.s_delta
  
# *** sides
# x
  b.sidexl = EM3D_FIELDtype()
  if xlb==openbc:
    allocatesf(b.sidexl)
    init_splitfield(b.sidexl.syf,nbndx,ny,nz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight,-1, 0, 0, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if xlb==periodic:
    b.sidexl=b.core
  b.sidexr = EM3D_FIELDtype()
  if xrb==openbc:
    allocatesf(b.sidexr)
    init_splitfield(b.sidexr.syf,nbndx,ny,nz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 1, 0, 0, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if xrb==periodic:
    b.sidexr=b.core
# y
  b.sideyl = EM3D_FIELDtype()
  if ylb==openbc:
    allocatesf(b.sideyl)
    init_splitfield(b.sideyl.syf,nx,nbndy,nz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 0,-1, 0, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if ylb==periodic:
    b.sideyl=b.core
  b.sideyr = EM3D_FIELDtype()
  if yrb==openbc:
    allocatesf(b.sideyr)
    init_splitfield(b.sideyr.syf,nx,nbndy,nz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 0, 1, 0, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if yrb==periodic:
    b.sideyr=b.core
# z
  b.sidezl = EM3D_FIELDtype()
  if zlb==openbc:
    allocatesf(b.sidezl)
    init_splitfield(b.sidezl.syf,nx,ny,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 0, 0,-1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zlb==periodic:
    b.sidezl=b.core
  if zlb==em3d.otherproc:
    allocatef(b.sidezl)
    if l_1d_decomposition:
      b.sidezl.proc=(me-1)%npes      
  b.sidezr = EM3D_FIELDtype()
  if zrb==openbc:
    allocatesf(b.sidezr)
    init_splitfield(b.sidezr.syf,nx,ny,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 0, 0, 1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zrb==periodic:
    b.sidezr=b.core
  if zrb==em3d.otherproc:
    allocatef(b.sidezr)
    if l_1d_decomposition:
      b.sidezr.proc=(me+1)%npes      
    
# *** edges
# xy
  b.edgexlyl = EM3D_FIELDtype()
  if xlb==openbc and ylb==openbc:
    allocatesf(b.edgexlyl)
    init_splitfield(b.edgexlyl.syf,nbndx,nbndy,nz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight,-1,-1, 0, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
# partial periodic bc to be completed 
#  if xlb==openbc and ylb==periodic:
#    b.edgexlyl=b.sidexl
#  if ylb==openbc and xlb==periodic:
#    b.edgexlyl=b.sideyl
  b.edgexryl = EM3D_FIELDtype()
  if xrb==openbc and ylb==openbc:
    allocatesf(b.edgexryl)
    init_splitfield(b.edgexryl.syf,nbndx,nbndy,nz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 1,-1, 0, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  b.edgexlyr = EM3D_FIELDtype()
  if xlb==openbc and yrb==openbc:
    allocatesf(b.edgexlyr)
    init_splitfield(b.edgexlyr.syf,nbndx,nbndy,nz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight,-1, 1, 0, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  b.edgexryr = EM3D_FIELDtype()
  if xrb==openbc and yrb==openbc:
    allocatesf(b.edgexryr)
    init_splitfield(b.edgexryr.syf,nbndx,nbndy,nz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 1, 1, 0, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
# xz
  b.edgexlzl = EM3D_FIELDtype()
  if xlb==openbc and zlb==openbc:
    allocatesf(b.edgexlzl)
    init_splitfield(b.edgexlzl.syf,nbndx,ny,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight,-1, 0,-1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zlb==em3d.otherproc:
    allocatesf(b.edgexlzl)
    if l_1d_decomposition:
      b.edgexlzl.proc=(me-1)%npes      
  b.edgexrzl = EM3D_FIELDtype()
  if xrb==openbc and zlb==openbc:
    allocatesf(b.edgexrzl)
    init_splitfield(b.edgexrzl.syf,nbndx,ny,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 1, 0,-1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zlb==em3d.otherproc:
    allocatesf(b.edgexrzl)
    if l_1d_decomposition:
      b.edgexrzl.proc=(me-1)%npes      
  b.edgexlzr = EM3D_FIELDtype()
  if xlb==openbc and zrb==openbc:
    allocatesf(b.edgexlzr)
    init_splitfield(b.edgexlzr.syf,nbndx,ny,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight,-1, 0, 1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zrb==em3d.otherproc:
    allocatesf(b.edgexlzr)
    if l_1d_decomposition:
      b.edgexlzr.proc=(me+1)%npes      
  b.edgexrzr = EM3D_FIELDtype()
  if xrb==openbc and zrb==openbc:
    allocatesf(b.edgexrzr)
    init_splitfield(b.edgexrzr.syf,nbndx,ny,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 1, 0, 1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zrb==em3d.otherproc:
    allocatesf(b.edgexrzr)
    if l_1d_decomposition:
      b.edgexrzr.proc=(me+1)%npes      
# yz
  b.edgeylzl = EM3D_FIELDtype()
  if ylb==openbc and zlb==openbc:
    allocatesf(b.edgeylzl)
    init_splitfield(b.edgeylzl.syf,nx,nbndy,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 0,-1,-1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zlb==em3d.otherproc:
    allocatesf(b.edgeylzl)
    if l_1d_decomposition:
      b.edgeylzl.proc=(me-1)%npes      
  b.edgeyrzl = EM3D_FIELDtype()
  if yrb==openbc and zlb==openbc:
    allocatesf(b.edgeyrzl)
    init_splitfield(b.edgeyrzl.syf,nx,nbndy,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 0, 1,-1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zlb==em3d.otherproc:
    allocatesf(b.edgeyrzl)
    if l_1d_decomposition:
      b.edgeyrzl.proc=(me-1)%npes      
  b.edgeylzr = EM3D_FIELDtype()
  if ylb==openbc and zrb==openbc:
    allocatesf(b.edgeylzr)
    init_splitfield(b.edgeylzr.syf,nx,nbndy,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 0,-1, 1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zrb==em3d.otherproc:
    allocatesf(b.edgeylzr)
    if l_1d_decomposition:
      b.edgeylzr.proc=(me+1)%npes      
  b.edgeyrzr = EM3D_FIELDtype()
  if yrb==openbc and zrb==openbc:
    allocatesf(b.edgeyrzr)
    init_splitfield(b.edgeyrzr.syf,nx,nbndy,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 0, 1, 1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zrb==em3d.otherproc:
    allocatesf(b.edgeyrzr)
    if l_1d_decomposition:
      b.edgeyrzr.proc=(me+1)%npes      

# *** corners
  b.cornerxlylzl = EM3D_FIELDtype()
  if xlb==openbc and ylb==openbc and zlb==openbc:
    allocatesf(b.cornerxlylzl)
    init_splitfield(b.cornerxlylzl.syf,nbndx,nbndy,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight,-1,-1,-1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zlb==em3d.otherproc:
    allocatesf(b.cornerxlylzl)
    if l_1d_decomposition:
      b.cornerxlylzl.proc=(me-1)%npes      
  b.cornerxrylzl = EM3D_FIELDtype()
  if xrb==openbc and ylb==openbc and zlb==openbc:
    allocatesf(b.cornerxrylzl)
    init_splitfield(b.cornerxrylzl.syf,nbndx,nbndy,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 1,-1,-1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zlb==em3d.otherproc:
    allocatesf(b.cornerxrylzl)
    if l_1d_decomposition:
      b.cornerxrylzl.proc=(me-1)%npes      
  b.cornerxlyrzl = EM3D_FIELDtype()
  if xlb==openbc and yrb==openbc and zlb==openbc:
    allocatesf(b.cornerxlyrzl)
    init_splitfield(b.cornerxlyrzl.syf,nbndx,nbndy,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight,-1, 1,-1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zlb==em3d.otherproc:
    allocatesf(b.cornerxlyrzl)
    if l_1d_decomposition:
      b.cornerxlyrzl.proc=(me-1)%npes      
  b.cornerxryrzl = EM3D_FIELDtype()
  if xrb==openbc and yrb==openbc and zlb==openbc:
    allocatesf(b.cornerxryrzl)
    init_splitfield(b.cornerxryrzl.syf,nbndx,nbndy,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 1, 1,-1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zlb==em3d.otherproc:
    allocatesf(b.cornerxryrzl)
    if l_1d_decomposition:
      b.cornerxryrzl.proc=(me-1)%npes      
  b.cornerxlylzr = EM3D_FIELDtype()
  if xlb==openbc and ylb==openbc and zrb==openbc:
    allocatesf(b.cornerxlylzr)
    init_splitfield(b.cornerxlylzr.syf,nbndx,nbndy,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight,-1,-1, 1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zrb==em3d.otherproc:
    allocatesf(b.cornerxlylzr)
    if l_1d_decomposition:
      b.cornerxlylzr.proc=(me+1)%npes      
  b.cornerxrylzr = EM3D_FIELDtype()
  if xrb==openbc and ylb==openbc and zrb==openbc:
    allocatesf(b.cornerxrylzr)
    init_splitfield(b.cornerxrylzr.syf,nbndx,nbndy,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 1,-1, 1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zrb==em3d.otherproc:
    allocatesf(b.cornerxrylzr)
    if l_1d_decomposition:
      b.cornerxrylzr.proc=(me+1)%npes      
  b.cornerxlyrzr = EM3D_FIELDtype()
  if xlb==openbc and yrb==openbc and zrb==openbc:
    allocatesf(b.cornerxlyrzr)
    init_splitfield(b.cornerxlyrzr.syf,nbndx,nbndy,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight,-1, 1, 1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zrb==em3d.otherproc:
    allocatesf(b.cornerxlyrzr)
    if l_1d_decomposition:
      b.cornerxlyrzr.proc=(me+1)%npes      
  b.cornerxryrzr = EM3D_FIELDtype()
  if xrb==openbc and yrb==openbc and zrb==openbc:
    allocatesf(b.cornerxryrzr)
    init_splitfield(b.cornerxryrzr.syf,nbndx,nbndy,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 1, 1, 1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zrb==em3d.otherproc:
    allocatesf(b.cornerxryrzr)
    if l_1d_decomposition:
      b.cornerxryrzr.proc=(me+1)%npes      

  return b

class EM3Dold(FieldSolver):
  
#  __w3dinputs__ = ['solvergeom','nx','ny','nzlocal','nz',
#                   'xmmin','xmmax','ymmin','ymmax','zmmin','zmmax',
#                   'zmminlocal','zmmaxlocal',
#                   'bound0','boundnz','boundxy','l2symtry','l4symtry',
#                   'solvergeom']
  __em3dinputs__ = []
#  __em3dinputs__ = ['l_onegrid','l_copyfields','l_moving_window',
#                    'tmin_moving_main_window','l_smoothdensity',
#                    'l_elaser_out_plane','ndelta_t',
#                   'ntamp_scatter','ntamp_gather']
#  __topinputs__ = ['pbound0','pboundnz','pboundxy',
#                   'my_index','nslaves','lfsautodecomp','zslave','lautodecomp',
#                   'debug']
  __flaginputs__ = {'l_apply_pml':true,'nbndx':10,'nbndy':10,'nbndz':10,
                    'nxguard':1,'nyguard':1,'nzguard':1,
                    'l_particles_weight':false,'l_usecoeffs':false,
                    'l_pushf':false,'l_pushpot':false,'l_verbose':false,
                    'laser_amplitude':1.,'laser_profile':None,
                    'laser_gauss_width':None,'laser_angle':0.,
                    'laser_wavelength':None,'laser_wavenumber':None,
                    'laser_frequency':None,'laser_source':2,
                    'laser_focus':None,'laser_focus_velocity':0.,
                    'nfield_subcycle':1,'density_1d':0,
                    'autoset_timestep':true,'dtcoef':1.,#0.99,
                    'deposit_energy_density':false}

  def __init__(self,**kw):
    self.grid_overlap = 1
    FieldSolver.__init__(self,kwdict=kw)
#    top.allspecl = true
    top.lcallfetchb = true
    top.lgridqnt = true
    self.zgridprv=top.zgrid
    # --- Make sure the refinement is turned off

    # --- Save input parameters
#    self.processdefaultsfrompackage(EM3D.__w3dinputs__,w3d,kw)
#    self.processdefaultsfrompackage(EM3D.__topinputs__,top,kw)
    self.processdefaultsfrompackage(EM3D.__em3dinputs__,em3d,kw)
    self.processdefaultsfromdict(EM3D.__flaginputs__,kw)

    # --- bounds is special since it will sometimes be set from the
    # --- variables bound0, boundnz, boundxy, l2symtry, and l4symtry
    if 'bounds' not in self.__dict__:
      if 'bounds' in kw:
        self.bounds = kw['bounds']
      else:
        self.bounds = zeros(6,'l')
        self.bounds[0] = self.boundxy
        self.bounds[1] = self.boundxy
        self.bounds[2] = self.boundxy
        self.bounds[3] = self.boundxy
        self.bounds[4] = self.bound0
        self.bounds[5] = self.boundnz
        if self.l2symtry:
          self.bounds[2] = neumann
          if self.boundxy == periodic: self.bounds[3] = neumann
          if self.forcesymmetries: self.ymmin = 0.
        elif self.l4symtry:
          self.bounds[0] = neumann
          self.bounds[2] = neumann
          if self.boundxy == periodic: self.bounds[1] = neumann
          if self.boundxy == periodic: self.bounds[3] = neumann
          if self.forcesymmetries: self.xmmin = 0.
          if self.forcesymmetries: self.ymmin = 0.

    # --- If there are any remaning keyword arguments, raise an error.
    assert len(kw.keys()) == 0,"Bad keyword arguemnts %s"%kw.keys()

    # --- Calculate mesh sizes
    self.dx = (self.xmmax - self.xmmin)/self.nx
    self.xmesh = self.xmmin + arange(0,self.nx+1)*self.dx
    self.dy = (self.ymmax - self.ymmin)/self.ny
    self.ymesh = self.ymmin + arange(0,self.ny+1)*self.dy
    self.dz = (self.zmmax - self.zmmin)/self.nz
    self.zmesh = self.zmmin + arange(0,self.nz+1)*self.dz
#    self.zmeshlocal = self.zmminlocal + arange(0,self.nzlocal+1)*self.dz

    # --- set time step as a fraction of Courant condition
    # --- also set self.nfield_subcycle if top.dt over Courant condition times dtcoef
    if self.autoset_timestep:
      dt=self.dtcoef/(clight*sqrt(1./self.dx**2+1./self.dy**2+1./self.dz**2))
      if top.dt==0.:
        top.dt=dt
      else:
        if top.dt<>dt:
          self.nfield_subcycle=int(top.dt/dt)+1
    self.dtinit = top.dt
    
    # --- Create field and source arrays and other arrays.
    self.allocatefieldarrays()
    self.initializeconductors()

    # --- Handle laser inputs
    self.setuplaser()

    installbeforeloadrho(self.solve2ndhalf)
    registersolver(self)

  def processdefaultsfrompackage(self,defaults,package,kw):
    for name in defaults:
      if name not in self.__dict__:
        #self.__dict__[name] = kw.pop(name,getattr(w3d,name)) # Python2.3
        self.__dict__[name] = kw.get(name,getattr(package,name))
      if kw.has_key(name): del kw[name]

  def processdefaultsfromdict(self,dict,kw):
    for name,defvalue in dict.iteritems():
      if name not in self.__dict__:
        #self.__dict__[name] = kw.pop(name,getattr(top,name)) # Python2.3
        self.__dict__[name] = kw.get(name,defvalue)
      if kw.has_key(name): del kw[name]

  def allocatefieldarrays(self):
    self.block = pyinit_3dem_blockold(self.nx, 
                                   self.ny, 
                                   self.nzlocal, 
                                   self.nbndx, 
                                   self.nbndy, 
                                   self.nbndz, 
                                   self.nxguard, 
                                   self.nyguard, 
                                   self.nzguard, 
                                   top.dt, 
                                   self.dx, 
                                   self.dy, 
                                   self.dz, 
                                   clight, 
                                   mu0, 
                                   self.xmmin, 
                                   self.ymmin, 
                                   self.zmminlocal, 
                                   self.bounds[0], 
                                   self.bounds[2], 
                                   self.bounds[4], 
                                   self.bounds[1], 
                                   self.bounds[3], 
                                   self.bounds[5],
                                   self.deposit_energy_density)
    self.fields = self.block.core.yf
    
  def addpatch(self,ixpatch,iypatch,nxpatch,nypatch,rap):
    # --- Initializes refinement patch
    em3d.l_onegrid = false
    self.l_onegrid = em3d.l_onegrid
    self.fpatches.append([EM3D_FIELDtype(),EM3D_FIELDtype()])
    self.fpatchcoarse = self.fpatches[-1][0]
    self.fpatchfine   = self.fpatches[-1][1]
    self.fpatchfine.rap = rap
    xlb = xrb = ylb = yrb = absorb
    if ixpatch==0:xlb = self.bounds[0]
    if iypatch==0:ylb = self.bounds[1]
    if ixpatch+nxpatch==self.nx:xrb = self.bounds[2]
    if iypatch+nypatch==self.ny:yrb = self.bounds[3]
    init_fields(self.fpatchcoarse,nxpatch,nypatch,self.nbndx,self.nbndx,top.dt/self.nfield_subcycle,
                self.dx,self.dy,clight,mu0,
                self.xmmin+ixpatch*self.dx,self.ymmin+iypatch*self.dy,1,
                xlb,ylb,xrb,yrb)
    init_fields(self.fpatchfine,nxpatch*rap,nypatch*rap,self.nbndx,self.nbndx,top.dt/self.nfield_subcycle,
                self.dx/rap,self.dy/rap,clight,mu0,
                self.xmmin+ixpatch*self.dx,self.ymmin+iypatch*self.dy,rap,
                xlb,ylb,xrb,yrb)
    self.fpatchcoarse.js = self.laser_source
    self.fpatchfine.js = self.laser_source*rap
    self.fpatchfine.xminpatch_scatter = self.fpatchfine.xmin+em3d.ntamp_scatter*rap*self.fpatchfine.dx
    self.fpatchfine.xmaxpatch_scatter = self.fpatchfine.xmax-em3d.ntamp_scatter*rap*self.fpatchfine.dx
    self.fpatchfine.yminpatch_scatter = self.fpatchfine.ymin+em3d.ntamp_scatter*rap*self.fpatchfine.dy
    self.fpatchfine.ymaxpatch_scatter = self.fpatchfine.ymax-em3d.ntamp_scatter*rap*self.fpatchfine.dy
    self.fpatchfine.xminpatch_gather = self.fpatchfine.xmin+em3d.ntamp_gather*rap*self.fpatchfine.dx
    self.fpatchfine.xmaxpatch_gather = self.fpatchfine.xmax-em3d.ntamp_gather*rap*self.fpatchfine.dx
    self.fpatchfine.yminpatch_gather = self.fpatchfine.ymin+em3d.ntamp_gather*rap*self.fpatchfine.dy
    self.fpatchfine.ymaxpatch_gather = self.fpatchfine.ymax-em3d.ntamp_gather*rap*self.fpatchfine.dy
    self.fpatchfine.nxfsum = self.fpatchfine.nx
    self.fpatchfine.nyfsum = self.fpatchfine.ny
    self.fpatchfine.gchange()
    self.setuplaser_profile(self.fpatchcoarse)
    self.setuplaser_profile(self.fpatchfine)

  def setuplaser(self):
    if self.laser_profile is not None:
      if self.laser_frequency is None:
        if self.laser_wavenumber is not None:
          self.laser_wavelength = clight*self.laser_wavenumber
        elif self.laser_wavelength is not None:
          self.laser_wavelength = 2.*pi*clight/self.laser_wavelength
      assert self.laser_frequency is not None,\
             "One of the frequency, wavenumber, or wavelength must be given"

    # --- Check if laser_amplitude is a function, table, or constant
    self.laser_amplitude_func = None
    self.laser_amplitude_table = None
    if operator.isSequenceType(self.laser_amplitude):
      assert len(self.laser_amplitude.shape) == 2 and \
             self.laser_amplitude.shape[1] == 2,\
             "The laser_amplitude table is not formatted properly"
      self.laser_amplitude_table = self.laser_amplitude
      self.laser_amplitude_table_i = -1
    elif callable(self.laser_amplitude):
      self.laser_amplitude_func = self.laser_amplitude
      
    self.setuplaser_profile(self.fields)

  def setuplaser_profile(self,f):
    # --- Check if laser_profile has a type, is a function, or a table
    self.laser_profile_func = None
    # --- disable laser emission on processors id>0 
    if me>0:return
    if self.laser_profile == 'gaussian':
      assert self.laser_gauss_width is not None,\
             "For a gaussian laser, the width must be specified using laser_gauss_width"
#      xx = arange(self.nx+4)*f.dx+f.xmin*f.dx - 0.5*f.nx*f.dx
#      self.laser_profile = exp(-(xx/self.laser_gauss_width)**2/2.)
      yy = arange(f.ny+3)*f.dy+f.ymin*f.dy - 0.5*f.ny*f.dy
      self.laser_profile = exp(-(yy/self.laser_gauss_width)**2/2.)
    elif operator.isSequenceType(self.laser_profile):
      assert len(self.laser_profile) == f.ny+3,"The specified profile must be of length ny+3"
    elif callable(self.laser_profile):
      self.laser_profile_func = self.laser_profile
    
  def fetchefrompositions(self,x,y,z,ex,ey,ez,nox,noy,noz):
    n = len(x)
    if n == 0: return
    f = self.block.core.yf
    if top.efetch[w3d.jsfsapi] in [1,3,5]:
      getf3d_linear(n,x,y,z,ex,ey,ez,
                    f.xmin,f.ymin,f.zmin,
                    f.dx,f.dy,f.dz,
                    f.nx,f.ny,f.nz,
                    f.nxguard,f.nyguard,f.nzguard,
                    f.Exp,f.Eyp,f.Ezp)
    elif top.efetch[w3d.jsfsapi]==4:
      gete3d_n_energy_conserving(n,x,y,z,ex,ey,ez,
                    f.xmin,f.ymin,f.zmin,
                    f.dx,f.dy,f.dz,
                    f.nx,f.ny,f.nz,
                    f.nxguard,f.nyguard,f.nzguard,
                    nox,noy,noz,
                    f.Exp,f.Eyp,f.Ezp,
                    w3d.l4symtry)

  def fetchbfrompositions(self,x,y,z,bx,by,bz,nox,noy,noz):
    n = len(x)
    if n == 0: return
    f = self.block.core.yf
    if top.efetch[w3d.jsfsapi] in [1,3,5]:
      getf3d_linear(n,x,y,z,bx,by,bz,
                    f.xmin,f.ymin,f.zmin,
                    f.dx,f.dy,f.dz,
                    f.nx,f.ny,f.nz,
                    f.nxguard,f.nyguard,f.nzguard,
                    f.Bxp,f.Byp,f.Bzp)
    elif top.efetch[w3d.jsfsapi]==4:
      getb3d_n_energy_conserving(n,x,y,z,bx,by,bz,
                    f.xmin,f.ymin,f.zmin,
                    f.dx,f.dy,f.dz,
                    f.nx,f.ny,f.nz,
                    f.nxguard,f.nyguard,f.nzguard,
                    nox,noy,noz,
                    f.Bxp,f.Byp,f.Bzp,
                    w3d.l4symtry)
    
  def fetchphifrompositions(self,x,z,phi):
    pass

  def loadrho(self,lzero=true):
    if not self.l_pushf:return
    if self.l_verbose:print 'loadrho',lzero
    # --- reallocate Jarray if needed
    if self.fields.ntimes<>top.nsndts:
      self.fields.ntimes=top.nsndts
      self.fields.gchange()
      force_deposition=true
    else:
      force_deposition=false

    # --- zero proper portion of Rhoarray
    if lzero: 
      for i in range(top.nsndts-1,-1,-1):
        if force_deposition or (top.it-1)%(2**i)==0:
          self.fields.Rhoarray[:,:,:,i] = 0.

    # --- loop over species
    for js,i,n,q,w in zip(arange(top.pgroup.ns),top.pgroup.ins-1,top.pgroup.nps,
                       top.pgroup.sq,top.pgroup.sw):
      if n == 0 or ((top.it-1)%top.pgroup.ndts[js]<>0 and not force_deposition): continue
      if top.wpid==0:
        wfact = zeros(n,'d')
        l_particles_weight = false
      else:
        wfact = top.pgroup.pid[i:i+n,top.wpid-1]
        l_particles_weight = true
      # --- point rho array to proper Jarray slice
      self.fields.Rho = self.fields.Rhoarray[:,:,:,top.ndtstorho[top.pgroup.ndts[js]-1]]
      # --- call routine performing current deposition
      f = self.block.core.yf
      depose_rho_n(self.fields.Rho,n,
                               top.pgroup.xp[i:i+n],
                               top.pgroup.yp[i:i+n],
                               top.pgroup.zp[i:i+n],
                               wfact,q*w,
                               f.xmin,f.ymin,f.zmin,
                               f.dx,f.dy,f.dz,
                               f.nx,f.ny,f.nz,
                               f.nxguard,f.nyguard,f.nzguard,
                               top.depos_order[0,js],
                               top.depos_order[1,js],
                               top.depos_order[2,js],
                               l_particles_weight,w3d.l4symtry)

    # --- add slices
    if top.nsndts>1:
      for i in range(top.nsndts-2,-1,-1):
        if force_deposition or (top.it-1)%(2**i)==0:
          add_rho_slice_3d(self.fields,i+1)

    # --- apply boundary condition on current
    self.apply_rho_bc()

    # --- point Rho to first slice of Rhoarray
    self.fields.Rho = self.fields.Rhoarray[:,:,:,0]

  def apply_rho_bc(self):
    # --- point Rho to first slice of Rhoarray
    self.fields.Rho = self.fields.Rhoarray[:,:,:,0]
    em3d_applybc_rho(self.block.core.yf, 
                     self.block.xlbnd,
                     self.block.xrbnd,
                     self.block.ylbnd,
                     self.block.yrbnd,
                     self.block.zlbnd,
                     self.block.zrbnd)
    em3d_exchange_rho(self.block)

  def loadj(self,ins_i=-1,nps_i=-1,is_i=-1,lzero=true):
    if self.l_verbose:print 'loadj'
    # --- reallocate Jarray if needed
    if self.fields.ntimes<>top.nsndts:
      self.fields.ntimes=top.nsndts
      self.fields.gchange()
      force_deposition=true
    else:
      force_deposition=false

    # --- zero proper portion of Jarray
    if lzero: 
      for i in range(top.nsndts-1,-1,-1):
        if force_deposition or (top.it-1)%(2**i)==0:
          self.fields.Jarray[:,:,:,:,i] = 0.
          if self.deposit_energy_density:
            self.fields.Mp[...] = 0.
        
    # --- loop over species
    for js,i,n,q,m,w in zip(arange(top.pgroup.ns),top.pgroup.ins-1,top.pgroup.nps,
                       top.pgroup.sq,top.pgroup.sm,top.pgroup.sw):
      if n == 0 or ((top.it-1)%top.pgroup.ndts[js]<>0): continue
      if top.wpid==0:
        wfact = w
        l_particles_weight = false
      else:
        wfact = w*top.pgroup.pid[i:i+n,top.wpid-1]
        l_particles_weight = true
      # --- point J array to proper Jarray slice
      self.fields.J = self.fields.Jarray[:,:,:,:,top.ndtstorho[top.pgroup.ndts[js]-1]]
      # --- call routine performing current deposition
      f = self.block.core.yf
      if 1:
        if not self.deposit_energy_density:
          depose_jxjyjz_esirkepov_n(self.fields.J,n,
                                            top.pgroup.xp[i:i+n],
                                            top.pgroup.yp[i:i+n],
                                            top.pgroup.zp[i:i+n],
                                            top.pgroup.uxp[i:i+n],
                                            top.pgroup.uyp[i:i+n],
                                            top.pgroup.uzp[i:i+n],
                                            top.pgroup.gaminv[i:i+n],wfact,q,
                                            f.xmin,f.ymin,f.zmin,
                                            top.dt*top.pgroup.ndts[js],
                                            f.dx,f.dy,f.dz,
                                            f.nx,f.ny,f.nz,
                                            f.nxguard,f.nyguard,f.nzguard,
                                            top.depos_order[0,js],
                                            top.depos_order[1,js],
                                            top.depos_order[2,js],
                                            l_particles_weight,w3d.l4symtry)
        else:
          depose_jxjyjz_pxpypz_esirkepov_linear_serial(self.fields.J,self.fields.Mp,n,
                                            top.pgroup.xp[i:i+n],
                                            top.pgroup.yp[i:i+n],
                                            top.pgroup.zp[i:i+n],
                                            top.pgroup.uxp[i:i+n],
                                            top.pgroup.uyp[i:i+n],
                                            top.pgroup.uzp[i:i+n],
                                            top.pgroup.gaminv[i:i+n],wfact,q,m,
                                            f.xmin,f.ymin,f.zmin,
                                            top.dt*top.pgroup.ndts[js],
                                            f.dx,f.dy,f.dz,
                                            f.nx,f.ny,f.nz,
                                            f.nxguard,f.nyguard,f.nzguard,
                                            l_particles_weight,
                                            top.lrelativ)
      else:
        Jx=self.fields.J[1,1,:,0].copy()
        Jy=self.fields.J[1,1,:,1].copy()
        Jz=self.fields.J[1,1,:,2].copy()
        depose_jxjyjz_esirkepov_linear_serial1d(Jx,Jy,Jz,n,
                                            top.pgroup.xp[i:i+n],
                                            top.pgroup.yp[i:i+n],
                                            top.pgroup.zp[i:i+n],
                                            top.pgroup.uxp[i:i+n],
                                            top.pgroup.uyp[i:i+n],
                                            top.pgroup.uzp[i:i+n],
                                            top.pgroup.gaminv[i:i+n],wfact,q,
                                            f.xmin,f.ymin,f.zmin,
                                            top.dt*top.pgroup.ndts[js],
                                            f.dx,f.dy,f.dz,
                                            f.nx,f.ny,f.nz,
                                            f.nxguard,f.nyguard,f.nzguard,
                                            l_particles_weight)
        self.fields.J[1,1,:,0]+=Jx
        self.fields.J[1,1,:,1]+=Jy
        self.fields.J[1,1,:,2]+=Jz
    # --- add slices
    if top.nsndts>1:
      for i in range(top.nsndts-2,-1,-1):
        if force_deposition or (top.it-1)%(2**i)==0:
          add_current_slice_3d(self.fields,i+1)

    # --- apply boundary condition on current
    if not self.density_1d:
      self.apply_current_bc()

    # --- point J to first slice of Jarray
    self.fields.J = self.fields.Jarray[:,:,:,:,0]
    
    # --- get 1-D density
    if self.density_1d:
      for i in range(3):
       J = sum(sum(self.fields.J[:,:,:,i],0),0)
       for ii in range(shape(self.fields.J[:,:,:,i])[1]):
        for jj in range(shape(self.fields.J[:,:,:,i])[0]):
         self.fields.J[ii,jj,:,i] = J

    # --- smooth current density 
#    if self.l_smoothdensity:self.smoothdensity()
    if self.l_verbose:print 'loadj done'
#    print 'sum',sum(self.fields.Mp[...,-1]*w3d.dz), \
#          sum(emass*(1./where(top.pgroup.gaminv==0.,1.,top.pgroup.gaminv)-1.)*top.pgroup.sw[0]*clight**2), \
#          sum(self.fields.J[...,-1]**2)*(top.dt/eps0)**2*eps0*w3d.dz/2
#    print 'max',maxnd(self.fields.Mp[...,-1]*w3d.dz), \
#          maxnd(emass*(1./where(top.pgroup.gaminv==0.,1.,top.pgroup.gaminv)-1.)*top.pgroup.sw[0]*clight**2), \
#          maxnd(self.fields.J[...,-1]**2)*(top.dt/eps0)**2*eps0*w3d.dz/2


  def apply_current_bc(self):
    # --- point J to first slice of Jarray
    self.fields.J = self.fields.Jarray[:,:,:,:,0]
    em3d_applybc_j(self.block.core.yf, 
                   self.block.xlbnd,
                   self.block.xrbnd,
                   self.block.ylbnd,
                   self.block.yrbnd,
                   self.block.zlbnd,
                   self.block.zrbnd)
    em3d_exchange_j(self.block)

  def smoothdensity(self):
    smooth2d_lindman(self.fields.J[:,:,0],self.fields.nx,self.fields.ny)
    smooth2d_lindman(self.fields.J[:,:,1],self.fields.nx,self.fields.ny)
    smooth2d_lindman(self.fields.J[:,:,2],self.fields.nx,self.fields.ny)

  def fetche(self):
    if self.l_verbose:print 'fetche'
    x=top.pgroup.xp[w3d.ipminfsapi-1:w3d.ipminfsapi-1+w3d.npfsapi]
    y=top.pgroup.yp[w3d.ipminfsapi-1:w3d.ipminfsapi-1+w3d.npfsapi]
    z=top.pgroup.zp[w3d.ipminfsapi-1:w3d.ipminfsapi-1+w3d.npfsapi]
    ex = top.pgroup.ex[w3d.ipminfsapi-1:w3d.ipminfsapi-1+w3d.npfsapi]
    ey = top.pgroup.ey[w3d.ipminfsapi-1:w3d.ipminfsapi-1+w3d.npfsapi]
    ez = top.pgroup.ez[w3d.ipminfsapi-1:w3d.ipminfsapi-1+w3d.npfsapi]
    ex[:] = 0.
    ey[:] = 0.
    ez[:] = 0.
    nox = top.depos_order[0,w3d.jsfsapi]
    noy = top.depos_order[1,w3d.jsfsapi]
    noz = top.depos_order[2,w3d.jsfsapi]
    self.fetchefrompositions(x,y,z,ex,ey,ez,nox,noy,noz)
#    print 'e',ex,ey,ez

  def fetchb(self):
    if self.l_verbose:print 'fetchb'
    x=top.pgroup.xp[w3d.ipminfsapi-1:w3d.ipminfsapi-1+w3d.npfsapi]
    y=top.pgroup.yp[w3d.ipminfsapi-1:w3d.ipminfsapi-1+w3d.npfsapi]
    z=top.pgroup.zp[w3d.ipminfsapi-1:w3d.ipminfsapi-1+w3d.npfsapi]
    bx = top.pgroup.bx[w3d.ipminfsapi-1:w3d.ipminfsapi-1+w3d.npfsapi]
    by = top.pgroup.by[w3d.ipminfsapi-1:w3d.ipminfsapi-1+w3d.npfsapi]
    bz = top.pgroup.bz[w3d.ipminfsapi-1:w3d.ipminfsapi-1+w3d.npfsapi]
    bx[:] = 0.
    by[:] = 0.
    bz[:] = 0.
    nox = top.depos_order[0,w3d.jsfsapi]
    noy = top.depos_order[1,w3d.jsfsapi]
    noz = top.depos_order[2,w3d.jsfsapi]
    self.fetchbfrompositions(x,y,z,bx,by,bz,nox,noy,noz)
    if self.l_verbose:print 'fetchb done.'
#    print 'b',bx,by,bz,top.pgroup.bz

  def fetchphi(self):
    pass

  def fetcha(self):
    pass

  def initializeconductors(self):
    # --- Create the attributes for holding information about conductors
    # --- and conductor objects.
    # --- Note that a conductor object will be created for each value of
    # --- fselfb. This is needed since fselfb effects how the coarsening
    # --- is done, and different conductor data sets are needed for
    # --- different coarsenings.

    # --- This stores the ConductorType objects. Note that the objects are
    # --- not actually created until getconductorobject is called.
    self.conductorobjects = {}

    # --- This stores the conductors that have been installed in each
    # --- of the conductor objects.
    self.installedconductorlists = {}

    # --- This is a list of conductors that have been added.
    # --- New conductors are not actually installed until the data is needed,
    # --- when getconductorobject is called.
    # --- Each element of this list contains all of the input to the
    # --- installconductor method.
    self.conductordatalist = []
#    self.mgmaxlevels=0
#    self.mglevels=0

  def installconductor(self,conductor,
                            xmin=None,xmax=None,
                            ymin=None,ymax=None,
                            zmin=None,zmax=None,
                            dfill=top.largepos):
    # --- This only adds the conductor to the list. The data is only actually
    # --- installed when it is needed, during a call to getconductorobject.
    self.conductordatalist.append((conductor,xmin,xmax,ymin,ymax,zmin,zmax,dfill))

  def _installconductor(self,conductorobject,installedlist,conductordata,fselfb):
    # --- This does that actual installation of the conductor into the
    # --- conductor object

    # --- Extract the data from conductordata (the arguments to installconductor)
    conductor,xmin,xmax,ymin,ymax,zmin,zmax,dfill = conductordata

    if conductor in installedlist: return
    installedlist.append(conductor)

    if fselfb == 'p':
      zscale = 1.
      nx,ny,nzlocal,nz = self.nxp,self.nyp,self.nzp,self.nz
      xmmin,xmmax = self.xmminp,self.xmmaxp
      ymmin,ymmax = self.ymminp,self.ymmaxp
      zmmin,zmmax = self.zmminp,self.zmmaxp
      mgmaxlevels = 1
    else:
      # --- Get relativistic longitudinal scaling factor
      # --- This is quite ready yet.
      beta = fselfb/clight
      zscale = 1./sqrt((1.-beta)*(1.+beta))
      nx,ny,nzlocal,nz = self.nx,self.ny,self.nzlocal,self.nz
      xmmin,xmmax = self.xmmin,self.xmmax
      ymmin,ymmax = self.ymmin,self.ymmax
      zmmin,zmmax = self.zmmin,self.zmmax
      mgmaxlevels = None

    mgmaxlevels=1
    installconductors(conductor,xmin,xmax,ymin,ymax,zmin,zmax,dfill,
                      self.getzgrid(),
                      nx,ny,nzlocal,nz,
                      xmmin,xmmax,ymmin,ymmax,zmmin,zmmax,
                      zscale,self.l2symtry,self.l4symtry,
                      installrz=0,
                      solvergeom=self.solvergeom,conductors=conductorobject,
                      mgmaxlevels=mgmaxlevels,
                      my_index=self.my_index,nslaves=self.nslaves,
                      izfsslave=self.izfsslave,nzfsslave=self.nzfsslave)

    self.fields.nconds = self.getconductorobject().interior.n
    self.fields.nxcond = self.fields.nx
    self.fields.nycond = self.fields.ny
    self.fields.nzcond = self.fields.nz
    self.fields.gchange()
    self.fields.incond=False
    if self.fields.nconds>0:
      set_incond(self.fields,self.fields.nconds,int(self.getconductorobject().interior.indx))
    if self.block.xlbnd==openbc:self.fields.incond[:3,:,:]=False
    if self.block.xrbnd==openbc:self.fields.incond[-3:,:,:]=False
    if self.block.ylbnd==openbc:self.fields.incond[:,:3,:]=False
    if self.block.yrbnd==openbc:self.fields.incond[:,-3:,:]=False
    if self.block.zlbnd==openbc:self.fields.incond[:,:,:3]=False
    if self.block.zrbnd==openbc:self.fields.incond[:,:,-3:]=False
    
  def hasconductors(self):
    return len(self.conductordatalist) > 0

  def clearconductors(self):
    "Clear out the conductor data"
    for fselfb in top.fselfb:
      if fselfb in self.conductorobjects:
        conductorobject = self.conductorobjects[fselfb]
        conductorobject.interior.n = 0
        conductorobject.evensubgrid.n = 0
        conductorobject.oddsubgrid.n = 0
        self.installedconductorlists[fselfb] = []

  def getconductorobject(self,fselfb=0.):
    "Checks for and installs any conductors not yet installed before returning the object"
    # --- This is the routine that does the creation of the ConductorType
    # --- objects if needed and ensures that all conductors are installed
    # --- into it.

    # --- This method is needed during a restore from a pickle, since this
    # --- object may be restored before the conductors. This delays the
    # --- installation of the conductors until they are really needed.

    # --- There is a special case, fselfb='p', which refers to the conductor
    # --- object that has the data generated relative to the particle domain,
    # --- which can be different from the field domain, especially in parallel.
    if fselfb == 'p':
      # --- In serial, just use a reference to the conductor object for the
      # --- first iselfb group.
      if not lparallel and 'p' not in self.conductorobjects:
        self.conductorobjects['p'] = self.conductorobjects[top.fselfb[0]]
        self.installedconductorlists['p'] = self.installedconductorlists[top.fselfb[0]]
      # --- In parallel, a whole new instance is created (using the
      # --- setdefaults below).
      # --- Check to make sure that the grid the conductor uses is consistent
      # --- with the particle grid. This is needed so that the conductor
      # --- data is updated when particle load balancing is done. If the
      # --- data is not consistent, delete the conductor object so that
      # --- everything is reinstalled.
      try:
        conductorobject = self.conductorobjects['p']
        if (conductorobject.leveliz[0] != self.izpslave[self.my_index] or
            conductorobject.levelnz[0] != self.nzpslave[self.my_index]):
          del self.conductorobjects['p']
          del self.installedconductorlists['p']
      except KeyError:
        # --- 'p' object has not yet been created anyway, so do nothing.
        pass

    conductorobject = self.conductorobjects.setdefault(fselfb,ConductorType())
    installedconductorlist = self.installedconductorlists.setdefault(fselfb,[])

    # --- Now, make sure that the conductors are installed into the object.
    # --- This may be somewhat inefficient, since it loops over all of the
    # --- conductors everytime. This makes the code more robust, though, since
    # --- it ensures that all conductors will be properly installed into
    # --- the conductor object.
    for conductordata in self.conductordatalist:
      self._installconductor(conductorobject,installedconductorlist,
                             conductordata,fselfb)
 
    # --- Return the desired conductor object
    return conductorobject

  def setconductorvoltage(self,voltage,condid=0,discrete=false,
                          setvinject=false):
    'calls setconductorvoltage'
    # --- Loop over all of the selfb groups to that all conductor objects
    # --- are handled.
    for iselfb in range(top.nsselfb):
      conductorobject = self.getconductorobject(top.fselfb[iselfb])
      setconductorvoltage(voltage,condid,discrete,setvinject,
                          conductors=conductorobject)

  def optimizeconvergence(self,resetpasses=1):
    pass

  def move_window_fields(self):
    if (w3d.solvergeom not in [w3d.XZgeom]) or \
       (abs(top.zgrid-self.zgridprv)<0.5*self.dz):return 
#    if not self.l_moving_window or ((top.it%self.ndelta_t)!=0): return
#    if top.time < self.tmin_moving_main_window: return
    move_window_field(self.fields)
    self.zgridprv=top.zgrid
    if not self.l_onegrid:
      self.fpatchfine.xminscatter+=w3d.dz
      self.fpatchfine.xmaxscatter+=w3d.dz
      self.fpatchfine.xmingather+=w3d.dz
      self.fpatchfine.xmaxgather+=w3d.dz

  def add_laser(self,field):
    if self.laser_profile is None: return

    if self.laser_amplitude_func is not None:
      self.laser_amplitude = self.laser_amplitude_func(top.time)
    elif self.laser_amplitude_table is not None:
      if top.time < self.laser_amplitude_table[0,1]:
        self.laser_amplitude = self.laser_amplitude_table[0,0]
      elif top.time >= self.laser_amplitude_table[-1,1]:
        self.laser_amplitude = self.laser_amplitude_table[-1,0]
      else:
        i = self.laser_amplitude_table_i
        while top.time > self.laser_amplitude_table[i+1,1]:
          i = i + 1
        self.laser_amplitude_table_i = i
        ww = ((top.time - self.laser_amplitude_table[i,1])/
           (self.laser_amplitude_table[i+1,1]-self.laser_amplitude_table[i,1]))
        self.laser_amplitude = ((1.-ww)*self.laser_amplitude_table[i,0] +
                                    ww *self.laser_amplitude_table[i+1,0])

    if self.laser_profile_func is not None:
      self.laser_profile = self.laser_profile_func(top.time)
      assert len(self.laser_profile) == field.ny+3,"The specified profile must be of length ny+4"

#    if (self.l_elaser_out_plane):
#      xx = (arange(self.nx+4) - 0.5)*self.fields.dx+self.fields.xmin
#    else:
    xx = (arange(field.ny+3) - 0.5)*field.dy+field.ymin

    if self.laser_frequency is not None:
      if self.laser_focus is not None:
        z0 = self.laser_focus+self.laser_focus_velocity*top.time
        if self.laser_focus>0.:
          phase = (-(sqrt(xx**2+z0**2)-z0)/clight-top.time)*self.laser_frequency
        else:
          phase = ((sqrt(xx**2+z0**2)-z0)/clight-top.time)*self.laser_frequency
      else:
        phase = (xx*sin(self.laser_angle)/clight-top.time)*self.laser_frequency
    else:
      phase = 0.

    self.block.core.yf.Ey[:,:,1] = self.laser_amplitude*self.laser_profile[0]*cos(phase[0])
    return
    if (self.l_elaser_out_plane):
      field.Ez_in = self.laser_amplitude*field.laser_profile[:-1]*cos(phase)
    else:
      field.Bz_in = self.laser_amplitude*field.laser_profile[:-1]*cos(phase)

  def solve2ndhalf(self):
    if self.l_verbose:print 'solve 2nd half'
    if top.dt<>self.dtinit:raise('Time step has been changed since initialization of EM3D.')
    # --- Set nxl and nyl if using large stencil
#    if(not self.l_onegrid):
#      project_j(self.fields,self.fpatchcoarse,self.fpatchfine)
#    for field in fields:
#      if field.l_uselargestencil and field.nxl<>field.nx:
#        field.nxl=field.nx
#        field.nyl=field.ny
#        field.gchange()
#      self.add_laser(field)
#      grimax(field)
    if top.efetch[0]<>4:node2yee3d(self.block.core.yf)
    dt = top.dt/self.nfield_subcycle
    push_em3d_bf(self.block,dt,2,self.l_pushf,self.l_pushpot)
    if self.l_verbose:print 'solve 2nd half done'
#    yee2node3d(self.block.core.yf)
 #   self.move_window_fields()
 #   for field in fields:
 #     griuni(field)
 #   if not self.l_onegrid:set_substitute_fields(self.fields,self.fpatchcoarse,self.fpatchfine)

  def solve(self):
    if self.l_verbose:print 'solve 1st half'
    if top.dt<>self.dtinit:raise('Time step has been changed since initialization of EM3D.')
    # --- Set nxl and nyl if using large stencil
#    if(not self.l_onegrid):
#      project_j(self.fields,self.fpatchcoarse,self.fpatchfine)
#    for field in fields:
#      if field.l_uselargestencil and field.nxl<>field.nx:
#        field.nxl=field.nx
#        field.nyl=field.ny
#        field.gchange()
#      self.add_laser(field)
#      grimax(field)
#    node2yee3d(self.block.core.yf)
    dt = top.dt/self.nfield_subcycle
    if self.l_pushf and self.nfield_subcycle>1:
      w = 1./self.nfield_subcycle
      self.fields.Rho = (1.-w)*self.fields.Rhoold + w*self.fields.Rhoarray[...,0] 
    push_em3d_eef(self.block,dt,0,self.l_pushf,self.l_pushpot)
    self.add_laser(self.fields)
    for i in range(self.nfield_subcycle-1):
      push_em3d_bf(self.block,dt,0,self.l_pushf,self.l_pushpot)
      if self.l_pushf and self.nfield_subcycle>1:
        w = float(i+2)/self.nfield_subcycle
        self.fields.Rho = (1.-w)*self.fields.Rhoold + w*self.fields.Rhoarray[...,0] 
      push_em3d_eef(self.block,dt,0,self.l_pushf,self.l_pushpot)
    push_em3d_bf(self.block,dt,1,self.l_pushf,self.l_pushpot)
#    push_em3d_eef(self.block,dt,0,self.l_pushf)
#    push_em3d_bf(self.block,dt,1,self.l_pushf)
#    for i in range(self.nfield_subcycle-1):
#      push_em3d_bf(self.block,dt,2,self.l_pushf)
#      push_em3d_eef(self.block,dt,0,self.l_pushf)
#      push_em3d_bf(self.block,dt,1,self.l_pushf)
    if not all(top.efetch==top.efetch[0]):raise('Error:top.efetch must have same value for every species when using EM solver.')
    if top.efetch[0]<>4:yee2node3d(self.block.core.yf)
 #   self.move_window_fields()
 #   for field in fields:
 #     griuni(field)
 #   if not self.l_onegrid:set_substitute_fields(self.fields,self.fpatchcoarse,self.fpatchfine)
    if self.l_verbose:print 'solve 1st half done'

  ##########################################################################
  # Define the basic plot commands
  def genericfp(self,data,title,l_transpose=true,direction=2,slice=None,**kw):
    f=self.block.core.yf
    if direction==0:
      if slice is None:
        if w3d.l4symtry:
          slice=0
        else:
          slice=w3d.nx/2
      if l_transpose:
        settitles(title,'Z','Y','t = %gs'%(top.time))
        kw.setdefault('xmin',w3d.zmmin)
        kw.setdefault('xmax',w3d.zmmax)
        kw.setdefault('ymin',w3d.ymmin)
        kw.setdefault('ymax',w3d.ymmax)
      else:
        settitles(title,'Y','Z','t = %gs'%(top.time))
        kw.setdefault('xmin',w3d.ymmin)
        kw.setdefault('xmax',w3d.ymmax)
        kw.setdefault('ymin',w3d.zmmin)
        kw.setdefault('ymax',w3d.zmmax)
      if l_transpose:
        data=transpose(data[slice,:,:])
      else:
        data=data[slice,:,:]
    if direction==1:
      if slice is None:
        if w3d.l4symtry:
          slice=0
        else:
          slice=w3d.ny/2
      if l_transpose:
        settitles(title,'Z','X','t = %gs'%(top.time))
        kw.setdefault('ymin',w3d.xmmin)
        kw.setdefault('ymax',w3d.xmmax)
        kw.setdefault('xmin',w3d.zmmin)
        kw.setdefault('xmax',w3d.zmmax)
      else:
        settitles(title,'X','Z','t = %gs'%(top.time))
        kw.setdefault('xmin',w3d.xmmin)
        kw.setdefault('xmax',w3d.xmmax)
        kw.setdefault('ymin',w3d.zmmin)
        kw.setdefault('ymax',w3d.zmmax)
      if l_transpose:
        data=transpose(data[:,slice,:])
      else:
        data=data[:,slice,:]
    if direction==2:
      if slice is None:slice=w3d.nz/2
      if l_transpose:
        settitles(title,'X','Y','t = %gs'%(top.time))
        kw.setdefault('xmin',w3d.ymmin)
        kw.setdefault('xmax',w3d.ymmax)
        kw.setdefault('ymin',w3d.xmmin)
        kw.setdefault('ymax',w3d.xmmax)
      else:
        settitles(title,'Y','X','t = %gs'%(top.time))
        kw.setdefault('xmin',w3d.xmmin)
        kw.setdefault('xmax',w3d.xmmax)
        kw.setdefault('ymin',w3d.ymmin)
        kw.setdefault('ymax',w3d.ymmax)
      if l_transpose:
        data=transpose(data[:,:,slice])
      else:
        data=data[:,:,slice]
    ppgeneric(grid=data,**kw)
      
  def gatherarray(self,g):
    f=self.fields
    if me<npes-1:
      return transpose(gatherarray(transpose(g[f.nxguard:-f.nxguard,f.nyguard:-f.nyguard,f.nzguard:-f.nzguard-1],[2,1,0])),[2,1,0])
    else:
      return transpose(gatherarray(transpose(g[f.nxguard:-f.nxguard,f.nyguard:-f.nyguard,f.nzguard:-f.nzguard],[2,1,0])),[2,1,0])
      
  def fpex(self,**kw):
      self.genericfp(self.gatherarray(self.fields.Ex),'E_x',**kw)

  def fpey(self,**kw):
      self.genericfp(self.gatherarray(self.fields.Ey),'E_y',**kw)

  def fpez(self,**kw):
      self.genericfp(self.gatherarray(self.fields.Ez),'E_z',**kw)

  def fpbx(self,**kw):
      self.genericfp(self.gatherarray(self.fields.Bx),'B_x',**kw)

  def fpby(self,**kw):
      self.genericfp(self.gatherarray(self.fields.By),'B_y',**kw)

  def fpbz(self,**kw):
      self.genericfp(self.gatherarray(self.fields.Bz),'B_z',**kw)

  def fpjx(self,**kw):
      self.genericfp(self.gatherarray(self.fields.J[:,:,:,0]),'J_x',**kw)

  def fpjy(self,**kw):
      self.genericfp(self.gatherarray(self.fields.J[:,:,:,1]),'J_y',**kw)

  def fpjz(self,**kw):
      self.genericfp(self.gatherarray(self.fields.J[:,:,:,2]),'J_z',**kw)

  def fprho(self,**kw):
      self.genericfp(self.gatherarray(self.fields.Rho),'Rho',**kw)

  def fpf(self,**kw):
      self.genericfp(self.gatherarray(self.fields.F),'F',**kw)

  def sezax(self):
    pass

  def sphiax(self):
    pass

  def rhodia(self):
    pass

  def gtlchg(self):
    pass

  def srhoax(self):
    pass

  def getese(self):
    pass

  def getjx(self):
      return self.gatherarray(self.fields.J[:,:,:,0])

  def getjy(self):
      return self.gatherarray(self.fields.J[:,:,:,1])

  def getjz(self):
      return self.gatherarray(self.fields.J[:,:,:,2])

  def getex(self):
      return self.gatherarray(self.fields.Ex)

  def getey(self):
      return self.gatherarray(self.fields.Ey)
        
  def getez(self):
      return self.gatherarray(self.fields.Ez)
        
  def getbx(self):
      return self.gatherarray(self.fields.Bx)

  def getby(self):
      return self.gatherarray(self.fields.By)
        
  def getbz(self):
      return self.gatherarray(self.fields.Bz)
        
  def getrho(self):
      return self.gatherarray(self.fields.Rho)

  def getf(self):
      return self.gatherarray(self.fields.F)

  ##########################################################################
  # Define the basic plot commands
  def genericpfem3d(self,kw,pffunc):
    fselfb = kw.get('fselfb',top.fselfb[0])
    if 'fselfb' in kw: del kw['fselfb']
    kw['conductors'] = self.getconductorobject(fselfb)
    kw['solver'] = self
    pffunc(**kw)
  def pfxy(self,**kw): self.genericpfem3d(kw,pfxy)
  def pfzx(self,**kw): self.genericpfem3d(kw,pfzx)
  def pfzy(self,**kw): self.genericpfem3d(kw,pfzy)
  def pfxyg(self,**kw): self.genericpfem3d(kw,pfxyg)
  def pfzxg(self,**kw): self.genericpfem3d(kw,pfzxg)
  def pfzyg(self,**kw): self.genericpfem3d(kw,pfzyg)

def allocatesf(f):
    f.syf = EM3D_SPLITYEEFIELDtype()
    f.fieldtype = f.syf.fieldtype
    f.proc=me

def allocatef(f):
    f.yf = EM3D_YEEFIELDtype()
    f.fieldtype = f.yf.fieldtype
    f.proc=me

def pyinit_3dem_blockold(nx, ny, nz, nbndx, nbndy, nbndz, nxguard, nyguard, nzguard, dt, dx, dy, dz, clight, mu0, xmin, ymin, zmin, xlb, ylb, zlb, xrb, yrb, zrb,deposit_energy_density):
  
  l_1d_decomposition=true
  
  b = EM3D_BLOCKtype()
  b.core = EM3D_FIELDtype()
  b.core.yf = EM3D_YEEFIELDtype()
  b.core.fieldtype = b.core.yf.fieldtype
  b.core.proc = me

  if npes>1:
    if l_1d_decomposition:
      if me==0:
        zrb = em3d.otherproc
        if zlb==periodic:
          zlb = em3d.otherproc
      elif me==npes-1:
        zlb = em3d.otherproc
        if zrb==periodic:
          zrb = em3d.otherproc
      else:
        zlb=zrb=em3d.otherproc

  b.nx = nx
  b.ny = ny
  b.nz = nz
  b.nbndx = nbndx
  b.nbndy = nbndy
  b.nbndz = nbndz
  b.xmin = xmin
  b.ymin = ymin
  b.zmin = zmin
  b.dx = dx
  b.dy = dy
  b.dz = dz
  b.xmax = xmin+dx*nx
  b.ymax = ymin+dy*ny
  b.zmax = zmin+dz*nz
  b.dxi = 1./dx
  b.dyi = 1./dy
  b.dzi = 1./dz
  b.xlbnd = xlb
  b.xrbnd = xrb
  b.ylbnd = ylb
  b.yrbnd = yrb
  b.zlbnd = zlb
  b.zrbnd = zrb

  f=b.core.yf
  f.nx = nx
  f.ny = ny
  f.nz = nz
  f.nxguard = nxguard
  f.nyguard = nyguard
  f.nzguard = nzguard
  # set min/max of cells positions with FORTRAN indexing
  f.ixmin = 0
  f.iymin = 0
  f.izmin = 0
  f.ixmax = f.nx
  f.iymax = f.ny
  f.izmax = f.nz
  f.ixming = -f.nxguard
  f.iyming = -f.nyguard
  f.izming = -f.nzguard
  f.ixmaxg = f.ixmax+f.nxguard
  f.iymaxg = f.iymax+f.nyguard
  f.izmaxg = f.izmax+f.nzguard
  # set min/max of cells positions with Python indexing
  f.jxmin = f.ixmin-f.ixming
  f.jymin = f.iymin-f.iyming
  f.jzmin = f.izmin-f.izming
  f.jxmax = f.ixmax-f.ixming
  f.jymax = f.iymax-f.iyming
  f.jzmax = f.izmax-f.izming
  f.jxming = 0
  f.jyming = 0
  f.jzming = 0
  f.jxmaxg = f.ixmaxg-f.ixming
  f.jymaxg = f.iymaxg-f.iyming
  f.jzmaxg = f.izmaxg-f.izming
  f.xmin = xmin
  f.ymin = ymin
  f.zmin = zmin
  f.dx = dx
  f.dy = dy
  f.dz = dz
  f.xmax = xmin+dx*nx
  f.ymax = ymin+dy*ny
  f.zmax = zmin+dz*nz
  f.dxi = 1./dx
  f.dyi = 1./dy
  f.dzi = 1./dz
  f.clight = clight
  f.mu0    = mu0

  if deposit_energy_density:
    f.nxmp=f.nx
    f.nymp=f.ny
    f.nzmp=f.nz

  f.gchange()
  
  nnx=em3d.nn
  nny=em3d.nn
  nnz=em3d.nn
  smaxx=em3d.s_max_init
  smaxy=em3d.s_max_init
  smaxz=em3d.s_max_init
  sdeltax=em3d.s_delta
  sdeltay=em3d.s_delta
  sdeltaz=em3d.s_delta
  
# *** sides
# x
  b.sidexl = EM3D_FIELDtype()
  if xlb==openbc:
    allocatesf(b.sidexl)
    init_splitfield(b.sidexl.syf,nbndx,ny,nz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight,-1, 0, 0, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if xlb==periodic:
    b.sidexl=b.core
  b.sidexr = EM3D_FIELDtype()
  if xrb==openbc:
    allocatesf(b.sidexr)
    init_splitfield(b.sidexr.syf,nbndx,ny,nz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 1, 0, 0, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if xrb==periodic:
    b.sidexr=b.core
# y
  b.sideyl = EM3D_FIELDtype()
  if ylb==openbc:
    allocatesf(b.sideyl)
    init_splitfield(b.sideyl.syf,nx,nbndy,nz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 0,-1, 0, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if ylb==periodic:
    b.sideyl=b.core
  b.sideyr = EM3D_FIELDtype()
  if yrb==openbc:
    allocatesf(b.sideyr)
    init_splitfield(b.sideyr.syf,nx,nbndy,nz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 0, 1, 0, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if yrb==periodic:
    b.sideyr=b.core
# z
  b.sidezl = EM3D_FIELDtype()
  if zlb==openbc:
    allocatesf(b.sidezl)
    init_splitfield(b.sidezl.syf,nx,ny,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 0, 0,-1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zlb==periodic:
    b.sidezl=b.core
  if zlb==em3d.otherproc:
    allocatef(b.sidezl)
    if l_1d_decomposition:
      b.sidezl.proc=(me-1)%npes      
  b.sidezr = EM3D_FIELDtype()
  if zrb==openbc:
    allocatesf(b.sidezr)
    init_splitfield(b.sidezr.syf,nx,ny,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 0, 0, 1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zrb==periodic:
    b.sidezr=b.core
  if zrb==em3d.otherproc:
    allocatef(b.sidezr)
    if l_1d_decomposition:
      b.sidezr.proc=(me+1)%npes      
    
# *** edges
# xy
  b.edgexlyl = EM3D_FIELDtype()
  if xlb==openbc and ylb==openbc:
    allocatesf(b.edgexlyl)
    init_splitfield(b.edgexlyl.syf,nbndx,nbndy,nz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight,-1,-1, 0, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
# partial periodic bc to be completed 
#  if xlb==openbc and ylb==periodic:
#    b.edgexlyl=b.sidexl
#  if ylb==openbc and xlb==periodic:
#    b.edgexlyl=b.sideyl
  b.edgexryl = EM3D_FIELDtype()
  if xrb==openbc and ylb==openbc:
    allocatesf(b.edgexryl)
    init_splitfield(b.edgexryl.syf,nbndx,nbndy,nz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 1,-1, 0, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  b.edgexlyr = EM3D_FIELDtype()
  if xlb==openbc and yrb==openbc:
    allocatesf(b.edgexlyr)
    init_splitfield(b.edgexlyr.syf,nbndx,nbndy,nz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight,-1, 1, 0, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  b.edgexryr = EM3D_FIELDtype()
  if xrb==openbc and yrb==openbc:
    allocatesf(b.edgexryr)
    init_splitfield(b.edgexryr.syf,nbndx,nbndy,nz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 1, 1, 0, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
# xz
  b.edgexlzl = EM3D_FIELDtype()
  if xlb==openbc and zlb==openbc:
    allocatesf(b.edgexlzl)
    init_splitfield(b.edgexlzl.syf,nbndx,ny,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight,-1, 0,-1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zlb==em3d.otherproc:
    allocatesf(b.edgexlzl)
    if l_1d_decomposition:
      b.edgexlzl.proc=(me-1)%npes      
  b.edgexrzl = EM3D_FIELDtype()
  if xrb==openbc and zlb==openbc:
    allocatesf(b.edgexrzl)
    init_splitfield(b.edgexrzl.syf,nbndx,ny,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 1, 0,-1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zlb==em3d.otherproc:
    allocatesf(b.edgexrzl)
    if l_1d_decomposition:
      b.edgexrzl.proc=(me-1)%npes      
  b.edgexlzr = EM3D_FIELDtype()
  if xlb==openbc and zrb==openbc:
    allocatesf(b.edgexlzr)
    init_splitfield(b.edgexlzr.syf,nbndx,ny,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight,-1, 0, 1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zrb==em3d.otherproc:
    allocatesf(b.edgexlzr)
    if l_1d_decomposition:
      b.edgexlzr.proc=(me+1)%npes      
  b.edgexrzr = EM3D_FIELDtype()
  if xrb==openbc and zrb==openbc:
    allocatesf(b.edgexrzr)
    init_splitfield(b.edgexrzr.syf,nbndx,ny,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 1, 0, 1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zrb==em3d.otherproc:
    allocatesf(b.edgexrzr)
    if l_1d_decomposition:
      b.edgexrzr.proc=(me+1)%npes      
# yz
  b.edgeylzl = EM3D_FIELDtype()
  if ylb==openbc and zlb==openbc:
    allocatesf(b.edgeylzl)
    init_splitfield(b.edgeylzl.syf,nx,nbndy,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 0,-1,-1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zlb==em3d.otherproc:
    allocatesf(b.edgeylzl)
    if l_1d_decomposition:
      b.edgeylzl.proc=(me-1)%npes      
  b.edgeyrzl = EM3D_FIELDtype()
  if yrb==openbc and zlb==openbc:
    allocatesf(b.edgeyrzl)
    init_splitfield(b.edgeyrzl.syf,nx,nbndy,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 0, 1,-1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zlb==em3d.otherproc:
    allocatesf(b.edgeyrzl)
    if l_1d_decomposition:
      b.edgeyrzl.proc=(me-1)%npes      
  b.edgeylzr = EM3D_FIELDtype()
  if ylb==openbc and zrb==openbc:
    allocatesf(b.edgeylzr)
    init_splitfield(b.edgeylzr.syf,nx,nbndy,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 0,-1, 1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zrb==em3d.otherproc:
    allocatesf(b.edgeylzr)
    if l_1d_decomposition:
      b.edgeylzr.proc=(me+1)%npes      
  b.edgeyrzr = EM3D_FIELDtype()
  if yrb==openbc and zrb==openbc:
    allocatesf(b.edgeyrzr)
    init_splitfield(b.edgeyrzr.syf,nx,nbndy,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 0, 1, 1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zrb==em3d.otherproc:
    allocatesf(b.edgeyrzr)
    if l_1d_decomposition:
      b.edgeyrzr.proc=(me+1)%npes      

# *** corners
  b.cornerxlylzl = EM3D_FIELDtype()
  if xlb==openbc and ylb==openbc and zlb==openbc:
    allocatesf(b.cornerxlylzl)
    init_splitfield(b.cornerxlylzl.syf,nbndx,nbndy,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight,-1,-1,-1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zlb==em3d.otherproc:
    allocatesf(b.cornerxlylzl)
    if l_1d_decomposition:
      b.cornerxlylzl.proc=(me-1)%npes      
  b.cornerxrylzl = EM3D_FIELDtype()
  if xrb==openbc and ylb==openbc and zlb==openbc:
    allocatesf(b.cornerxrylzl)
    init_splitfield(b.cornerxrylzl.syf,nbndx,nbndy,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 1,-1,-1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zlb==em3d.otherproc:
    allocatesf(b.cornerxrylzl)
    if l_1d_decomposition:
      b.cornerxrylzl.proc=(me-1)%npes      
  b.cornerxlyrzl = EM3D_FIELDtype()
  if xlb==openbc and yrb==openbc and zlb==openbc:
    allocatesf(b.cornerxlyrzl)
    init_splitfield(b.cornerxlyrzl.syf,nbndx,nbndy,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight,-1, 1,-1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zlb==em3d.otherproc:
    allocatesf(b.cornerxlyrzl)
    if l_1d_decomposition:
      b.cornerxlyrzl.proc=(me-1)%npes      
  b.cornerxryrzl = EM3D_FIELDtype()
  if xrb==openbc and yrb==openbc and zlb==openbc:
    allocatesf(b.cornerxryrzl)
    init_splitfield(b.cornerxryrzl.syf,nbndx,nbndy,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 1, 1,-1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zlb==em3d.otherproc:
    allocatesf(b.cornerxryrzl)
    if l_1d_decomposition:
      b.cornerxryrzl.proc=(me-1)%npes      
  b.cornerxlylzr = EM3D_FIELDtype()
  if xlb==openbc and ylb==openbc and zrb==openbc:
    allocatesf(b.cornerxlylzr)
    init_splitfield(b.cornerxlylzr.syf,nbndx,nbndy,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight,-1,-1, 1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zrb==em3d.otherproc:
    allocatesf(b.cornerxlylzr)
    if l_1d_decomposition:
      b.cornerxlylzr.proc=(me+1)%npes      
  b.cornerxrylzr = EM3D_FIELDtype()
  if xrb==openbc and ylb==openbc and zrb==openbc:
    allocatesf(b.cornerxrylzr)
    init_splitfield(b.cornerxrylzr.syf,nbndx,nbndy,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 1,-1, 1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zrb==em3d.otherproc:
    allocatesf(b.cornerxrylzr)
    if l_1d_decomposition:
      b.cornerxrylzr.proc=(me+1)%npes      
  b.cornerxlyrzr = EM3D_FIELDtype()
  if xlb==openbc and yrb==openbc and zrb==openbc:
    allocatesf(b.cornerxlyrzr)
    init_splitfield(b.cornerxlyrzr.syf,nbndx,nbndy,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight,-1, 1, 1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zrb==em3d.otherproc:
    allocatesf(b.cornerxlyrzr)
    if l_1d_decomposition:
      b.cornerxlyrzr.proc=(me+1)%npes      
  b.cornerxryrzr = EM3D_FIELDtype()
  if xrb==openbc and yrb==openbc and zrb==openbc:
    allocatesf(b.cornerxryrzr)
    init_splitfield(b.cornerxryrzr.syf,nbndx,nbndy,nbndz,nxguard,nyguard,nzguard, dt, dx, dy, dz, clight, 1, 1, 1, nnx, smaxx, sdeltax, nny, smaxy, sdeltay, nnz, smaxz, sdeltaz)
  if zrb==em3d.otherproc:
    allocatesf(b.cornerxryrzr)
    if l_1d_decomposition:
      b.cornerxryrzr.proc=(me+1)%npes      

  return b

##############################################################################
# --- This can only be done after the class is defined.
#try:
#  psyco.bind(EM3D)
#except NameError:
#  pass

