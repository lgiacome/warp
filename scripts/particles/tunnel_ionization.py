from warp import *
from warp.particles.ionization import *

warp_elements=elements=[]
for k in periodic_table:
  exec('elements.append('+k+')')
del k

for elemt in elements:
   elemt.ionization_levels=zeros(elemt.Z)
   Ionization_Levels=elemt.ionization_levels
   Z = elemt.Z
   if Z==1:#----- Hydrogen
      Ionization_Levels[0] = 13.6;
   if Z==2:#---- Helium
      Ionization_Levels[0] = 24.58;
      Ionization_Levels[1] = 54.4;
   if Z==3:#----Lithium
      Ionization_Levels[0] = 5.392;
      Ionization_Levels[1] = 75.638;
      Ionization_Levels[2] = 122.451;
   if Z==5:#---Boron
      Ionization_Levels[0] = 8.297;
      Ionization_Levels[1] = 25.154;
      Ionization_Levels[2] = 37.929;
      Ionization_Levels[3] = 259.367;
      Ionization_Levels[4] = 340.216;
   if Z==6:#---Carbon
      Ionization_Levels[0] = 11.261;
      Ionization_Levels[1] = 24.383;
      Ionization_Levels[2] = 47.887;
      Ionization_Levels[3] = 64.492;
      Ionization_Levels[4] = 392.081;
      Ionization_Levels[5] = 489.979;
   if Z==7:#--N
      Ionization_Levels[0] = 14.534;
      Ionization_Levels[1] = 29.601;
      Ionization_Levels[2] = 47.448;
      Ionization_Levels[3] = 77.472;
      Ionization_Levels[4] = 97.888;
      Ionization_Levels[5] = 552.057;
      Ionization_Levels[6] = 667.029;
   if Z==10:#--Ne
      Ionization_Levels[0] = 21.564;
      Ionization_Levels[1] = 40.962;
      Ionization_Levels[2] = 63.45;
      Ionization_Levels[3] = 97.11;
      Ionization_Levels[4] = 126.21;
      Ionization_Levels[5] = 157.93;
      Ionization_Levels[6] = 207.27;
      Ionization_Levels[7] = 239.09;
      Ionization_Levels[8] = 1195.79;
      Ionization_Levels[9] = 1362.164;
   if Z==13:#--Aluminium
      Ionization_Levels[0] = 5.968;
      Ionization_Levels[1] = 18.796;
      Ionization_Levels[2] = 28.399;
      Ionization_Levels[3] = 119.78;
      Ionization_Levels[4] = 153.561;
      Ionization_Levels[5] = 190.156;
      Ionization_Levels[6] = 241.34;
      Ionization_Levels[7] = 284.163;
      Ionization_Levels[8] = 329.564;
      Ionization_Levels[9] = 398.057;
      Ionization_Levels[10] = 441.232;
      Ionization_Levels[11] = 2082.379;
      Ionization_Levels[12] = 2300.16;
   if Z==14:#Silicon:
      Ionization_Levels[0] = 7.264E0;
      Ionization_Levels[1] = 1.695E1;
      Ionization_Levels[2] = 3.427E1;
      Ionization_Levels[3] = 4.665E1;
      Ionization_Levels[4] = 1.598E2;
      Ionization_Levels[5] = 2.105E2;
      Ionization_Levels[6] = 2.613E2;
      Ionization_Levels[7] = 3.120E2;
      Ionization_Levels[8] = 3.640E2;
      Ionization_Levels[9] = 4.151E2;
      Ionization_Levels[10] = 5.037E2;
      Ionization_Levels[11] = 5.522E2;
      Ionization_Levels[12] = 2.324E3;
      Ionization_Levels[13] = 2.569E3;
   if Z==18:#----Argon
      Ionization_Levels[0] = 15.759;
      Ionization_Levels[1] = 27.6;
      Ionization_Levels[2] = 40.7;
      Ionization_Levels[3] = 59.81;
      Ionization_Levels[4] = 75.02;
      Ionization_Levels[5] = 91.007;
      Ionization_Levels[6] = 124.319;
      Ionization_Levels[7] = 143.456;
      Ionization_Levels[8] = 422.44;
      Ionization_Levels[9] = 478.68;
      Ionization_Levels[10] = 538.95;
      Ionization_Levels[11] = 618.24;
      Ionization_Levels[12] = 686.09;
      Ionization_Levels[13] = 755.73;
      Ionization_Levels[14] = 854.75;
      Ionization_Levels[15] = 918;
      Ionization_Levels[16] = 4120.778;
      Ionization_Levels[17] = 4426.114;
   if Z==29:#Copper:
      Ionization_Levels[0] = 7.70;
      Ionization_Levels[1] = 20.234;
      Ionization_Levels[2] = 36.739;
      Ionization_Levels[3] = 57.213;
      Ionization_Levels[4] = 79.577;
      Ionization_Levels[5] = 102.313;
      Ionization_Levels[6] = 138.48;
      Ionization_Levels[7] = 165.354;
      Ionization_Levels[8] = 198.425;
      Ionization_Levels[9] = 231.492;
      Ionization_Levels[10] = 264.566;
      Ionization_Levels[11] = 367.913;
      Ionization_Levels[12] = 399.951;
      Ionization_Levels[13] = 434.055;
      Ionization_Levels[14] = 482.627;
      Ionization_Levels[15] = 518.798;
      Ionization_Levels[16] = 554.970;
      Ionization_Levels[17] = 631.446;
      Ionization_Levels[18] = 668.672;
      Ionization_Levels[19] = 1691.78;
      Ionization_Levels[20] = 1799.261;
      Ionization_Levels[21] = 1916.0;
      Ionization_Levels[22] = 2060.0;
      Ionization_Levels[23] = 2182.0;
      Ionization_Levels[24] = 2308.0;
      Ionization_Levels[25] = 2478.0;
      Ionization_Levels[26] = 2587.5;
      Ionization_Levels[27] = 11062.38;
      Ionization_Levels[28] = 11567.617;
   if Z==36:#Krypton
      Ionization_Levels[0] = 1.3999e1;
      Ionization_Levels[1] = 2.4359e1;
      Ionization_Levels[2] = 3.695e1;
      Ionization_Levels[3] = 5.25e1;
      Ionization_Levels[4] = 6.47e1;
      Ionization_Levels[5] = 7.85e1;
      Ionization_Levels[6] = 1.11e2;
      Ionization_Levels[7] = 1.258e2;
      Ionization_Levels[8] = 2.3085e2;
      Ionization_Levels[9] = 2.682e2;
      Ionization_Levels[10] = 3.08e2;
      Ionization_Levels[11] = 3.50e2;
      Ionization_Levels[12] = 3.91e2;
      Ionization_Levels[13] = 4.47e2;
      Ionization_Levels[14] = 4.92e2;
      Ionization_Levels[15] = 5.41e2;
      Ionization_Levels[16] = 5.92e2;
      Ionization_Levels[17] = 6.41e2;
      Ionization_Levels[18] = 7.86e2;
      Ionization_Levels[19] = 8.33e2;
      Ionization_Levels[20] = 8.84e2;
      Ionization_Levels[21] = 9.37e2;
      Ionization_Levels[22] = 9.98e2;
      Ionization_Levels[23] = 1.051e3;
      Ionization_Levels[24] = 1.151e3;
      Ionization_Levels[25] = 1.2053e3;
      Ionization_Levels[26] = 2.928e3;
      Ionization_Levels[27] = 3.07e3;
      Ionization_Levels[28] = 3.227e3;
      Ionization_Levels[29] = 3.381e3;
      Ionization_Levels[30] = 0.0;
      Ionization_Levels[31] = 0.0;
      Ionization_Levels[32] = 0.0;
      Ionization_Levels[33] = 0.0;
      Ionization_Levels[34] = 0.0;
      Ionization_Levels[35] = 0.0;
   if Z==54:#Xenon
      Ionization_Levels[0] = 1.197e1;
      Ionization_Levels[1] = 2.354e1;
      Ionization_Levels[2] = 3.511e1;
      Ionization_Levels[3] = 4.668e1;
      Ionization_Levels[4] = 5.969e1;
      Ionization_Levels[5] = 7.183e1;
      Ionization_Levels[6] = 9.805e1;
      Ionization_Levels[7] = 1.123e2;
      Ionization_Levels[8] = 1.708e2;
      Ionization_Levels[9] = 2.017e2;
      Ionization_Levels[10] = 2.326e2;
      Ionization_Levels[11] = 2.635e2;
      Ionization_Levels[12] = 2.944e2;
      Ionization_Levels[13] = 3.253e2;
      Ionization_Levels[14] = 3.583e2;
      Ionization_Levels[15] = 3.896e2;
      Ionization_Levels[16] = 4.209e2;
      Ionization_Levels[17] = 4.522e2;
      Ionization_Levels[18] = 5.725e2;
      Ionization_Levels[19] = 6.077e2;
      Ionization_Levels[20] = 6.429e2;
      Ionization_Levels[21] = 6.781e2;
      Ionization_Levels[22] = 7.260e2;
      Ionization_Levels[23] = 7.627e2;
      Ionization_Levels[24] = 8.527e2;
      Ionization_Levels[25] = 8.906e2;
      Ionization_Levels[26] = 1.394e3;
      Ionization_Levels[27] = 1.491e3;
      Ionization_Levels[28] = 1.587e3;
      Ionization_Levels[29] = 1.684e3;
      Ionization_Levels[30] = 1.781e3;
      Ionization_Levels[31] = 1.877e3;
      Ionization_Levels[32] = 1.987e3;
      Ionization_Levels[33] = 2.085e3;
      Ionization_Levels[34] = 2.183e3;
      Ionization_Levels[35] = 2.281e3;
      Ionization_Levels[36] = 2.548e3;
      Ionization_Levels[37] = 2.637e3;
      Ionization_Levels[38] = 2.726e3;
      Ionization_Levels[39] = 2.814e3;
      Ionization_Levels[40] = 3.001e3;
      Ionization_Levels[41] = 3.093e3;
      Ionization_Levels[42] = 3.296e3;
      Ionization_Levels[43] = 3.386e3;
      Ionization_Levels[44] = 7.224e3;
      Ionization_Levels[45] = 7.491e3;
      Ionization_Levels[46] = 7.758e3;
      Ionization_Levels[47] = 8.024e3;
      Ionization_Levels[48] = 8.617e3;
      Ionization_Levels[49] = 8.899e3;
      Ionization_Levels[50] = 9.330e3;
      Ionization_Levels[51] = 9.569e3;
      Ionization_Levels[52] = 3.925e4;
      Ionization_Levels[53] = 4.027e4;
   if Z==55:#----Cesium
      Ionization_Levels[0] = 3.894;


class TunnelIonization(Ionization):
  def __init__(self,**kw):
    Ionization.__init__(self,**kw)
    self.alpha=1./137
    self.hbar=1.054571726e-34
    self.rbohr=4.*pi*eps0*self.hbar**2/(emass*echarge**2)
    self.Ea=echarge/(self.rbohr**2*4.*pi*eps0)
    self.Wa=self.Ea/(emass*echarge/(4.*pi*eps0*self.hbar))
    
  def add_tunnel_ionization(self,incident_species,target_species=None,emitted_species=None,cross_section=None,**kw):
    """An electron will be detached from the incident species when it
interacts with the target electric field, resulting in a reduced charge state
particle and an electron.  If the cross section is not given, it will be
obtained from the tunnel ionization rate subroutine. 
The reduced charge state species or its particle
type must be listed in the emitted species so that the proper cross section
can be found. If specified, the emitted species will be emitted with the
velocity of the incident particle.
    """
    # --- Generate a temporary list of emitted species, making sure that it includes all of the
    # --- appropriate species so that the proper cross section can be found. This includes
    # --- the target species, the reduced charge species and an electron. The user must have
    # --- provided the reduced charge species.
    e_species = copy.copy(emitted_species)
    if not iterable(e_species): e_species = [e_species]
    l_need_electron = True
    for es in e_species:
        if es is Electron: l_need_electron = False
        if isinstance(es,Species) and es.type is Electron: l_need_electron = False
    if l_need_electron: e_species.append(Electron)

    cross_section = self.setupcross_section(incident_species,e_species,cross_section,target_species)

    self.add(incident_species,emitted_species,
             l_remove_incident=True,**kw)


  def add(self,incident_species,emitted_species,ndens=None,target_fluidvel=None,
          emitted_energy0=None,emitted_energy_sigma=None,
          incident_pgroup=top.pgroup,target_pgroup=top.pgroup,emitted_pgroup=top.pgroup,
          l_remove_incident=None,l_remove_target=None,emitted_tag=None):
    if incident_species not in self.inter:
        self.inter[incident_species]={}
        for key in ['target_species','emitted_species',
                    'remove_incident','remove_target',
                    'emitted_energy0','emitted_energy_sigma','emitted_tag',
                    'incident_pgroup','target_pgroup','emitted_pgroup']:
          self.inter[incident_species][key]=[]
    if not iterable(emitted_species): emitted_species=[emitted_species]

    # --- Only include Species instances in the emitted species. It may have contained
    # --- Particle types that were needed for setting up the cross section calculation.
    e_species = []
    for es in emitted_species:
      if isinstance(es,Species): e_species.append(es)

    self.inter[incident_species]['emitted_species'] +=[e_species]
    if l_remove_incident is None:
      if incident_species.type is e_species[0].type:
        self.inter[incident_species]['remove_incident']+=[1]
      else:
        self.inter[incident_species]['remove_incident']+=[0]
    else:
      self.inter[incident_species]['remove_incident']+=[l_remove_incident]
    if emitted_energy0 is None and not self.inter[incident_species]['remove_incident'][-1]:
      # --- If the incident species is not being removed, then the emitted
      # --- particles are drawn from a random distribution. If not specified,
      # --- the default energy of the emitted particles is zero.
      emitted_energy0 = 0.
      emitted_energy_sigma = 0.
    self.inter[incident_species]['emitted_energy0']   +=[emitted_energy0]
    self.inter[incident_species]['emitted_energy_sigma']   +=[emitted_energy_sigma]
    self.inter[incident_species]['emitted_tag']   +=[emitted_tag]
    if emitted_tag is not None and self.emitted_id is None:
      self.emitted_id = nextpid()
    self.inter[incident_species]['incident_pgroup']=incident_pgroup
    self.inter[incident_species]['target_pgroup']  =target_pgroup
    self.inter[incident_species]['emitted_pgroup'] =emitted_pgroup

    for e in e_species:
      js=e.jslist[0]
      if emitted_pgroup not in self.x:
        self.nps[emitted_pgroup]={}
        self.x[emitted_pgroup]={}
        self.y[emitted_pgroup]={}
        self.z[emitted_pgroup]={}
        self.ux[emitted_pgroup]={}
        self.uy[emitted_pgroup]={}
        self.uz[emitted_pgroup]={}
        self.gi[emitted_pgroup]={}
        self.w[emitted_pgroup]={}
        self.pidtag[emitted_pgroup]={}
        self.injdatapid[emitted_pgroup]={}
      if js not in self.x[emitted_pgroup]:
        self.nps[emitted_pgroup][js]=0
        self.x[emitted_pgroup][js]=fzeros(self.npmax,'d')
        self.y[emitted_pgroup][js]=fzeros(self.npmax,'d')
        self.z[emitted_pgroup][js]=fzeros(self.npmax,'d')
        self.ux[emitted_pgroup][js]=fzeros(self.npmax,'d')
        self.uy[emitted_pgroup][js]=fzeros(self.npmax,'d')
        self.uz[emitted_pgroup][js]=fzeros(self.npmax,'d')
        self.gi[emitted_pgroup][js]=fzeros(self.npmax,'d')
        if top.wpid > 0:
          self.w[emitted_pgroup][js]=fzeros(self.npmax,'d')
        if emitted_tag is not None:
          self.pidtag[emitted_pgroup][js]=fzeros(self.npmax,'d')
        if top.injdatapid > 0:
          self.injdatapid[emitted_pgroup][js]=fzeros(self.npmax,'d')

  def GetADKrateSI(self,E,charge,elemnt,dt,l_dc2ac=False):
      Z=elemnt.Z
      # ------------CONSTANTS--(cgs)----------------------------------#
      wAtomic = 4.134E16; # Atomic Unit. of freq.
      rBohr = 5.2918E-11; # Bohr radius.
      rElec=echarge**2/(4.*pi*eps0*emass*clight**2) # Electron radius.
      #-----------Ion parameters----------------------------------#
      IPotAE = (1/27.212)*elemnt.ionization_levels[charge];# Ionization Potential. in A.U.
      zAtom = charge + 1.; # charge of ion created after ionization; should be +1 after(?)
      nEff = zAtom/sqrt(2.*IPotAE);# effective principle q no.
      E_atomic_rel = self.alpha*self.alpha*self.alpha*self.alpha*emass*clight*clight/(echarge*rElec)
      Eeff = E/E_atomic_rel; #Elec. field Normalized.
      #-------ADK Ionization formula in At. Units---------------#
      C_nl = ( ((2.*e)/nEff)**nEff ) /sqrt(2*pi*nEff); #ADK Costant C_n*l
      F_lm = 1.; #ADK constant f(l,m)
      #-------------------------------------------------------#
      frac = ((2*IPotAE)**1.5)/Eeff;
      term1 = (2.*frac)**(2*nEff-1);
      term2 = exp(-0.66667*frac);
      Wadk = C_nl*C_nl*F_lm*IPotAE*term1*term2
      if l_dc2ac:
        dc2ac = sqrt(3.0/pi*Eeff*(nEff/zAtom)*(nEff/zAtom)*(nEff/zAtom));
        Wadk*=dc2ac;
      return Wadk*self.Wa # --- rate in SI

  def GetADKprobSI(self,E,charge,elemnt,dt,l_dc2ac=False):
      ADKrate = self.GetADKrateSI(E,charge,elemnt,dt,l_dc2ac)
      return dt*ADKrate
#      return 1.-exp(-dt*ADKrate);

  def generate(self,dt=None):
    if dt is None:dt=top.dt/top.boost_gamma
    if self.l_timing:t1 = time.clock()
    for incident_species in self.inter:
      npinc = 0
      ispushed=0
      ipg=self.inter[incident_species]['incident_pgroup']
      tpg=self.inter[incident_species]['target_pgroup']
      epg=self.inter[incident_species]['emitted_pgroup']
      for js in incident_species.jslist:
        npinc+=ipg.nps[js]
        if ipg.ldts[js]:ispushed=1
      if npinc==0 or not ispushed:continue
      if 1:
        it=0
        for js in incident_species.jslist:
          i1 = ipg.ins[js] - 1 + top.it%self.stride
          i2 = ipg.ins[js] + ipg.nps[js] - 1
          xi=ipg.xp[i1:i2:self.stride]#.copy()
          ni = shape(xi)[0]
          if ni==0: continue
          yi=ipg.yp[i1:i2:self.stride]#.copy()
          zi=ipg.zp[i1:i2:self.stride].copy()
          ex=ipg.ex[i1:i2:self.stride].copy()
          ey=ipg.ey[i1:i2:self.stride].copy()
          ez=ipg.ez[i1:i2:self.stride].copy()
          if top.boost_gamma>1.:
            bx=ipg.bx[i1:i2:self.stride].copy()
            by=ipg.by[i1:i2:self.stride].copy()
            bz=ipg.bz[i1:i2:self.stride].copy()
          gaminvi=ipg.gaminv[i1:i2:self.stride].copy()
          uxi=ipg.uxp[i1:i2:self.stride].copy()
          uyi=ipg.uyp[i1:i2:self.stride].copy()
          uzi=ipg.uzp[i1:i2:self.stride].copy()
          
          if top.wpid > 0:
            # --- Save the wpid of the incident particles so that it can be
            # --- passed to the emitted particles.
            wi = ipg.pid[i1:i2:self.stride,top.wpid-1]
          else:
            wi = 1.
          if top.injdatapid > 0:
            # --- Save the injdatapid of the incident particles so that it can be
            # --- passed to the emitted particles.
            injdatapid = ipg.pid[i1:i2:self.stride,top.injdatapid-1]
          else:
            injdatapid = None
          # --- get velocity in lab frame if using a boosted frame of reference
          if top.boost_gamma>1.:
            uzboost = clight*sqrt(top.boost_gamma**2-1.)
            setu_in_uzboosted_frame3d(ni,uxi,uyi,uzi,gaminvi,
                                      -uzboost,
                                      top.boost_gamma)
          if top.boost_gamma>1.:
            seteb_in_boosted_frame(ni,ex,ey,ez,bx,by,bz,0.,0.,-uzboost,top.boost_gamma)

          Emag = sqrt(ex*ex+ey*ey+ez*ez)
          Emag = where(Emag<1.e-10,1.e-10,Emag)

          # probability
          Z = nint(incident_species.charge/echarge)
          prob = self.GetADKprobSI(Emag,Z,incident_species.type,dt*ipg.ndts[js]*self.stride,l_dc2ac=False)

          # --- Get a count of the number of collisions for each particle.
          ncoli = where(ranf(prob)<prob,1,0)

          # --- Select the particles that will collide
          io=compress(ncoli>0,arange(ni))
          nnew = len(io)

          if 1:#self.inter[incident_species]['emitted_energy0'][it] is None:
            # --- When emitted_energy0 is not specified, use the velocity of
            # --- the incident particles for the emitted particles.
            uxnew = uxi
            uynew = uyi
            uznew = uzi

          if 1:#self.inter[incident_species]['remove_incident'][it]:
            # --- if projectile is modified, then need to delete it
            put(ipg.gaminv,array(io)*self.stride+i1,0.)

          # --- The position of the incident particle is at or near the incident particle
          xnew = xi
          ynew = yi
          znew = zi

          # --- Loop until there are no more collision events that need handling
          while(nnew>0):

            # --- The emitted particles positions, in some cases, are slightly
            # --- offset from the incident
            xnewp = xnew[io]
            ynewp = ynew[io]
            znewp = znew[io]
            xnew = xnewp#+(ranf(xnewp)-0.5)*1.e-10*self.dx
            ynew = ynewp#+(ranf(ynewp)-0.5)*1.e-10*self.dy
            znew = znewp#+(ranf(znewp)-0.5)*1.e-10*self.dz
            if top.wpid == 0: 
              w = 1.
            else:
              w = wi[io]

            # --- The injdatapid value needs to be copied to the emitted particles
            # --- so that they are handled properly in the region near the source.
            if top.injdatapid > 0: injdatapid = injdatapid[io]

            for emitted_species in self.inter[incident_species]['emitted_species'][it]:

              if self.inter[incident_species]['emitted_energy0'][it] is not None:
                # --- Create new velocities for the emitted particles.
                ek0ionel = self.inter[incident_species]['emitted_energy0'][it]
                esigionel = self.inter[incident_species]['emitted_energy_sigma'][it]
                if esigionel==0.:
                  ek = zeros(nnew)
                else:
                  ek = SpRandom(0.,esigionel,nnew) #kinetic energy
                ek=abs(ek+ek0ionel) #kinetic energy
                fact = jperev/(emass*clight**2)
                gamma=ek*fact+1.
                u=clight*sqrt(ek*fact*(gamma+1.))
                # velocity direction: random in (x-y) plane plus small longitudinal component:
                phi=2.*pi*ranf(u)
                vx=cos(phi); vy=sin(phi); vz=0.01*ranf(u)
                # convert into a unit vector:
                vu=sqrt(vx**2+vy**2+vz**2)
                # renormalize:
                vx/=vu; vy/=vu; vz/=vu
                # find components of v*gamma:
                uxnewp=u*vx
                uynewp=u*vy
                uznewp=u*vz
              else:
                # --- If the emitted energy was not specified, the emitted particle will be
                # --- given the same velocity of the incident particle.
                uxnewp=uxnew[io]
                uynewp=uynew[io]
                uznewp=uznew[io]

              ginewp = 1./sqrt(1.+(uxnewp**2+uynewp**2+uznewp**2)/clight**2)
              # --- get velocity in boosted frame if using a boosted frame of reference
              if top.boost_gamma>1.:
                setu_in_uzboosted_frame3d(shape(ginewp)[0],uxnewp,uynewp,uznewp,ginewp,
                                          uzboost,
                                          top.boost_gamma)

              if self.l_verbose:print 'add ',nnew, emitted_species.name,' from by impact ionization:',incident_species.name,'+',((target_species is None and 'background gas') or target_species.name)
              if self.inter[incident_species]['remove_incident'][it] and (emitted_species.type is incident_species.type):
                self.addpart(nnew,xnewp,ynewp,znewp,uxnewp,uynewp,uznewp,ginewp,epg,emitted_species.jslist[0],
                             self.inter[incident_species]['emitted_tag'][it],injdatapid,w)
              else:
                self.addpart(nnew,xnew,ynew,znew,uxnewp,uynewp,uznewp,ginewp,epg,emitted_species.jslist[0],
                             self.inter[incident_species]['emitted_tag'][it],injdatapid,w)
            ncoli = ncoli[io] - 1
            io = arange(nnew)[ncoli>0]
            nnew = len(io)

    # make sure that all particles are added and cleared
    for pg in self.x:
      for js in self.x[pg]:
        self.flushpart(pg,js)
        processlostpart(pg,js+1,top.clearlostpart,top.time,top.zbeam)

    if self.l_timing:print 'time ionization = ',time.clock()-t1,'s'
