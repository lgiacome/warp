"""
Example Pierce diode calculation.
Hot plate source emitting singly ionized potassium
"""
from warp import *
from warp.run_modes.egun_like import gun

# --- Set four-character run id, comment lines, user's name.
top.pline2   = "Pierce diode example"
top.pline1   = "Injected beam. Semi-Gaus."
top.runmaker = "DPG"

# --- Invoke setup routine for the plotting
setup()

# --- Set the dimensionality
w3d.solvergeom = w3d.RZgeom

# --- Sets method of running
# ---   Steady state gun mode
# ---   Time dependent simulation (when False)
steady_state_gun = True

# --- Basic parameters
channel_radius = 15.*cm

diode_voltage = 93.*kV

# --- Setup source plate
source_radius = 5.5*cm
source_temperature = 0.1 # in eV
source_curvature_radius = 30.*cm # --- radius of curvature of emitting surface
pierce_angle = 67.

# --- Setup diode aperture plate
zplate = 8.*cm # --- plate location
rplate = 5.5*cm # --- aperture radius
plate_width = 2.5*cm # --- thickness of aperture plate

# --- Setup simulation species
beam = Species(type=Potassium, charge_state=+1, name='beam')

# --- Child-Langmuir current between parallel plates
j = 4./9.*eps0*sqrt(2.*echarge*beam.charge_state/beam.mass)*diode_voltage**1.5/zplate**2
diode_current = pi*source_radius**2*j

print("Child-Langmuir current density = ", j)
print("Child-Langmuir current = ", diode_current)

# --- Set basic beam parameters
beam.a0       = source_radius
beam.b0       = source_radius
beam.ap0      = .0e0
beam.bp0      = .0e0
beam.ibeam    = diode_current
beam.vthz     = sqrt(source_temperature*jperev/beam.mass)
beam.vthperp  = sqrt(source_temperature*jperev/beam.mass)
derivqty()

# --- Length of simulation box
runlen = zplate + 5.*cm

# --- Set boundary conditions
# ---   for field solve
w3d.bound0 = dirichlet
w3d.boundnz = neumann
w3d.boundxy = neumann
# ---   for particles
top.pbound0 = absorb
top.pboundnz = absorb
top.prwall = channel_radius

# --- Set field grid size
w3d.xmmin = -channel_radius
w3d.xmmax = +channel_radius
w3d.ymmin = -channel_radius
w3d.ymmax = +channel_radius
w3d.zmmin = 0.
w3d.zmmax = runlen

# --- Field grid dimensions - note that nx and ny must be even.
w3d.nx = w3d.ny = 32
w3d.nz = 32

# --- Set the time step size. This needs to be small enough to satisfy the Courant limit.
dz = (w3d.zmmax - w3d.zmmin)/w3d.nz
vzfinal = sqrt(2.*diode_voltage*jperev/beam.mass)
top.dt = 0.4*(dz/vzfinal)

# --- Specify injection of the particles
top.inject = 2 # 2 means space-charge limited injection
top.rinject = source_curvature_radius # Source radius of curvature
top.npinject = 150 # Approximate number of particles injected each step
top.vinject = diode_voltage
w3d.l_inj_exact = true

# --- If using the RZ geometry, set so injection uses the same geometry
w3d.l_inj_rz = (w3d.solvergeom == w3d.RZgeom)

# --- Set up fieldsolver
f3d.mgtol = 1.e-1 # Multigrid solver convergence tolerance, in volts

solver = MultiGrid2D()
registersolver(solver)

piercezlen = (channel_radius - source_radius)*tan((90.-pierce_angle)*pi/180.)
piercezlen = 0.04
rround = plate_width/2.

# --- Create source conductors

# --- Outer radius of Pierce cone
rpierce = source_radius + piercezlen*tan(pierce_angle*pi/180.)

# --- Depth of curved emitting surface
sourcezlen = (source_radius**2/(source_curvature_radius + sqrt(source_curvature_radius**2 - source_radius**2)))

# --- the rsrf and zsrf specify the line in RZ describing the shape of the source and Pierce cone.
# --- The first segment is an arc, the curved emitting surface.
source = ZSrfrv(rsrf=[0., source_radius, rpierce, channel_radius, channel_radius],
                zsrf=[0., sourcezlen, sourcezlen + piercezlen, sourcezlen + piercezlen, 0.],
                zc=[source_curvature_radius, None, None, None, None],
                rc=[0., None, None, None, None],
                voltage=diode_voltage)

installconductor(source, dfill=largepos)

# --- Create aperture plate
plate = ZRoundedCylinderOut(radius=rplate, length=plate_width, radius2=rround, voltage=0., zcent=zplate)

installconductor(plate,dfill=largepos)

# --- Setup the particle scraper
scraper = ParticleScraper([source, plate])

# --- Set pline1 to include appropriate parameters
if w3d.solvergeom == w3d.RZgeom:
    top.pline1 = ("Injected beam. Semi-Gaus. %dx%d. npinject=%d, dt=%d"%
                  (w3d.nx, w3d.nz, top.npinject, top.dt))
else:
    top.pline1 = ("Injected beam. Semi-Gaus. %dx%dx%d. npinject=%d, dt=%d"%
                  (w3d.nx, w3d.ny, w3d.nz, top.npinject, top.dt))

# --- Generate the PIC code (allocate storage, load ptcls, t=0 plots, etc.)
package("w3d")
generate()

# --- Open up plotting windows
winon()
winon(1, suffix='current')

def beamplots():
    window(0)
    fma()
    pfzr(plotsg=0, cond=0, titles=False)
    source.draw(filled=150, fullplane=False)
    plate.draw(filled=100, fullplane=False)
    ppzr(titles=False)
    limits(w3d.zmminglobal, w3d.zmmaxglobal, 0., channel_radius)
    ptitles('Hot plate source', 'Z (m)', 'R (m)')
    refresh()

    window(1)
    fma()
    pzcurr()
    limits(w3d.zmminglobal, w3d.zmmaxglobal, 0., diode_current*1.5)
    refresh()

if steady_state_gun:
    # --- Steady-state operation
    # --- This does steady-state gun iterations, plotting the z versus r
    # --- after each iteration.
    top.inj_param = 0.2
    for iter in range(10):
        gun(1, ipstep=1, lvariabletimestep=1)
        beamplots()

else:

    # --- Call beamplots after every 20 steps
    @callfromafterstep
    def makebeamplots():
        if top.it%20 == 0:
            beamplots()

    step(700)

# --- Make sure that last plot frames get sent to the cgm file
window(0)
hcp()
window(1)
hcp()
