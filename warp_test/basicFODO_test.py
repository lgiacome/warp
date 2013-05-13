from warp import *
from nose.tools import assert_almost_equal

# --- This is needed to turn off graphical output
top.lprntpara = false
top.lpsplots = false
def setup():
    """This dummy function replaces the setup from warp. Nose calls setup
    before running the test functions (and there is no way of preventing it)."""
    pass

# --- Set four-character run id, comment lines, user's name.
top.pline2   = "Test of FODO lattice"
top.pline1   = "S-G beam"
top.runmaker = "David P. Grote"

# --- Create the beam species
beam = Species(type=Potassium,charge_state=+1,name="Beam species")

# --- Set input parameters describing the beam, 72 to 17.
beam.a0       = 15.601528729428563*mm
beam.b0       =  8.759246730187062*mm
beam.emit     = 62.46985168627256e-06
beam.ap0      = 0.e0
beam.bp0      = 0.e0
beam.ibeam    = 2.e-03
beam.vbeam    = 0.e0
beam.ekin     = 80.e3
derivqty()
beam.vthz     = .5e0*beam.vbeam*beam.emit/sqrt(beam.a0*beam.b0) # Vthz ~ Vthperp

# +++ Set up arrays describing lattice.
# --- Set temp variables.
hlp     = 36.0e-2  # half lattice period length
piperad = 2.0e-2 # pipe radius
quadlen = 11.e-2   # quadrupole length
dbdx    = 8.4817660221033488

# --- Set general lattice variables.
top.tunelen   = 2.e0*hlp
env.zl        = -hlp
env.zu        = +hlp
env.dzenv     = top.tunelen/64

# --- Set up quadrupoles (matching, laps, and end quads are all the same).
top.zlatperi  = 2.e0*hlp
addnewquad(zs=    - quadlen/2.,
           ze=    + quadlen/2.,
           db=-dbdx)
addnewquad(zs=hlp - quadlen/2.,
           ze=hlp + quadlen/2.,
           db=+dbdx)
addnewquad(zs=2.*hlp - quadlen/2.,
           ze=2.*hlp + quadlen/2.,
           db=-dbdx)

# +++ Set input parameters describing the 3d simulation.
w3d.nx = 16
w3d.ny = 16
w3d.nz = 64
steps_p_perd = 50
top.dt = (top.tunelen/steps_p_perd)/beam.vbeam
top.prwall = piperad

# --- Set to finite beam.
top.pbound0  = top.pboundnz = periodic
top.pboundxy = absorb
w3d.xmmin = -piperad
w3d.xmmax =  piperad
w3d.ymmin = -piperad
w3d.ymmax =  piperad
w3d.zmmin = -hlp
w3d.zmmax = +hlp

# --- Set pulse length.
top.zimin = -hlp
top.zimax = +hlp

# --- Load Semi-Gaussian cigar beam.
top.npmax = 20000
w3d.distrbtn = "semigaus"
w3d.xrandom = "digitrev"
w3d.vtrandom = "digitrev"
w3d.vzrandom = "digitrev"
w3d.ldprfile = "polar"

# --- set up field solver
w3d.l4symtry = true
w3d.bound0 = periodic
w3d.boundnz = periodic
w3d.boundxy = dirichlet

top.lhxrmsz = true
top.lhyrmsz = true

# --- Run the envelope solver to provide data used to initialize particles.
package("env")
generate()
step()

# --- Generate the PIC code (allocate storage, load ptcls, t=0 plots, etc.).
package("w3d")
generate()

step(steps_p_perd)

def test_sigma0x():
    assert_almost_equal(env.sig0x,72.)
def test_sigma0y():
    assert_almost_equal(env.sig0y,72.)
def test_sigmax():
    assert_almost_equal(env.sigx,20.)
def test_sigmay():
    assert_almost_equal(env.sigy,20.)

def test_deltaa():
    assert_almost_equal(env.deltaa,0.)
def test_deltaap():
    assert_almost_equal(env.deltaap,0.)
def test_deltab():
    assert_almost_equal(env.deltab,0.)
def test_deltabp():
    assert_almost_equal(env.deltabp,0.)

def test_initxenv():
    assert_almost_equal(abs(2*top.hxrmsz[:,0,0] - env.aenv).max(),0.,delta=0.0005)
def test_inityenv():
    assert_almost_equal(abs(2*top.hyrmsz[:,0,0] - env.benv).max(),0.,delta=0.0005)

def test_xenvafteroneperiod():
    assert_almost_equal(abs(2*top.xrmsz[:,0] - env.aenv).max(),0.,delta=0.001)
def test_yenvafteroneperiod():
    assert_almost_equal(abs(2*top.yrmsz[:,0] - env.benv).max(),0.,delta=0.001)

