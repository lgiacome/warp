from generic_tools import absorb, EM3D, any

def initialize_beam_fields( em, dim, beam, w3d, top,
                            beam_pgroup=None, beam_jslist=None ) :
    """
    Fill the fields Ex, Ey, Ez, Bx, By, Bz of the object 'em'
    with values corresponding to the space charge fields that correspond
    to the species contained in the species beam.

    Parameters
    ----------
    em : an EM3D object
        Contains the field data and methods

    dim : string
        Either "2d" or "3d"

    beam : a Species object
        Contains the species that represent the beam.
        Instead of `beam`, the user also has the option of passing
        the corresponding variables `beam_pgroup` and `beam_jslist`
        In this case, `beam` should be None.

    w3d, top : Forthon objects

    beam_pgroup: a ParticleGroup
        Contain the species which represents the beam (along with others)
        
    beam_jslist: a list of ints
        Indicates which species in beam_pgroup represents the beam
    """

    # Change the boundary conditions to absorbing
    # (required by the multi-grid solver)
    saved_bound0 = w3d.bound0
    saved_boundnz = w3d.boundnz
    saved_boundxy = w3d.boundxy
    w3d.bound0  = 0
    w3d.boundnz = 0
    w3d.boundxy = 0

    # Create a helper solver to calculate the fields of the bunch
    # (This is not the actual solver on which the simulation will be run,
    # since the boundary conditions are different)
    em_help = EM3D(
        stencil=em.stencil,
        npass_smooth=em.npass_smooth,
        alpha_smooth=em.alpha_smooth,
        stride_smooth=em.stride_smooth,
        dtcoef=em.dtcoef,
        l_2dxz= (dim in ["2d", "circ"]),
        l_2drz= (dim in ["circ"]),
        l_1dz= (dim=="1d"),
        circ_m = (dim =="circ")*em.circ_m,
        type_rz_depose=1 )
#        l_setcowancoefs=True,
#        l_correct_num_Cherenkov=True )
    
    # Get the initial fields
    em_help.initstaticfields( relat_species=beam, relat_pgroup=beam_pgroup,
                             relat_jslist=beam_jslist )
    
    # Allocate the arrays in em class that contain the fields
    em.allocatedataarrays()

    # Add the fields calculated by the Poisson solver into field arrays
    em.fields.Ex[...] += em_help.fields.Ex[...]
    em.fields.Ey[...] += em_help.fields.Ey[...]
    em.fields.Ez[...] += em_help.fields.Ez[...]
    em.fields.Bx[...] += em_help.fields.Bx[...]
    em.fields.By[...] += em_help.fields.By[...]
    em.fields.Bz[...] += em_help.fields.Bz[...]

    # This is copied from the end of dosolve, and finalizes the fields
    if em.ntsub<1.:
        em.novercycle = nint(1./em.ntsub)
        em.icycle = (top.it-1)%em.novercycle
    else:
        em.novercycle = 1
        em.icycle = 0
    em.setebp()
    if top.efetch[0] != 4:
        em.yee2node3d()
    if any(em.npass_smooth>0):
        em.smoothfields()

    # Reset the field boundary conditions to their initial value
    w3d.bound0 = saved_bound0
    w3d.boundnz = saved_boundnz
    w3d.boundxy = saved_boundxy
