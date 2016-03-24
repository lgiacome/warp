"""
This files defines a set of function that extract
the fields in the physical part of each subdomain.

Different function are used for different geometries.

The fields are returned in a data structure which is close to
their final layout, in the openPMD file.
"""
import numpy as np
from data_dict import circ_dict_quantity, cart_dict_quantity, \
    x_offset_dict, y_offset_dict

def get_dataset( dim, em, quantity, lgather,
                 sbs=[1,1,1],iz_slice=None, transverse_centered=False ):
    """
    Extract fields from the grid and return them in a format
    which is close to their final layout in the openPMD file.

    This format depends on the slicing (iz_slice) and geometry (dim)
    See below for more information of the shape of the output of this function

    Parameters
    ----------
    dim: string
        Either "2d", "circ" or "3d"
        Indicates the geometry of the fields

    em: an EM3DSolver object
        The object from which to extract the fields

    quantity: string
        Describes which field is being written.
        (Either rho, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy or Jz)

    lgather: boolean
        Defines if data is gathered on me (process) = 0
        If "False": No gathering is done

    iz_slice: int or None
        If None, the full field array is returned
        If not None, this is the index of the slice to be returned,
        within the local subdomain
        
    sbs: array of int of size 3 defining subsampling period in each dir
        by default, sbs=[1,1,1] i.e all grid points are dumped 
        
    transverse_centered: bool, optional
        Whether to return fields that are always transversally centered
        (implies that staggered fields will be transversally averaged)

    Returns
    -------
    An array of reals is returned with a final format close to the final openPMD layout.

    When there is no slicing (iz_slice is None), the returned array is of shape
    - (Nx+1, Nz+1) if dim="2d"
    - (Nx+1, Ny+1, Nz+1) if dim="3d"
    - ( 2*em.circ_m+1, Nx+1, Nz+1) if dim="circ"
      (real and imaginary part are separated for each mode)
    When there is slicing (iz_slice is an integer), the shape is
    - (Nx+1,) if dim="2d"
    - (Nx+1, Ny+1) if dim="3d"
    - ( 2*em.circ_m+1, Nx+1) if dim="circ"
      (real and imaginary part are separated for each mode)

    In the above Nx is either em.nxlocal (if lgather is False) or the global
    em.nx (if lgather is True). The same holds for Ny and Nz
    """
    if dim=="circ":
        return( get_circ_dataset( em, quantity, lgather=lgather,
            iz_slice=iz_slice,sbs=sbs, transverse_centered=transverse_centered ) )
    elif dim=="2d":
        return( get_cart2d_dataset( em, quantity, lgather=lgather,
            iz_slice=iz_slice, sbs=sbs, transverse_centered=transverse_centered ) )
    elif dim=="3d":
        return( get_cart3d_dataset( em, quantity, lgather=lgather,
            iz_slice=iz_slice,sbs=sbs, transverse_centered=transverse_centered ) )

def get_circ_dataset( em, quantity, lgather,
                      iz_slice=None, sbs=[1,1,1], transverse_centered=False ):
    """
    Get a given quantity in Circ coordinates

    Parameters
    ----------
    See the docstring of the function get_dataset

    Returns
    -------
    An array of reals whose format is close to the final openPMD layout.

    When there is no slicing (iz_slice is None), the returned array is of shape
    ( 2*em.circ_m+1, Nx+1, Nz+1)
    When there is slicing (iz_slice is an integer), the shape is
    ( 2*em.circ_m+1, Nx+1)
    (real and imaginary part are separated for each mode)

    In the above Nx is either em.nxlocal (if lgather is False) or the global
    em.nx (if lgather is True). The same holds for Nz.
    """
    # Extract either a slice or the full array

    if quantity in ['Er', 'Et', 'Ez', 'Br', 'Bt', 'Bz', \
                        'Jr', 'Jt', 'Jz', 'rho' ]:
        # Get the field name in Warp
        field_name = circ_dict_quantity[quantity]
        # Extract the data, either of a slice or the full array
        if iz_slice is None:
            # Extract mode 0
            F = getattr( em.fields, field_name )[:,0,:]
            # Extract higher modes
            if em.circ_m > 0:
                F_circ =  getattr( em.fields, field_name + '_circ')
        else:
            iz = em.nzguard + iz_slice
            # Extract mode 0
            F = getattr( em.fields, field_name )[:,0,iz]
            # Extract higher modes
            if em.circ_m > 0:
                F_circ =  getattr( em.fields, field_name + '_circ')[:,iz,:]
    else:
        raise ValueError('Unknown quantity: %s' %quantity)

    # Cut away the guard cells
    # If needed, average the slices for a staggered field
    # In the x direction
    nxg = em.nxguard
    if transverse_centered and (x_offset_dict[quantity]==0.5):
        F = 0.5*( F[ nxg-1:-nxg-1 ] + F[ nxg:-nxg ] )
        if em.circ_m > 0:
            F_circ = 0.5*( F_circ[ nxg-1:-nxg-1 ] + F_circ[ nxg:-nxg ] )
    else:
        F = F[ nxg:-nxg:sbs[0] ]
        if em.circ_m > 0:
            F_circ = F_circ[ nxg:-nxg ]
    # In the z direction
    if iz_slice is None:
        nzg = em.nzguard
        F = F[ :, nzg:-nzg:sbs[2] ]
        if em.circ_m > 0:
            F_circ = F_circ[:, nzg:-nzg, :]
                        
    # Gather array if lgather = True
    # (Multi-proc operations using gather)
    # Only done in non-parallel case
    if lgather is True:
        F = em.gatherarray( F )
        if em.circ_m > 0:
            F_circ = em.gatherarray( F_circ )

    # Subsample field
    F=F[::sbs[0], :]
    if em.circ_m>0: 
        F_circ=F[::sbs[0], :, :]
    if (iz_slice is None):
        F=F[:, ::sbs[2]]
        if em.circ_m>0: 
            F_circ=F[:, ::sbs[2], :]

    # Reshape the array so that it is stored in openPMD layout,
    # with real and imaginary part of each mode separated
    if F is not None:
        # Check that the present process participates in writing data
        if iz_slice is None:
            Ftot = np.empty( ( 1+2*em.circ_m, F.shape[0], F.shape[1] ) )
        else:
            Ftot = np.empty( ( 1+2*em.circ_m, F.shape[0] ) )
        # Fill the array
        Ftot[ 0, ... ] = F[...]
        for m in range(em.circ_m):
            Ftot[ 2*m+1, ... ] = F_circ[..., m].real
            Ftot[ 2*m+2, ... ] = F_circ[..., m].imag
    else:
        Ftot = None

    return( Ftot )

def get_cart3d_dataset( em, quantity, lgather,
                      iz_slice=None, sbs=[1,1,1], transverse_centered=False ):
    """
    Get a given quantity in 3D Cartesian coordinates

    Parameters
    ----------
    See the docstring of the function get_dataset

    Returns
    -------
    An array of reals whose format is close to the final openPMD layout.

    When there is no slicing (iz_slice is None), the returned array is of shape
    ( Nx+1, Ny+1, Nz+1)
    When there is slicing (iz_slice is an integer), the shape is
    ( Nx+1, Ny+1)

    In the above Nx is either em.nxlocal (if lgather is False) or the global
    em.nx (if lgather is True). The same holds for Ny and Nz.
    """
    # Extract either a slice or the full array

    # Treat the fields E, B, rho in a more systematic way
    if quantity in ['Ex', 'Ey', 'Ez', \
            'Bx', 'By', 'Bz', 'Jx', 'Jy', 'Jz', 'rho' ]:
        # Get the field name in Warp
        field_name = cart_dict_quantity[quantity]
        # Extract the data
        if iz_slice is None:
            F = getattr( em.fields, field_name )[:,:,:]
        else:
            F = getattr( em.fields, field_name )[:,:,em.nzguard+iz_slice]
    else:
        raise ValueError('Unknown quantity: %s' %quantity)

    # Cut away the guard cells
    # If needed, average the slices for a staggered field
    # In x
    nxg = em.nxguard
    if transverse_centered and (x_offset_dict[quantity]==0.5):
        F = 0.5*( F[ nxg-1:-nxg-1 ] + F[ nxg:-nxg ] )
    else:
        F = F[ nxg:-nxg ]
    # In y
    nyg = em.nyguard
    if transverse_centered and (y_offset_dict[quantity]==0.5):
        F = 0.5*( F[ :, nyg-1:-nyg-1 ] + F[ :, nyg:-nyg ] )
    else:
        F = F[ :, nyg:-nyg ]
    # In the z direction
    if iz_slice is None:
        nzg = em.nzguard
        F = F[ :, :, nzg: -nzg ]

    # Gather array if lgather = True
    # (Mutli-proc operations using gather)
    # Only done in non-parallel case
    if lgather is True:
        if iz_slice is not None:
            raise ValueError('Incompatible parameters in '
                'get_cart2d_dataset: lgather=True and iz_slice not None')
        F = em.gatherarray( F )

    # Subsample field
    if (F is not None): 
        F=F[::sbs[0],::sbs[1],:]
        if (iz_slice is None):
            F=F[:,:,::sbs[2]]

    return( F )


def get_cart2d_dataset( em, quantity, lgather,
                      iz_slice=None, sbs=[1,1,1], transverse_centered=False ):
    """
    Get a given quantity in 2D Cartesian coordinates

    Parameters
    ----------
    See the docstring of the function get_dataset

    Returns
    -------
    An array of reals whose format is close to the final openPMD layout.

    When there is no slicing (iz_slice is None), the returned array is of shape
    ( Nx+1, Nz+1)
    When there is slicing (iz_slice is an integer), the shape is
    ( Nx+1,)

    In the above Nx is either em.nxlocal (if lgather is False) or the global
    em.nx (if lgather is True). The same holds for Nz.
    """
    # Extract either a slice or the full array

    # Treat the fields E, B, rho in a more systematic way
    if quantity in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', \
                        'Jx', 'Jy', 'Jz', 'rho' ]:
        # Get the field name in Warp
        field_name = cart_dict_quantity[quantity]
        # Extract the data
        if iz_slice is None:
            F = getattr( em.fields, field_name )[:,0,:]
        else:
            F = getattr( em.fields, field_name )[:,0,em.nzguard+iz_slice]
    else:
        raise ValueError('Unknown quantity: %s' %quantity)

    # Cut away the guard cells
    # If needed, average the slices for a staggered field
    # In the x direction
    nxg = em.nxguard
    if transverse_centered and (x_offset_dict[quantity]==0.5):
        F = 0.5*( F[ nxg-1:-nxg-1 ] + F[ nxg:-nxg ] )
    else:
        F = F[ nxg : -nxg ]
    # In the z direction
    if iz_slice is None:
        nzg = em.nzguard
        F = F[ :, nzg: -nzg ]

    # Gather array if lgather = True
    # (Mutli-proc operations using gather)
    # Only done in non-parallel case
    if lgather is True:
        if iz_slice is not None:
            raise ValueError('Incompatible parameters in '
                'get_cart2d_dataset: lgather=True and iz_slice not None')
        F = em.gatherarray( F )
        
    # Subsample field
    F=F[::sbs[0],:]
    if (iz_slice is None):
        F=F[:,::sbs[2]]
    return( F )

def get_global_indices(ifull,nfull,sbsp): 
    """
    Get new grid subdomain start indices and sizes with subsampling 

    Parameters
    ----------
    ifull: array of int of size nproc_dir with nproc_dir the 
        number of procs along oa given direction. This array 
        contains start indices of each MPI subdomain 

    nfull: array of int of size nproc_dir with nproc_dir the 
        number of procs along oa given direction. This array 
        contains the sizes (number of cells) of each MPI subdomain 

    sbsp: int defining subsampling period in a given dir

    Returns
    -------
    isub: (array of int) start indices of each MPI subdomain for dumping with subsampling

    nsub: (array of int) sizes of each MPI subdomain for dumping with subsampling
    """
    isub=np.zeros(np.size(ifull))
    nsub=np.zeros(np.size(nfull))
    isub[0] = ifull[0]
    nsub[0] = nfull[0]
    for i in xrange(1,len(ifull)): 
        isub[i]=isub[i-1]+nsub[i-1]
        nsub[i]=np.size(np.arange(ifull[i],ifull[i]+nfull[i]+1,sbsp))-1
    return [isub,nsub]
