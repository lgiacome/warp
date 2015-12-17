"""
This files defines a set of function that extract
the fields in the physical part of each subdomain.

Different function are used for different geometries.

The fields are returned in a data structure which is close to
their final layout, in the openPMD file.
"""
import numpy as np
from data_dict import circ_dict_quantity, cart_dict_quantity, \
        circ_dict_Jindex, cart_dict_Jindex, x_offset_dict, y_offset_dict

def get_circ_dataset( em, quantity, lgather,
                      iz_slice=None, transversally_centered=False ):
    """
    Get a given quantity in Circ coordinates

    Parameters
    ----------
    em: an EM3DSolver object
    
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

    transversally_centered: bool, optional
        Whether to return fields that are always transversally centered
        (implies that staggered fields will be transversally averaged)

    Returns
    -------
    A 3d array of reals, of shape (2*em.circ_m+1, nxlocal+1, nzlocal+1)
    if iz_slice is None
    A 2darray of reals, of shape (2*em.circ_m+1, nxlocal+1)
    if iz_slice is not None
    """
    # Extract either a slice or the full array

    # The data layout of J in Warp is special
    if quantity in ['Jr', 'Jt', 'Jz']:
        # Get the array index that corresponds to that component
        i = circ_dict_Jindex[ quantity ]
        # Extract the data, either of a slice or the full array
        if iz_slice is None:
            # Extract mode 0
            F = em.fields.J[:,0,:,i]
            # Extract higher modes
            if em.circ_m > 0:
                F_circ = em.fields.J_circ[:,:,i,:]
        else:
            iz = em.nzguard + iz_slice
            # Extract mode 0
            F = em.fields.J[:,0,iz,i]
            # Extract higher modes
            if em.circ_m > 0:
                F_circ = em.fields.J_circ[:,iz,i,:]
    # Treat the fields E, B, rho in a more systematic way
    elif quantity in ['Er', 'Et', 'Ez', 'Br', 'Bt', 'Bz', 'rho' ]:
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
    if transversally_centered and (x_offset_dict[quantity]==0.5):
        F = 0.5*( F[ nxg-1:-nxg-1 ] + F[ nxg:-nxg ] )
        if em.circ_m > 0:
            F_circ = 0.5*( F_circ[ nxg-1:-nxg-1 ] + F_circ[ nxg:-nxg ] )
    else:
        F = F[ nxg:-nxg ]
        if em.circ_m > 0:
            F_circ = F_circ[ nxg:-nxg ]
    # In the z direction
    if iz_slice is None:
        nzg = em.nzguard
        F = F[ :, nzg:-nzg ]
        if em.circ_m > 0:
            F_circ = F_circ[:, nzg:-nzg, :]

    # Gather array if lgather = True 
    # (Multi-proc operations using gather)
    # Only done in non-parallel case
    if lgather is True:
        F = em.gatherarray( F )
        if em.circ_m > 0:
            F_circ = em.gatherarray( F_circ )

    # Reshape the array so that it is stored in openPMD layout,
    # with real and imaginary part of each mode separated
    if iz_slice is None:
        Ftot = np.empty( ( 1+2*em.circ_m, F.shape[0], F.shape[1] ) )
    else:
        Ftot = np.empty( ( 1+2*em.circ_m, F.shape[0] ) )
    # Fill the array
    Ftot[ 0, ... ] = F[...]
    for m in range(em.circ_m):
        Ftot[ 2*m+1, ... ] = F_circ[..., m].real
        Ftot[ 2*m+2, ... ] = F_circ[..., m].imag
        
    return( Ftot )

def get_cart3d_dataset( em, quantity, lgather,
                      iz_slice=None, transversally_centered=False ):
    """
    Get a given quantity in 3D Cartesian coordinates

    Parameters
    ----------
    em: an EM3DSolver object
    
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

    transversally_centered: bool, optional
        Whether to return fields that are always transversally centered
        (implies that staggered fields will be transversally averaged)

    Returns
    -------
    A 3darray of shape (nxlocal+1, nylocal+1, nzlocal+1) if iz_slice is None
    A 2darray (nxlocal+1, nylocal+1) if iz_slice is not None
    """
    # Extract either a slice or the full array

    # The data layout of J in Warp is special
    if quantity in ['Jx', 'Jy', 'Jz']:
        # Get the array index that corresponds to that component
        i = cart_dict_Jindex[ quantity ]
        # Extract the data
        if iz_slice is None:
            F = em.fields.J[:,:,:,i]
        else:
            F = em.fields.J[:,:,em.nzguard+iz_slice,i]
    # Treat the fields E, B, rho in a more systematic way
    elif quantity in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'rho' ]:
        # Get the field name in Warp
        field_name = cart_dict_quantity[quantity]
        # Extract the data
        if iz_slice is None:
            F = getattr( em.fields, field_name )[:,:,:]
        else:
            F = getattr( em.fields, field_name )[:,:,em.nzguard+iz_slice]

    # Cut away the guard cells
    # If needed, average the slices for a staggered field
    # In x
    nxg = em.nxguard
    if transversally_centered and (x_offset_dict[quantity]==0.5):
        F = 0.5*( F[ nxg-1:-nxg-1 ] + F[ nxg:-nxg ] )
    else:
        F = F[ nxg:-nxg ]
    # In y
    nyg = em.nyguard
    if transversally_centered and (y_offset_dict[quantity]==0.5):
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

    return( F )


def get_cart2d_dataset( em, quantity, lgather,
                      iz_slice=None, transversally_centered=False ):
    """
    Get a given quantity in 2D Cartesian coordinates

    Parameters
    ----------
    em: an EM3DSolver object
    
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

    transversally_centered: bool, optional
        Whether to return fields that are always transversally centered
        (implies that staggered fields will be transversally averaged)

    Returns
    -------
    A 2darray of shape (nxlocal+1, nzlocal+1) if iz_slice is None
    A 1darray (nxlocal+1) if iz_slice is not None
    """
    # Extract either a slice or the full array

    # The data layout of J in Warp is special
    if quantity in ['Jx', 'Jy', 'Jz']:
        # Get the array index that corresponds to that component
        i = cart_dict_Jindex[ quantity ]
        # Extract the data
        if iz_slice is None:
            F = em.fields.J[:,0,:,i]
        else:
            F = em.fields.J[:,0,em.nzguard+iz_slice,i]
    # Treat the fields E, B, rho in a more systematic way
    elif quantity in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'rho' ]:
        # Get the field name in Warp
        field_name = cart_dict_quantity[quantity]
        # Extract the data
        if iz_slice is None:
            F = getattr( em.fields, field_name )[:,0,:]
        else:
            F = getattr( em.fields, field_name )[:,0,em.nzguard+iz_slice]

    # Cut away the guard cells
    # If needed, average the slices for a staggered field
    # In the x direction
    nxg = em.nxguard
    if transversally_centered and (x_offset_dict[quantity]==0.5):
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

    return( F )
