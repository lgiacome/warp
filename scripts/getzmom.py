from warp import *
getzmom_version = "$Id: getzmom.py,v 1.8 2002/03/14 18:34:55 dave Exp $"

def getzmomdoc():
  print """
zmmnt  makes appropriate calls to compiled code to calculate the
       particle moments
  """

def zmmnt(itask=0,js=None,jslist=None,groupsize=None):
  """
zmmnt(itask=0,js=None, jslist=xrange(0,top.ns))
  makes appropriate calls to compiled code to calculate the
  particle moments
  - itask=0 task to carry out
    0 All steps to calculate moments (doing steps 1, 2, and 3 below)
    1 initialization, zeros out moments variables
    2 gather moments
    3 final processing of moments, averaging and derived moments
  - js: select species over which to calculate moments,
        None: calculates moments over range 0 .. top.nx or,
        alternatively, over list of moments specified in jslist
  - jslist: list of species to calculate moments for
  - groupsize=256: size of particle groups sent to getzmmnt
                   Use caution since temp variables are used that are
                   18*groupsize in size.
  """
  uxpo = top.uxp
  uypo = top.uyp
  uzpo = top.uzp

  # If changed, fix Win_Moments arrays
  if (len(top.pnumz) != top.nzmmnt+1): gchange ("Z_Moments")

  # --- This is bad idea and so is commented out.
  # If itask is greater than zero, assume that the intent was to do the
  # calculation regardless of the value of laccumulate_zmoments.  Save
  # the value of laccumulate_zmoments and then set it to false.
# if (itask > 0):
#   l1 = top.laccumulate_zmoments
#   top.laccumulate_zmoments = false

  # Zero out the moments arrays
  if (itask == 0 or itask == 1):
    getzmmnt(1,top.xp,top.yp,top.zp,top.uxp,top.uyp,top.uzp,top.gaminv,
             top.sq,top.sm,top.sw,top.dt/2.,1,
             top.nplive,uxpo,uypo,uzpo,1,top.ns)

  # Calculate the moments
  if jslist is None:
    if js is None:
      jslist = xrange(top.ns)
    else:
      jslist = [js]
  if (itask == 0 or itask == 2):
    for js in jslist:
      for ipmin in xrange(top.ins[js]-1,top.ins[js]+top.nps[js]-1,groupsize):
         ip = min(groupsize, top.ins[js]+top.nps[js]-ipmin-1)
         getzmmnt(ip,top.xp[ipmin:],top.yp[ipmin:],top.zp[ipmin:],
                  top.uxp[ipmin:],top.uyp[ipmin:],top.uzp[ipmin:],
                  top.gaminv[ipmin:],top.sq[js],top.sm[js],top.sw[js],
                  top.dt,2,top.nplive,
                  uxpo[ipmin:],uypo[ipmin:],uzpo[ipmin:],js,top.ns)

  # Do final summing and averaging of the moments
  if (itask == 0 or itask == 3):
    getzmmnt(1,top.xp,top.yp,top.zp,top.uxp,top.uyp,top.uzp,top.gaminv,
             top.sq,top.sm,top.sw,top.dt/2.,3,
             top.nplive,uxpo,uypo,uzpo,1,top.ns)

  # Restore the value of laccumulate_zmoments
# if (itask > 0):
#   top.laccumulate_zmoments = l1

