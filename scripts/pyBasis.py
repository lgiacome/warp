# import all of the neccesary packages
from Numeric import *
from types import *
import RNG
import RandomArray
from pybasisC import *
import os
import string
import re
try:
  import PW
  import PR
except ImportError:
  pass
import __main__
import sys
import cPickle
try:
  import inspect
except ImportError:
  pass
# --- Add line completion capability
try:
  import readline
except ImportError:
  pass
else:
  import rlcompleter
  readline.parse_and_bind("tab: complete")

Basis_version = "$Id: pyBasis.py,v 1.29 2003/01/23 21:31:22 dave Exp $"

if sys.platform in ['sn960510','linux-i386']:
  true = -1
  false = 0
else:
  true = 1
  false = 0

# --- Convenience function modeled after the iota of basis
def iota(low,high=None,step=1):
  if high is None:
    if step > 0:
      return arange(1,low+1,step)
    else:
      return arange(low,0,step)
  else:
    if step > 0:
      return arange(low,high+1,step)
    else:
      return arange(low,high-1,step)

# --- Converts an array of characters into a string.
def arraytostr(a,strip=true):
  a = array(a)
  if len(shape(a)) == 1:
    result = ''
    for c in a:
      result = result + c
    if strip: result = string.strip(result)
  elif len(shape(a)) == 2:
    result = []
    for i in xrange(shape(a)[1]):
      result.append(arraytostr(a[:,i]))
  return result

# --- Convenience function to do printing
def remark(s):
  print s

# --- Allows int operation on arrrays
builtinint = int
def int(x):
  if type(x) == ArrayType:
    return x.astype(Int)
  else:
    return builtinint(x)

# --- Return the nearest integer
def nint(x):
  if type(x) == ArrayType:
    return where(greater(x,0),int(x+0.5),-int(abs(x)+0.5))
  else:
    if x >= 0: return int(x+0.5)
    else: return -int(abs(x)+0.5)

# --- Replicate the sign function
def sign(x,y):
  if type(x) == ArrayType:
    result = where(greater(y,0.),abs(x),-abs(x))
    result = where(equal(y,0.),0.,result)
    return result
  else:
    if y > 0:
      return abs(x)
    elif y < 0:
      return -abs(x)
    else:
      return 0

# --- These are replacements for array creation routines which create
# --- arrays which have the proper ordering for fortran. When arrays created
# --- with these commands are passed to a fortran subroutine, no copies are
# --- needed to get the data into the proper order for fortran.
def fones(shape,typecode=Int):
  try:
    s = list(shape)
  except TypeError:
    s = list([shape])
  s.reverse()
  return transpose(ones(s,typecode))
def fzeros(shape,typecode=Int):
  try:
    s = list(shape)
  except TypeError:
    s = list([shape])
  s.reverse()
  return transpose(zeros(s,typecode))

# --- This function appends a new element to the end of an array.
# --- It is not very efficient since it creates a whole new array each time.
def arrayappend(x,a):
  xshape = list(shape(x))
  if type(a) == ArrayType:
    pass
  elif type(a) == ListType:
    a = array(a)
  else:
    a = array([a])
  ashape = list(shape(a))
  if len(xshape)==1 and len(ashape)==1:
    xshape[0] = xshape[0] + ashape[0]
    y = zeros(xshape,x.typecode())
    y[0:xshape[0]-ashape[0]] = x
    y[xshape[0]-ashape[0]:] = a
  elif len(xshape)>1 and len(ashape)==1 and xshape[-2]==ashape[0]:
    xshape[-1] = xshape[-1] + 1
    y = zeros(xshape,x.typecode())
    y[...,0:-1] = x
    y[...,-1] = a
  return y

# Convenience function which returns true if variable exists
def exists(x):
  try:
    xtemp = eval(x,locals(),globals())
    return true
  except NameError:
    return false

# Returns the average of the input array
def ave(x,index=0):
  if shape(x)[index] > 0:
    return sum(x,index)/shape(x)[index]
  else:
    return 0.

# --- Returns the max of the multiarray
def maxnd(x):
  """Return the max element of an array of any dimension"""
  xtemp = reshape(x,tuple([product(array(x.shape))]))
  return max(xtemp)
# --- Returns the min of the multiarray
def minnd(x):
  """Return the min element of an array of any dimension"""
  xtemp = reshape(x,tuple([product(array(x.shape))]))
  return min(xtemp)

# Gets next available filename with the format 'root.nnn.suffix'.
def getnextfilename(root,suffix):
  dir = string.join(os.listdir('.'))
  i = 0
  name = root+('.%03d.'%i)+suffix
  while re.search(name,dir):
    i = i + 1
    name = root+('.%03d.'%i)+suffix
  return name

# --- Prints out the documentation of the subroutine or variable.
def doc(f,printit=1):
  # --- The for loop only gives the code something to break out of. There's
  # --- probably a better way of doing this.
  for i in range(1):
    if type(f) == StringType:
        # --- Check if it is a WARP variable
        try:
          d = listvar(f)
          break
        except NameError:
          pass
        # --- Check if it is a module name
        try:
          m = __import__(f)
          try:
            d = m.__dict__[f+'doc']()
            if d is None: d = ''
          except KeyError:
            d = m.__doc__
          break
        except ImportError:
          pass
        # --- Try to get the actual value of the object
        try:
          v = __main__.__dict__[f]
          d = v.__doc__
          break
        except KeyError:
          d = "Name not found"
        except AttributeError:
          d = "No documentation found"
    else:
      # --- Check if it has a doc string
      try:
        d = f.__doc__
      except AttributeError:
        d = "No documentation found"
  if printit: print d
  else:       return d

# --- Print out all variables in a group
def printgroup(pkg=None,group=None,maxelements=10):
  """
Print out all variables in a group or with an attribute
  - pkg: package name
  - group: group name
  - maxelements=10: only up to this many elements of arrays are printed
  """
  assert pkg != None,"package must be specified"
  assert group != None,"group name must be specified"
  if type(pkg) == StringType: pkg = __main__.__dict__[pkg]
  vlist = pkg.varlist(" "+group+" ")
  if not vlist:
    print "Unknown group name "+group
    return
  for vname in vlist:
    v = pkg.getpyobject(vname)
    if v is None:
      print vname+' is not allocated'
    elif type(v) != ArrayType:
      print vname+' = '+str(v)
    else:
      if v.typecode() == 'c':
        print vname+' = "'+str(arraytostr(v))+'"'
      elif size(v) <= maxelements:
        print vname+' = '+str(v)
      else:
        if rank(v) == 1:
          print vname+' = '+str(v[:maxelements])[:-1]+" ..."
        else:
          if shape(v)[0] <= maxelements:
            if rank(v) == 2:
              print vname+' = ['+str(v[:,0])+"] ..."
            elif rank(v) == 3:
              print vname+' = [['+str(v[:,0,0])+"]] ..."
            elif rank(v) == 4:
              print vname+' = [[['+str(v[:,0,0,0])+"]]] ..."
            elif rank(v) == 5:
              print vname+' = [[[['+str(v[:,0,0,0,0])+"]]]] ..."
            elif rank(v) == 6:
              print vname+' = [[[[['+str(v[:,0,0,0,0,0])+"]]]]] ..."
          else:
            if rank(v) == 2:
              print vname+' = ['+str(v[:maxelements,0])[:-1]+" ..."
            elif rank(v) == 3:
              print vname+' = [['+str(v[:maxelements,0,0])[:-1]+" ..."
            elif rank(v) == 4:
              print vname+' = [[['+str(v[:maxelements,0,0,0])[:-1]+" ..."
            elif rank(v) == 5:
              print vname+' = [[[['+str(v[:maxelements,0,0,0,0])[:-1]+" ..."
            elif rank(v) == 6:
              print vname+' = [[[[['+str(v[:maxelements,0,0,0,0,0])[:-1]+" ..."
  
##############################################################################
# Python version of the dump routine. This uses the varlist command to
# list of all of the variables in each package which have the
# attribute attr (and actually attr could be a group name too). It then
# checks on the state of the python object, making sure that unallocated
# arrays are not written out.  Finally, the variable is written out to the
# file with the name in the format vame@pkg.  Additionally, python
# variables can be written to the file by passing in a list of the names
# through vars. The '@' sign is used between the package name and the
# variable name so that no python variable names can be clobbered ('@'
# is not a valid character in python names). The 'ff.write' command is
# used, allowing names with an '@' in them. The writing of python variables
# is put into a 'try' command since some variables cannot be written to
# a pdb file.
def pydump(fname=None,attr=["dump"],vars=[],serial=0,ff=None,varsuffix=None,
           verbose=false):
  """
Dump data into a pdb file
  - fname: dump file name
  - attr=["dump"]: attribute or list of attributes of variables to dump
       Any items that are not strings are skipped. To write no variables,
       use attr=None.
  - vars=[]: list of python variables to dump
  - serial=0: switch between parallel and serial versions
  - ff=None: Allows passing in of a file object so that pydump can be called
       multiple times to pass data into the same file. Note that
       the file must be explicitly closed by the user.
  - varsuffix=None: Suffix to add to the variable names. If none is specified,
       the suffix '@pkg' is used, where pkg is the package name that the
       variable is in. Note that if varsuffix is specified, the simulation
       cannot be restarted from the dump file.
  - verbose=false: When true, prints out the names of the variables as they are
       written to the dump file
  """
  assert fname is not None or ff is not None,\
         "Either a filename must be specified or a pdb file pointer"
  # --- Open the file if the file object was not passed in.
  # --- If the file object was passed in, then don't close it.
  if not ff:
    ff = PW.PW(fname)
    closefile = 1
  else:
    closefile = 0
  # --- Convert attr into a list if needed
  if not (type(attr) == ListType): attr = [attr]
  # --- Loop through all of the packages (getting pkg object).
  # --- When varsuffix is specified, the list of variables already written
  # --- is created. This solves two problems. It gives proper precedence to
  # --- variables of the same name in different packages. It also fixes
  # --- an obscure bug in the pdb package - writing two different arrays with
  # --- the same name causes a problem and the pdb file header is not
  # --- properly written. The pdb code should really be fixed.
  pkgsuffix = varsuffix
  packagelist = package()
  writtenvars = []
  for pname in packagelist:
    pkg = __main__.__dict__[pname]
    if varsuffix is None: pkgsuffix = '@' + pname
    # --- Get variables in this package which have attribute attr.
    vlist = []
    for a in attr:
      if type(a) == StringType: vlist = vlist + pkg.varlist(a)
    # --- Loop over list of variables
    for vname in vlist:
      # --- Check if object is available (i.e. check if dynamic array is
      # --- allocated).
      v = pkg.getpyobject(vname)
      if v is not None:
        writevar = 1
        # --- If serial flag is set, get attributes and if has the parallel
        # --- attribute, don't write it.
        if serial:
          a = pkg.getvarattr(vname)
          if re.search('parallel',a):
            writevar = 0
        # --- Check if variable is a complex array. Currently, these
        # --- can not be written out.
        if type(v) == ArrayType and v.typecode() == Complex:
          writevar = 0
        # --- Write out the variable.
        if writevar:
          if varsuffix is not None:
            if vname in writtenvars:
              if verbose: print "variable "+pname+"."+vname+" skipped since other variable would have same name in the file"
              continue
            writtenvars.append(vname)
          if verbose: print "writing "+pname+"."+vname+" as "+vname+pkgsuffix
          ff.write(vname+pkgsuffix,v)

  # --- Now, write out the python variables (that can be written out).
  # --- If supplied, the varsuffix is append to the names here too.
  if varsuffix is None: varsuffix = ''
  for v in vars:
    # --- Skip python variables that would overwrite fortran variables.
    if len(writtenvars) > 0:
      if v in writtenvars:
        if verbose: print "variable "+v+" skipped since other variable would have same name in the file"
        continue
    # --- Get the value of the variable.
    vval = __main__.__dict__[v]
    # --- Don't try to write out classes. (They don't seem to
    # --- cause problems but this avoids potential problems. The
    # --- class body wouldn't be written out anyway.)
    if type(vval) in [ClassType]: continue
    # --- Write out the source of functions. Note that the source of functions
    # --- typed in interatively is not retrieveable - inspect.getsource
    # --- returns an IOError.
    if type(vval) in [FunctionType]:
      try:
        source = inspect.getsource(vval)
        #if verbose:
        if verbose: print "writing python function "+v+" as "+v+varsuffix+'@function'
        ff.write(v+varsuffix+'@function',source)
      except (IOError,NameError):
        if verbose: print "could not write python function "+v
      continue
    # --- Zero length arrays cannot by written out.
    if type(vval) == ArrayType and product(array(shape(vval))) == 0:
      continue
    # --- Try writing as normal variable.
    # --- The docontinue temporary is needed since python1.5.2 doesn't
    # --- seem to like continue statements inside of try statements.
    docontinue = 0
    try:
      if verbose: print "writing python variable "+v+" as "+v+varsuffix
      ff.write(v+varsuffix,vval)
      docontinue = 1
    except:
      pass
    if docontinue: continue
    # --- If that didn't work, try writing as a pickled object
    try:
      if verbose:
        print "writing python variable "+v+" as "+v+varsuffix+'@pickle'
      ff.write(v+varsuffix+'@pickle',cPickle.dumps(vval,0))
      docontinue = 1
    except (cPickle.PicklingError,TypeError):
      pass
    if docontinue: continue
    # --- All attempts failed so write warning message
    if verbose: print "cannot write python variable "+v
  if closefile: ff.close()

# --- Old version which has different naming for variables
def pydumpold(fname,attr="dump",vars=[]):
  ff = PW.PW(fname)
  for p in package():
    vlist = eval(p+'.varlist("'+attr+'")',__main__.__dict__)
    for v in vlist:
      if eval(p+'.getpyobject("'+v+'")',__main__.__dict__) is not None:
        #exec('ff.'+p+'_'+v+'='+p+'.'+v,__main__.__dict__,locals())
        exec('ff.write("'+p+'@'+v+'",'+p+'.'+v+')',__main__.__dict__,locals())
  for v in vars:
    try:
      exec('ff.'+v+'='+v,__main__.__dict__,locals())
    except:
      pass
  ff.close()


# Python version of the restore routine. It restores all of the variables
# in the pdb file, including any that are not part of a pybasis package.
# An '@' in the name distinguishes between the two. The 'ff.__getattr__' is
# used so that variables with an '@' in the name can be read. The reading
# in of python variables is put in a 'try' command to make it idiot proof.
# More fancy foot work is done to get new variables read in into the
# global dictionary.
def pyrestore(filename=None,fname=None,verbose=0,skip=[],varsuffix=None,ls=0):
  """
Restores all of the variables in the specified file.
  - filename: file to read in from (assumes PDB format)
  - verbose=0: When true, prints out the names of variables which are read in
  - skip=[]: list of variables to skip
  - varsuffix: when set, all variables read in will be given the suffix
               Note that fortran variables are then read into python vars
  - ls=0: when true, prints a list of the variables in the file
          when 1 prints as tuple
          when 2 prints in a column
  """
  # --- The original had fname, but changed to filename to be consistent
  # --- with restart and dump.
  if filename is None: filename = fname
  # --- Make sure a filename was input.
  assert filename is not None,"A filename must be specified"
  # --- open pdb file
  ff = PR.PR(filename)
  # --- Get a list of all of the variables in the file, loop over that list
  vlist = ff.inquire_ls()
  # --- Print list of variables
  if ls:
    if ls == 1: print vlist
    else:
      for l in vlist: print l

  # --- vlist is looped over twice. The first time reads in all of the scalar
  # --- variables and the python variables. The second reads in the arrays.
  # --- This is done so that the integers which specify the dimensions of
  # --- of arrays are read in before the array itself is. In some cases,
  # --- not having the integers read in first would cause problems
  # --- (so far only in the f90 version).
  for v in vlist:

    # --- First, extract variable name
    if len(v) > 4 and v[-4]  == '@': vname = v[:-4]
    elif v[-7:] == '@pickle':        vname = v[:-7]
    elif v[-7:] == '@global':        vname = v[:-7]
    elif v[-9:] == '@function':      vname = v[:-9]
    else:                            vname = v

    # --- If variable in the skip list, then skip
    if vname in skip or \
       (len(v) > 4 and v[-4]=='@' and v[-3:]+'.'+v[:-4] in skip):
      if verbose: print "skipping "+v
      continue

    # --- Add suffix to name is given.
    # --- varsuffix is wrapped in str in case a nonstring was passed in.
    if varsuffix is not None: vname = vname + str(varsuffix)

    # --- If the variable has the suffix '@pkg' then it is a warp variable.
    # --- If varsuffix is supplied, than this branch is skipped since
    # --- data is read into python variables.
    if len(v) > 4 and v[-4]=='@' and varsuffix is None:
      vname = v[-3:]+'.'+v[:-4]
      try:
        # --- On this pass, only assign to scalars.
        if type(ff.__getattr__(v)) != ArrayType:
          # --- Simple assignment is done for scalars, using the exec command
          if verbose: print "reading in "+vname
          exec(vname+'=ff.__getattr__(v)',__main__.__dict__,locals())
      except:
        # --- The catches errors in cases where the variable is not an
        # --- actual warp variable, for example if it had been deleted
        # --- after the dump was originally made.
        print "Warning: There was problem restoring %s"% (vname)
    elif v[-7:] == '@pickle':
      # --- Thses would be interpreter variables written to the file
      # --- as pickled objects. The data is unpickled and the variable
      # --- in put in the main dictionary.
      try:
        if verbose: print "reading in pickled variable "+v[:-7]
        __main__.__dict__[vname] = cPickle.loads(ff.__getattr__(v))
      except:
        if verbose: print "error with variable "+v[:-7]
    elif v[-7:] == '@global':
      # --- These would be interpreter variables written to the file
      # --- from Basis. A simple assignment is done and the variable
      # --- in put in the main dictionary.
      try:
        if verbose: print "reading in Basis variable "+v[:-7]
        __main__.__dict__[vname] = ff.__getattr__(v)
      except:
        if verbose: print "error with variable "+v[:-7]
    elif v[-9:] == '@function':
      # --- Skip functions which have already been defined in case the user
      # --- has made source updates since the dump was made.
      if __main__.__dict__.has_key(vname): 
        if verbose:
          print "skipping python function %s since it already is defined"%v[:-9]
      else:
        try:
          if verbose: print "reading in python function"+v[:-7]
          source = ff.__getattr__(v)
          exec(source,__main__.__dict__)
        except:
          if verbose: print "error with function "+v[:-7]
    elif v[-9:] == '@parallel':
      # --- Ignore variables with suffix @parallel
      pass
    else:
      # --- These would be interpreter variables written to the file
      # --- from python (or other sources). A simple assignment is done and
      # --- the variable in put in the main dictionary.
      # --- Note that this branch is used for fortran variables when varsuffix
      # --- is set.
      try:
        if verbose: print "reading in python variable "+v
        __main__.__dict__[vname] = ff.__getattr__(v)
      except:
        if verbose: print "error with variable "+v

  # --- Now loop again to read in the arrays. This does not need to be done
  # --- if varsuffix was specified since in that case everything has
  # --- already been read in.
  if varsuffix is None:
    for v in vlist:
      # --- First, extract variable name
      if len(v) > 4 and v[-4]  == '@': vname = v[:-4]
      elif v[-7:] == '@pickle':        vname = v[:-7]
      elif v[-7:] == '@global':        vname = v[:-7]
      else:                            vname = v
      # --- If variable in the skip list, then skip
      if vname in skip or \
         (len(v) > 4 and v[-4]=='@' and v[-3:]+'.'+v[:-4] in skip):
        if verbose: print "skipping "+v
        continue
      if len(v) > 4 and v[-4]=='@':
        try:
          pkg = eval(v[-3:],__main__.__dict__)
          if type(ff.__getattr__(v)) == ArrayType:
            # --- forceassign is used, allowing the array read in to have a
            # --- different size than the current size of the warp array.
            if verbose: print "reading in "+v[-3:]+"."+v[:-4]
            pkg.forceassign(v[:-4],ff.__getattr__(v))
        except:
          print "Warning: There was problem restoring %s"% (v[-3:]+'.'+v[:-4])
  ff.close()

# --- create an alias for pyrestore
restore = pyrestore

def restoreold(fname):
  ff = PR.PR(fname)
  vlist = ff.inquire_ls()
  for v in vlist:
    if len(v) > 4 and v[3]=='@':
      if type(eval('ff.__getattr__("'+v+'")')) == ArrayType:
        exec(v[0:3]+'.forceassign("'+v[4:]+'",ff.__getattr__("'+v+'"))',
             __main__.__dict__,locals())
      else:
        #exec(v[0:3]+'.'+v[4:]+'=ff.'+v,__main__.__dict__,locals())
        exec(v[0:3]+'.'+v[4:]+'=ff.__getattr__("'+v+'")',
             __main__.__dict__,locals())
    else:
      try:
        exec('%s=ff.%s;__main__.__dict__["%s"]=%s'%(v,v,v,v))
      except:
        pass
  ff.close()



def Basisdoc():
  print """
iota(): returns a array of sequential integers
arraytostr(): converts an array of chars to a string
remark(): same as print
int(): converts data to integer
nint(): converts data to nearest integer
sign(): emulation of sign function
fones(): returns multi-dimensional array with fortran ordering
fzeros(): returns multi-dimensional array with fortran ordering
arrayappend(): appends to multi-dimensional array
exists(): checks if a variable exists
ave(): averages an array of numbers
maxnd(): finds max of multi-dimensional array
minnd(): finds min of multi-dimensional array
getnextfilename(): finds next available file name in a numeric sequence
doc(): prints info about variables and functions
printgroup(): prints all variables in the group or with an attribute
pydump(): dumps data into pdb format file
pyrestore(): reads data from pdb format file
restore(): equivalent to pyrestore
"""
