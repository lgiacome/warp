#
# Python file with some parallel operations
#
from numpy import *
# --- Try import mpi (pyMPI) 
#     - if not found, try to from mpi4py import MPI as mpi (mpi4py) 
#     - if not found then run in serial mode
# --- Note that:
# --- mpi.COMM_WORLD is same as MPI_COMM_WORLD
# --- mpi.WORLD is a duplicate of MPI_COMM_WORLD (pyMPI)
# --- mpi.COMM_WORLD is directly used for mpi4py (mpi4py)
# --- comm_world is used for most communications (and defaults to mpi.WORLD)
try:
    #Try to import pyMPI
    import mpi
    mpi.synchronizeQueuedOutput(None)
    me = mpi.rank
    npes = mpi.procs
    comm_world = mpi.WORLD
    lpyMPIactive = True
    lmpi4pyactive = False
except:
    try:
        #Try to import mpi4py
        from mpi4py import MPI as mpi 
        #mpi.synchronizeQueuedOutput(None) 
        #not available for mpi4py (as far as i know)
        comm_world = mpi.COMM_WORLD    
        me = comm_world.Get_rank()
        npes = comm_world.Get_size()
        lmpi4pyactive = True
        lpyMPIactive = False
    except ImportError:
        #Single core version of WARP
        me = 0
        npes = 1
        comm_world = None
        lpyMPIactive = False
        lmpi4pyactive = False

lparallel = (npes > 1)

if lmpi4pyactive:
    import sys
    globalvar = sys._getframe().f_globals
    #-------------------------------------------------------------------------------
    # synchronizeQueuedOutput 
    # (mpi4pywork around... not synchronized, but only PE=0 prints to the console)
    #-------------------------------------------------------------------------------
    def synchronizeQueuedOutput_mpi4py(out = True, error = False):
        if out == False:
            exec("sys.stdout = sys.__stdout__", globalvar)
        else:
            if me > 0:
                exec("""__mpi_stdoutfile__ = open("/dev/null", "w"); sys.stdout = __mpi_stdoutfile__""",
                    globalvar)

        if error == False:
            exec("sys.stderr = sys.__stderr__", globalvar)
        else:
            if me > 0:
                exec("""__mpi_stderrfile__ = open("/dev/null", "w"); sys.stderr = __mpi_stderrfile__""",
                    globalvar)
        return

def setdefaultcomm_world(comm):
    global comm_world,me,npes
    if not lparallel: return
    if lpyMPIactive: #check if pyMPI is used
        comm_world = comm
        me = comm_world.rank
        npes = comm_world.procs
    if lmpi4pyactive: #check if mpi4py is used
        comm_world = comm
        me = comm_world.Get_rank()
        npes = comm_world.Get_size()

"""
# --- This check is old enough that it is no longer needed.

# --- The interface has changed some in the newest version of pyMPI.
# --- Check the interface to the mpi.recv command. The newer versions
# --- return a tuple instead of just the data itself.
# --- Is there a better way of doing this?
if lparallel:
  mpi.send(me,me)
  _i = mpi.recv(me)
  if type(_i) == TupleType: _newpympi = 1
  else:                     _newpympi = 0
else:
  _newpympi = 1
"""
#_newpympi = 1 #not needed anymore
"""
# --- Old routines for mpirecv for the support of old pyMPI versions.
if _newpympi:
    def mpirecv(pe=0,ms=0,comm=None):
        if comm is None: comm = comm_world
        result,stat = comm.recv(pe,ms)
        return result
else:
    def mpirecv(pe=0,ms=0,comm=None):
        if comm is None: comm = comm_world
        return comm.recv(pe,ms)
"""
# ---------------------------------------------------------------------------
# --- New routines for handling sends and receives supporting pyMPI and mpi4py
# --- Checks if pyMPI or mpi4py is used...
# --- If pyMPI is active, uses the standard pyMPI routines
# --- If mpi4py is active, uses the standard (slow) mpi4py routines
# --- Also the "fast" routines of mpi4py (sending/receiving numpy arrays)
# --- can be used with lmpi4pynp = True. Defaults to "False", as some routines
# --- in parallel.py (e.g. gather) do not behave correctly anymore.
# --- New routines that send and receive numpy arrays should make use of 
# --- the "fast" mpi4py routines. Therefore the array that is sent and received  
# --- needs to be defined on both, the sending and the receiving processor.
# ---------------------------------------------------------------------------

def mpirecv(data = None, source = 0, tag = 0, comm = None, lmpi4pynp = False):
    if comm is None: comm = comm_world
    if lpyMPIactive:
        result, status = comm.recv(source, tag)
    elif lmpi4pyactive and lmpi4pynp == False:
        result = comm.recv(source = source, tag = tag)
    else:
        result = comm.Recv(data, source = source, tag = tag)
    return result

def mpisend(data = None, dest = 0, tag = 0, comm = None, lmpi4pynp = False):
    if comm is None: comm = comm_world
    if lpyMPIactive:
        result = comm.send(data, dest, tag)
    elif lmpi4pyactive and lmpi4pynp == False:
        result = comm.send(data, dest = dest, tag = tag)
    else:
        result = comm.Send(data, dest = dest, tag = tag)
    return result

def mpibcast(data = None, root = 0, comm = None, lmpi4pynp = False):
    if comm is None: comm = comm_world
    if lpyMPIactive:
        result = comm.bcast(data, root)
    elif lmpi4pyactive and lmpi4pynp == False:
        result = comm.bcast(data, root = root)
    else: 
        result = comm.Bcast(data, root = root)
    return result

def mpiallreduce(data = None, op = mpi.SUM, comm = None, lmpi4pynp = False):
    if comm is None: comm = comm_world
    if lpyMPIactive:
        result = comm.allreduce(data, op)
    elif lmpi4pyactive and lmpi4pynp == False:
        result = comm.allreduce(data, op = op)
    else:
        result = comm.Allreduce(data, op = op)
    return result

# ---------------------------------------------------------------------------
# --- By default, set so that only PE0 sends output to the terminal.
# --- mpi4py version supports output to me = 0, but not "synchronized"
if lparallel and lpyMPIactive:
    mpi.synchronizeQueuedOutput('/dev/null')
if lparallel and lmpi4pyactive:
    synchronizeQueuedOutput_mpi4py(out = True, error = False)

def number_of_PE(comm=None):
    if not lparallel: return 1
    if comm is None: comm = comm_world
    if lpyMPIactive:
        return comm.procs
    else:
        return comm.Get_size()

def get_rank(comm=None):
    if not lparallel: return 0
    if comm is None: comm = comm_world
    if lpyMPIactive:
        return comm.rank
    else:
        return comm.Get_rank()

# ---------------------------------------------------------------------------
# Enable output from all processors
def EnableAll():
    if not lparallel: return
    if lpyMPIactive:
        mpi.synchronizeQueuedOutput(None)
    if lmpi4pyactive:
        synchronizeQueuedOutput_mpi4py(out = False, error = False)

# ---------------------------------------------------------------------------
# Disable output from all but processor 0
def DisableAll():
    if not lparallel: return
    if lpyMPIactive:
        mpi.synchronizeQueuedOutput('/dev/null')
    if lmpi4pyactive:
        synchronizeQueuedOutput_mpi4py(out = True, error = False)

# ---------------------------------------------------------------------------
# Print object on all processors
def pprint(obj):
    if not (lparallel and lpyMPIactive):
        print str(obj)
        return
    # Ignore all exceptions to make sure that there is not a lock up.
    try:
        ss = str(obj)
    except:
        ss = ''
    mpi.synchronizedWrite(ss+'\n')
    if mpi.rank == 0: print

# ---------------------------------------------------------------------------
# Print array (or list) from all processors
def aprint(obj):
    if not (lparallel and lpyMPIactive):
        print str(obj)
        return
    mpi.synchronizedWrite(str(obj))

# ---------------------------------------------------------------------------
# Get address of processor
def self_address(comm=None):
    if not lparallel: return 0
    if comm is None: comm = comm_world
    if lpyMPIactive:
        return comm.rank
    else:
        return comm.Get_rank()

# ---------------------------------------------------------------------------
# Copy an array from processor i to processor 0
def getarray(src,v,dest=0,comm=None):
    if not lparallel: return v
    if comm is None: comm = comm_world
    if get_rank(comm = comm) == src:
        mpisend(data = v, dest = dest, comm = comm)
    elif get_rank(comm = comm) == dest:
        return mpirecv(data = v, source = src, comm=comm)
    return v

# ---------------------------------------------------------------------------
# Gather an object from all processors in a communicator (default comm_world) into a list
def gather(obj,dest=0,comm=None):
    if not lparallel: return [obj]
    if comm is None: comm = comm_world
    if get_rank(comm = comm) == dest:
        result = []
        for i in range(number_of_PE(comm = comm)):
            if i == dest:
                result.append(obj)
            else:
                result.append(mpirecv(data = obj, source = i , comm=comm)) 
        return result
    else:
        mpisend(data = obj, dest = dest, comm = comm)
        return [obj]

# ---------------------------------------------------------------------------
# Gather an object from a list of processors into a list on destination processor,
# eventually broadcasting result.
def gatherlist(obj,dest=0,procs=None,bcast=0,comm=None):
    if not lparallel: return [obj]
    if comm is None: comm = comm_world
    if procs is None:
        procs=range(number_of_PE(comm = comm))
    else:
        procs=list(procs)
    result = []
    if get_rank(comm = comm) == dest:
        for i in procs:
            if i == dest:
                result.append(obj)
            else:
                result.append(mpirecv(data = obj, source = i , comm=comm))
    else:
        if get_rank(comm = comm) in procs:
            mpisend(data = obj, dest = dest, comm = comm)

    if bcast:
        result = mpibcast(data = result, root = dest, comm = comm)

    return result


# ---------------------------------------------------------------------------
# Gather an object from a list of processors into a list on destination processor,
# eventually broadcasting result. The send is decomposed into log2(n) sends
# between processors.
def gatherlog(obj,dest=0,procs=None,bcast=0,comm=None):
    if not lparallel: return [obj]
    if comm is None: comm = comm_world
    if procs is None:
        procs=range(number_of_PE(comm = comm))
    else:
        procs=list(procs)
    n = int(ceil(log2(len(procs))))
    obj = [obj]
    for i in range(n):
        st = 2**i
        stp = 2**(i+1)
        listsend = procs[st::stp]
        listrecv = procs[::stp][:len(listsend)]
        if get_rank(comm = comm) in listsend:
            ip = listsend.index(get_rank(comm = comm))
            mpisend(data = obj, dest = listrecv[ip], comm = comm)
        if get_rank(comm = comm) in listrecv:
            ip = listrecv.index(get_rank(comm = comm))
            obj+=mpirecv(data = obj, source = listsend[ip], comm=comm)

    if bcast:
        obj = mpibcast(data = obj, root = procs[0], comm = comm)
    else:
        if dest != procs[0]:
            if get_rank(comm = comm)==procs[0]:
                mpisend(data = obj, dest = dest, comm = comm)
            if get_rank(comm = comm)==dest:
                obj=mpirecv(data = obj, source = procs[0], comm=comm)

    return obj

# ---------------------------------------------------------------------------
# Define a barrier
def barrier(comm=None):
    if not lparallel: return
    if comm is None: comm = comm_world
    if lpyMPIactive:
        comm.barrier()
    else:
        comm.Barrier()

# ---------------------------------------------------------------------------
# Broadcast an object to all processors
def broadcast(obj,root=0,comm=None):
    if not lparallel: return obj
    if comm is None: comm = comm_world
    return mpibcast(data = obj, root = root, comm = comm)

# ---------------------------------------------------------------------------
# Gather an object from all processors into a list and scatter back to all
def gatherall(obj,comm=None):
    if not lparallel: return obj
    if comm is None: comm = comm_world
    obj = gather(obj, comm=comm)
    return mpibcast(data = obj, comm = comm)

# ---------------------------------------------------------------------------
# General gatherarray which returns an array object combining the
# first dimension.
def gatherarray(a,root=0,othersempty=0,bcast=0,comm=None):
    if not lparallel: return a
    if comm is None: comm = comm_world
    # --- First check if input can be converted to an array
    isinputok = 1
    try:
        if type(a) in [type(0.),type(0)]:
            a = array([a])
        else:
            a = array(a,copy=False)
    except:
        isinputok = 0
    # --- Make sure the input is ok on all of the processors
    isinputok = globalmin(isinputok,comm=comm)
    # --- If any returned an error, then all exit (to avoid a deadlock)
    if not isinputok:
        print "Object could not be converted to an array"
        return None
    # --- Now, actually gather the array.
    # --- The check of whether the result is ok may not be needed.
    try:
        result = gather(a,root,comm=comm)
        isinputok = 1
    except:
        isinputok = 0
    # --- Make sure again that the input is ok on all of the processors
    isinputok = globalmin(isinputok,comm=comm)
    if not isinputok:
        print "Error in gather object"
        try:
            "Object has shape ",shape(a)
        except NameError:
            pass
        return None
    # --- All processors but root simply return either the input argument
    # --- or an empty array unless the result is to be broadcast
    if get_rank(comm = comm) != root and not bcast:
        if othersempty: return zeros(len(shape(a))*[0],a.dtype.char)
        else: return a
    # --- Root processor reshapes the data, removing the first dimension
    # --- Do it bit by bit since the data passed by the other processors may
    # --- not be all the same size.
    if get_rank(comm = comm) == root:
        newlen = 0
        for i in range(number_of_PE(comm = comm)):
            newlen = newlen + shape(result[i])[0]
        newshape = list(shape(result[0]))
        newshape[0] = newlen
        newresult = zeros(newshape,a.dtype.char)
        i1 = 0
        for i in range(number_of_PE(comm = comm)):
            i2 = i1 + shape(result[i])[0]
            newresult[i1:i2,...] = result[i]
            i1 = i2
    else:
        newresult = 0
    if bcast: newresult = mpibcast(data = newresult, root = root, comm = comm)
    return newresult
    ## --- Old way
    ## --- Its easy if all of the arrays passed from the other processors
    ## --- are the same size.
    #result = array(result)
    #ss = list(shape(result))
    #snew = ss[1:]
    #snew[0] = ss[0]*ss[1]
    #result.shape = snew
    ## --- Return the result
    #return result


# ---------------------------------------------------------------------------
# Find the nonzero value of array over all processors. This assumes that the
# non-zero values for each index are the same for all processors.
# Resulting data is broadcast to all processors.
def parallelnonzeroarray(a,comm=None):
    if comm is None: comm = comm_world
    dmax = parallelmax(a,comm=comm)
    dmin = parallelmin(a,comm=comm)
    result = where(not_equal(dmax,0),dmax,dmin)
    return result

# ---------------------------------------------------------------------------
# Generic global operation on a distributed array.
def globalop(a,localop,mpiop,defaultval,comm=None):
    if comm is None: comm = comm_world
    if len(shape(a)) == 0:
        local = a
    elif len(a) > 0:
        try:
            # --- Protect application of localop since the data may not be
            # --- appropriate on all processors, for example it may be an empty array.
            local = localop(a)
        except:
            local = defaultval
    else:
        local = defaultval
    if not lparallel: return local
    return mpiallreduce(local, op = getattr(mpi,mpiop), comm = comm)

# ---------------------------------------------------------------------------
# Specific operations on a distributed array.
def globalmax(a,comm=None):
    def _max(a): return array(a,copy=False).max()
    if comm is None: comm = comm_world
    return globalop(a,_max,"MAX",-1.e36,comm=comm)
def globalmin(a,comm=None):
    def _min(a): return array(a,copy=False).min()
    if comm is None: comm = comm_world
    return globalop(a,_min,"MIN",+1.e36,comm=comm)
def globalsum(a,comm=None):
    def _sum(a): return array(a,copy=False).sum()
    if comm is None: comm = comm_world
    return globalop(a,_sum,"SUM",0.,comm=comm)
def globalave(a,comm=None):
    def _sum(a): return array(a,copy=False).sum()
    if comm is None: comm = comm_world
    s = globalop(a,_sum,"SUM",0.,comm=comm)
    if len(shape(a)) == 0: a = [a]
    n = globalsum(len(a),comm=comm)
    if n > 0: return s/n
    else:     return 0.

# ---------------------------------------------------------------------------
# Generic parallel element-by-element operation on a distributed array.
# --- Note that this is no long needed
def parallelop(a,mpiop,comm=None):
    if not lparallel: return a
    if comm is None: comm = comm_world
    if type(a) == type(array([])):
        a1d = ravel(a) + 0
        for i in range(len(a1d)):
            a1d[i] = mpiallreduce(a1d[i], op = getattr(mpi,mpiop), comm = comm)
        a1d.shape = shape(a)
        return a1d
    else:
        return mpiallreduce(a, op = getattr(mpi,mpiop), comm = comm)

# ---------------------------------------------------------------------------
# Specific parallel element-by-element operations on a distributed array.
def parallelmax(a,comm=None):
    #return parallelop(a,"MAX")
    if not lparallel: return a
    if comm is None: comm = comm_world
    if lpyMPIactive:
        return mpiallreduce(a, op = maximum, comm = comm)
    else:
        return mpiallreduce(a, op = mpi.MAX, comm = comm)

def parallelmin(a,comm=None):
    #return parallelop(a,"MIN")
    if not lparallel: return a
    if comm is None: comm = comm_world
    if lpyMPIactive:
        return mpiallreduce(a, op = minimum, comm = comm)
    else:
        return mpiallreduce(a, op = mpi.MIN, comm = comm)
def parallelsum(a,comm=None):
    if not lparallel: return a
    if comm is None: comm = comm_world
    return mpiallreduce(a, op = mpi.SUM, comm = comm)