FROM ubuntu:14.04

# Install a few packages, as root
RUN apt-get update \
    && apt-get install -y \
    wget \
    gfortran \
    make \
    git \
    libx11-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a new user and copy the current branch of Warp
# into the Docker container 
RUN useradd --create-home warp_user
RUN mkdir /home/warp_user/warp/
COPY ./* /home/warp_user/warp/
RUN chown -R warp_user /home/warp_user/warp/
# Grant sudo access without password
RUN echo 'warp_user ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers

# Switch to the new user
WORKDIR /home/warp_user
USER warp_user

# Install miniconda
RUN cd /home/warp_user \
    && wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b \
    && rm miniconda.sh 
ENV PATH /home/warp_user/miniconda2/bin:$PATH

# Install python dependencies for warp
RUN conda update conda \
    && conda install --yes \
    numpy \
    scipy \
    matplotlib \
    h5py \
    mpi4py \
    && conda clean --tarballs

# Install Forthon
RUN pip install --upgrade pip \
    && pip install Forthon

# Install warp 
RUN git clone https://bitbucket.org/berkeleylab/warp.git \
    && cd warp/pywarp90 \
    && echo 'FCOMP= -F gfortran --fcompex mpif90' >> Makefile.local.pympi \
    && echo "if parallel:" >> setup.local.py \
    && echo "   library_dirs += ['~/miniconda2/lib']" >> setup.local.py \
    && echo "   libraries = fcompiler.libs + ['mpichf90', 'mpich', 'opa', 'mpl']" >> setup.local.py \
    && make install \
    && make pinstall

# Install pygist
RUN git clone https://bitbucket.org/dpgrote/pygist.git \
    && cd pygist \
    && python setup.py config \
    && python setup.py build \
    && python setup.py install

# Prepare the run directory
WORKDIR /home/warp_user/warp/
