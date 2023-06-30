# Adapted from: https://github.com/ufoym/deepo

FROM nvidia/cudagl:11.1-devel-ubuntu20.04

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

ENV LANG C.UTF-8

ENV OS=ubuntu2004
ARG cuda_version=cuda11.1
ARG cudnn_version=8.0.5.39
ARG CU_VERSION=cu111
ARG CMAKE_VERSION=3.21.1
ARG TORCH_VERSION=1.9.0
ARG TORCHVISION_VERSION=0.10.0

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --recursive" && \

    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \

    apt-get update && \

# ==================================================================
# tools
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        software-properties-common \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        tree \
        unzip \
        unrar \
        && \

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        libglu1-mesa \
        libglib2.0-0 \
        libboost-all-dev \
        && \

# ==================================================================
# cudnn
# ------------------------------------------------------------------
    wget https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin && \
    mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/7fa2af80.pub && \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/ /" && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        libcudnn8=${cudnn_version}-1+${cuda_version} \
        libcudnn8-dev=${cudnn_version}-1+${cuda_version} \
    && \

# ==================================================================
# cmake
# ------------------------------------------------------------------

    wget -O ~/cmake.sh \
        https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    bash ~/cmake.sh --skip-license --prefix=/usr --exclude-subdir && \

# ==================================================================
# nodejs
# ------------------------------------------------------------------

    curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \
        DEBIAN_FRONTEND=noninteractive $APT_INSTALL nodejs && \

# ==================================================================
# python
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.8 \
        python3.8-dev \
        && \
    ln -s /usr/bin/python3.8 /usr/local/bin/python && \
    ln -s /usr/bin/python3.8 /usr/local/bin/python3 && \
    wget -O ~/get-pip.py https://bootstrap.pypa.io/get-pip.py && \
    python ~/get-pip.py && \

# ==================================================================
# essential libraries
# ------------------------------------------------------------------

    $PIP_INSTALL \
        setuptools==66.0.0 \
        future==0.18.3 \
        typing==3.7.4.3 \
        Cython==0.29.33 \
        numba==0.56.4 \
        h5py==3.7.0 \
        protobuf==3.20.1 \
        enum34==1.1.10 \
        pyyaml==6.0 \
        tqdm==4.64.1 \

        numpy==1.23.5 \
        scipy==1.9.3 \
        scikit-image==0.19.3 \
        scikit-learn==1.2.0 \
        pillow==9.4.0 \
        opencv-python==4.7.0.68 \
        pandas==1.4.4 \
        seaborn==0.12.2 \
        matplotlib==3.6.3 \
        ipdb==0.13.11 \ 
        IPython==8.8.0 \


        jupyter==1.0.0 \
        jupyterlab==3.5.2 \
        && \

# ==================================================================
# ML/CV libraries
# ------------------------------------------------------------------

    pip install git+https://github.com/argoai/argoverse-api.git && \
    pip install dgl-cu111 -f https://data.dgl.ai/wheels/repo.html && \
    pip install av2==0.2.1 && \
    pip install mpi4py==3.1.4 && \

    $PIP_INSTALL \
        torch==${TORCH_VERSION}+${CU_VERSION} \
        torchvision==${TORCHVISION_VERSION}+${CU_VERSION} \
        -f https://download.pytorch.org/whl/torch_stable.html \
        && \
    $PIP_INSTALL \
        skorch==0.12.1 \
        pytorch-lightning==1.8.6 \
        captum==0.6.0 \
        && \

    HOROVOD_GPU_OPERATIONS=NCCL pip install horovod==0.19.4 && \

    pip install lanelet2 && \

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*