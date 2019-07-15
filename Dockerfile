FROM jupyter/scipy-notebook:latest

USER root

# Add NVIDIA repositories to apt-get [1].
#
# [1] https://gitlab.com/nvidia/cuda/blob/ubuntu18.04/10.0/base/Dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --autoremove -y curl && \
    rm -rf /var/lib/apt/lists/*
ENV CUDA_VERSION 10.0.130
ENV CUDA_PKG_VERSION 10-0=$CUDA_VERSION-1
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION \
        cuda-compat-10-0 && \
    ln -s cuda-10.0 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.0 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=410,driver<411"

# Install TensorFlow dependencies [1], [2].
#
# [1] https://www.tensorflow.org/install/gpu#ubuntu_1804_cuda_10
# [2] https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/tools/dockerfiles/dockerfiles/gpu.Dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-10-0 \
        libcudnn7=7.6.0.64-1+cuda10.0 \
        libcudnn7-dev=7.6.0.64-1+cuda10.0
RUN apt-get update && apt-get install -y --no-install-recommends \
        libnvinfer5=5.1.5-1+cuda10.0 \
        libnvinfer-dev=5.1.5-1+cuda10.0

USER $NB_USER
