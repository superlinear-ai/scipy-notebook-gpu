FROM jupyter/scipy-notebook:latest

USER root

# TODO(lsorber): Reevaluate this when TensorFlow hits 1.13.

# Add NVIDIA repositories to apt-get [1].
#
# [1] https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/base/Dockerfile
RUN apt-get update && apt-get install -y gnupg && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb && \
    apt install ./cuda-repo-ubuntu1604_9.1.85-1_amd64.deb && \
    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb && \
    apt install ./nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
ENV CUDA_VERSION 9.0.176
ENV CUDA_PKG_VERSION 9-0=$CUDA_VERSION-1
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION && \
    ln -s cuda-9.0 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=9.0"

# Install TensorFlow dependencies [1], [2].
#
# [1] https://www.tensorflow.org/install/gpu
# [2] https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/tools/dockerfiles/dockerfiles/gpu.Dockerfile
RUN apt-get update && apt-get install -y \
        cuda9.0 \
        cuda-cublas-9-0 \
        cuda-cufft-9-0 \
        cuda-curand-9-0 \
        cuda-cusolver-9-0 \
        cuda-cusparse-9-0 \
        libcudnn7=7.2.1.38-1+cuda9.0 \
        libnccl2=2.2.13-1+cuda9.0 \
        cuda-command-line-tools-9-0
RUN apt-get update && apt-get install -y \
        nvinfer-runtime-trt-repo-ubuntu1604-4.0.1-ga-cuda9.0 && \
    apt-get update && apt-get install -y \
        libnvinfer4=4.1.2-1+cuda9.0

USER $NB_USER
