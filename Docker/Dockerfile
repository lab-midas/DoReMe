#FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel

ARG http_proxy
ARG https_proxy
ARG no_proxy
ENV http_proxy="$http_proxy"
ENV https_proxy="$https_proxy"
ENV no_proxy="$no_proxy"
#RUN echo $http_proxy


WORKDIR /root
USER DoReMe_tester
WORKDIR /home/DoReMe_tester

COPY Docker/requirements.txt ./
RUN pip install --break-system-packages --user -r requirements.txt

ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;10.0"

COPY bspline_gradient/ ./
RUN pip install --break-system-packages --user --no-cache-dir --no-build-isolation .

WORKDIR /DoReMe_tester