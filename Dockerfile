FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ARG BUILD_TYPE=all

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN if [ -f /etc/apt/sources.list ]; then sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list; fi \
    && if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list.d/ubuntu.sources; fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends git curl libnuma1 ca-certificates bzip2 build-essential cmake pkg-config \
    && curl -fsSL https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm -f /tmp/miniconda.sh \
    && (/opt/conda/bin/conda config --remove channels defaults || true) \
    && /opt/conda/bin/conda config --add channels conda-forge \
    && /opt/conda/bin/conda config --set channel_priority strict \
    && /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && /opt/conda/bin/conda create -y -n prism-sglang python=3.12 pip \
    && /opt/conda/bin/conda create -y -n prism python=3.12 pip \
    && /opt/conda/bin/conda create -y -n eagle3 python=3.12 pip \
    && /opt/conda/bin/conda clean -afy \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/conda/bin:${PATH}"
ENV PRISM_PYTHON=/opt/conda/envs/prism/bin/python \
    PRISM_PIP=/opt/conda/envs/prism/bin/pip \
    SGLANG_PYTHON=/opt/conda/envs/prism-sglang/bin/python \
    SGLANG_PIP=/opt/conda/envs/prism-sglang/bin/pip \
    EAGLE3_PYTHON=/opt/conda/envs/eagle3/bin/python \
    EAGLE3_PIP=/opt/conda/envs/eagle3/bin/pip

RUN echo 'source /opt/conda/etc/profile.d/conda.sh' >> /root/.bashrc \
    && echo 'conda activate prism' >> /root/.bashrc


WORKDIR /prism
COPY . /prism

RUN ${SGLANG_PIP} install -i https://mirrors.cloud.tencent.com/pypi/simple/  -r /prism/sglang/requirements.txt && \
    ${SGLANG_PIP} install -i https://mirrors.cloud.tencent.com/pypi/simple/  matplotlib modelscope "setuptools<82.0.0"

RUN cd sglang && ${SGLANG_PIP} install -i https://mirrors.cloud.tencent.com/pypi/simple/  \
    -e "python[${BUILD_TYPE}]"

RUN ${SGLANG_PYTHON} -m venv /prism/sglang054 && \
    /prism/sglang054/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  -r /prism/sglang/requirements_054.txt && \
    /prism/sglang054/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  sglang==0.5.4

RUN ${PRISM_PIP} install -i https://mirrors.cloud.tencent.com/pypi/simple/  -r /prism/PRISM/requirements.txt && \
    ${PRISM_PIP} install -i https://mirrors.cloud.tencent.com/pypi/simple/  matplotlib modelscope "setuptools<82.0.0"

RUN ${EAGLE3_PIP} install -i https://mirrors.cloud.tencent.com/pypi/simple/  -r /prism/Eagle3/requirements_eagle3.txt && \
    ${EAGLE3_PIP} install -i https://mirrors.cloud.tencent.com/pypi/simple/  matplotlib modelscope "setuptools<82.0.0"



CMD ["/bin/bash"]

