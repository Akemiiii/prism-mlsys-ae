FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

ARG BUILD_TYPE=all

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /prism
COPY . /prism

RUN cd sglang && python3 -m pip install --upgrade pip "setuptools<81" wheel && \
    python3 -m pip install -i https://mirrors.aliyun.com/pypi/simple \
    -e "python[${BUILD_TYPE}]" \
    --find-links ./wheels

RUN python3 -m pip install -i https://mirrors.aliyun.com/pypi/simple matplotlib

RUN python3 -m venv /prism/sglang054 && \
    . /prism/sglang054/bin/activate && \
    pip install -i https://mirrors.aliyun.com/pypi/simple sglang==0.5.4 && \
    deactivate

