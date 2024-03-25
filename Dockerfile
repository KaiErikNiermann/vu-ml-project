FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

ARG CODE_DIR=/app
WORKDIR $CODE_DIR

# Avoid hang on tzdata install.
# https://serverfault.com/a/992421
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Build dependencies for pyenv.
# https://github.com/pyenv/pyenv/wiki#suggested-build-environment
RUN apt-get update && apt-get install -y \
    git curl build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl \
    libncursesw5-dev xz-utils tk-dev libxml2-dev \
    libxmlsec1-dev libffi-dev liblzma-dev

# Install Python 3.11.
RUN curl https://pyenv.run | bash
ARG PYTHON_VERSION=3.11
ARG PYENV_ROOT=/root/.pyenv
ENV PATH=$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH
RUN pyenv install $PYTHON_VERSION
RUN pyenv global $PYTHON_VERSION

# Install Poetry.
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH=$PATH:/root/.local/bin
RUN poetry config virtualenvs.create false

# Install torch separately, to avoid issues with Poetry.
# https://github.com/python-poetry/poetry/issues/6409
RUN pip install install torch==2.0.1 lightning \
    --extra-index-url https://download.pytorch.org/whl/cu116

ADD ./pyproject.toml ./poetry.lock $CODE_DIR/
RUN poetry install && rm -rf ~/.cache

ADD . $CODE_DIR
RUN poetry install && rm -rf ~/.cache
RUN pyenv rehash
RUN export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/root/.pyenv/versions/3.11.6/lib/python3.11/site-packages/nvidia/cusparse/lib/:/root/.pyenv/versions/3.11.6/lib/python3.11/site-packages/nvidia/cudnn/lib/:/root/.pyenv/versions/3.11.6/lib/python3.11/site-packages/nvidia/cublas/lib/:/root/.pyenv/versions/3.11.6/lib/python3.11/site-packages/nvidia/cufft/lib/"
