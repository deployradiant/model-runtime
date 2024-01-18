###############################################
# Base Image
###############################################
FROM nvcr.io/nvidia/pytorch:23.03-py3 as python-base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.1.4  \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv" 

# prepend poetry and venv to path
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"
ENV POETRY_VERSION=1.5.1
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=${POETRY_HOME} python3 - --version ${POETRY_VERSION} && \
    chmod a+x /opt/poetry/bin/poetry


###############################################
# Builder Image
###############################################
ENV DEBIAN_FRONTEND=noninteractive
FROM python-base as builder-base
RUN apt-get update
# RUN apt-get install gcc -y
# RUN apt-get install build-essential -y

RUN apt install software-properties-common -y
RUN { echo;echo; } | add-apt-repository ppa:deadsnakes/ppa
RUN apt install python3.10 -y
RUN apt install python3.10-distutils -y
RUN apt-get -y install intel-mkl
ENV LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH

WORKDIR $PYSETUP_PATH
COPY libs/* ./libs/
COPY poetry.lock pyproject.toml ./
# install runtime deps - uses $POETRY_VIRTUALENVS_IN_PROJECT internally
RUN poetry install --no-dev

###############################################
# Production Image
###############################################
FROM builder-base as production

# Create and run container as radiant user
ARG USERNAME=radiant
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && chown -R $USER_UID:$USER_GID /home/$USERNAME

USER $USERNAME

COPY --from=builder-base $PYSETUP_PATH $PYSETUP_PATH
WORKDIR /app
COPY . .
# Copying in our entrypoint
EXPOSE 8000
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port=8000"]
