FROM busybox AS model

# Create a folder where to store the BEATs checkpoints
ADD "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D" /model/BEATs_iter3_plus_AS2M.pt

FROM busybox AS data

# Create a folder where to store the data
RUN mkdir /data && \
    cd /data && \
    wget -qO - "https://github.com/karoldvl/ESC-50/archive/master.tar.gz" | \
    tar xzf -

FROM python:3.8

# Install the system software
ARG PACKAGES="ffmpeg build-essential"

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -yqq --no-install-recommends $PACKAGES

# Install the python dependancies
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install poetry~=1.3.2

# Create the working directory with the necessary inside
WORKDIR /app
COPY ./pyproject.toml ./poetry.lock ./
RUN --mount=type=cache,target=/root/.cache/pypoetry \
    poetry config virtualenvs.create false && \
    poetry install --no-root
COPY . ./

COPY --from=data /data /data
COPY --from=model /model /model

# Update the pythonpath
ENV PYTHONPATH "${PYTHONPATH}:/app"

