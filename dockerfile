FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y git build-essential cmake python3-dev wget bzip2 pkg-config \
    libblas-dev liblapack-dev && \
    rm -rf /var/lib/apt/lists/*


RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml

SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# clone and compile acados
WORKDIR /opt
RUN git clone https://github.com/acados/acados.git && \
    cd acados && \
    git submodule update --recursive --init && \
    mkdir -p build && cd build && \
    cmake -DACADOS_WITH_QPOASES=ON -DACADOS_WITH_OPENMP=ON .. && \
    make install -j4

# install acados Python interface
RUN pip install -e /opt/acados/interfaces/acados_template

# set acados environment variables
ENV LD_LIBRARY_PATH="/opt/acados/lib:${LD_LIBRARY_PATH}"
ENV ACADOS_SOURCE_DIR="/opt/acados"

COPY t_renderer /opt/acados/bin/t_renderer
RUN chmod +x /opt/acados/bin/t_renderer

WORKDIR /app

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
CMD ["/app/entrypoint.sh"]