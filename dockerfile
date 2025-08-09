FROM ubuntu:22.04

# 安装系统依赖和 Python
RUN apt-get update && apt-get install -y \
    build-essential cmake git wget curl \
    python3 python3-pip python3-venv \
    libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# 创建 virtualenv
RUN pip install --upgrade pip && \
    pip install virtualenv==20.30.0
RUN virtualenv /venv
ENV PATH="/venv/bin:$PATH"

# 安装 Python 依赖
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade setuptools wheel && \
    pip install -r /tmp/requirements.txt

# clone and compile acados
WORKDIR /opt
RUN git clone https://github.com/acados/acados.git && \
    cd acados && \
    git submodule update --recursive --init && \
    mkdir -p build && cd build && \
    cmake -DACADOS_WITH_QPOASES=ON -DACADOS_WITH_OPENMP=ON .. && \
    make install

# install acados Python interface
RUN pip install -e /opt/acados/interfaces/acados_template

# set acados environment variables
ENV LD_LIBRARY_PATH="/opt/acados/lib:${LD_LIBRARY_PATH}"
ENV ACADOS_SOURCE_DIR="/opt/acados"

# 添加 t_renderer
COPY t_renderer /opt/acados/bin/t_renderer
RUN chmod +x /opt/acados/bin/t_renderer

# 添加入口脚本
WORKDIR /app
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
