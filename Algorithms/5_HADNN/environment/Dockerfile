# hash:sha256:414e24e5ea7c1b1da5dea598869954de0bb09df8604b4c139278ea4f3bace367
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        conda==4.11.0 \
    && conda clean -ya
RUN conda install -y \
        pandas==1.3.5 \
        scikit-learn==1.0.2 \
        tensorflow==2.2.0 \
    && conda clean -ya
