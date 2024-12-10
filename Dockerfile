FROM mambaorg/micromamba:1.5.8-lunar

COPY conda.yml /tmp/conda.yml

ENV PATH="$MAMBA_ROOT_PREFIX/bin:$PATH" \
    MPLCONFIGDIR=/tmp/mpl_cache \
    HOME=/tmp

RUN echo "Installing dependencies with micromamba..." && \
    micromamba install -y -n base -f /tmp/conda.yml && \
    micromamba install -y -n base conda-forge::procps-ng && \
    micromamba clean -a -y
