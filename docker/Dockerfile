# syntax=docker/dockerfile:1

# Base image as an argument
ARG DOCKER_BASE
FROM ${DOCKER_BASE}

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Update and install necessary packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    git wget

# [OPTIONAL] Install TeX Live packages for plotting
RUN apt-get clean && apt-get update && apt-get install -y \
    texlive-latex-base \
    texlive-fonts-recommended \
    texlive-latex-extra \
    dvipng \
    cm-super

# Create a non-root user with the same UID/GID as the host user
ARG USER_ID
ARG GROUP_ID
ARG USER
RUN addgroup --gid ${GROUP_ID} ${USER}
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER
RUN usermod -d /home/${USER} ${USER}
USER ${USER}
WORKDIR /home/${USER}

# Install Miniconda as the non-root user
RUN wget --quiet \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/${USER}/miniconda3 \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Add Miniconda to PATH
ENV PATH="/home/${USER}/miniconda3/bin:${PATH}"

# Dependencies
RUN echo new_change
COPY --chown=${USER}:${USER} docker/environment.yaml /home/${USER}/
RUN . /home/${USER}/miniconda3/etc/profile.d/conda.sh \
    && conda update conda \
    && conda install python=3.10 \
    && conda install pytorch=2.1.0 torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia \
    && conda env update -n base --file /home/${USER}/environment.yaml

# Append Miniconda to PATH in .bashrc for interactive shells
RUN echo "export PATH=/home/${USER}/miniconda3/bin:$PATH" >> /home/${USER}/.bashrc

# Clear apt install cache and switch back to dialog for any ad-hoc use of apt-get
USER root
RUN rm -rf /var/lib/apt/lists/* && apt-get clean
ENV DEBIAN_FRONTEND=dialog
USER ${USER}

# The CMD command can be used to run a specific command within the container
CMD ["bash"]
