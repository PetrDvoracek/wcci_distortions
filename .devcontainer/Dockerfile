FROM nvcr.io/nvidia/pytorch:24.01-py3

# TODO change
ARG USERNAME=pedro
# TODO change
ARG USER_EMAIL=pedro.dvoracek@gmail.com
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC


COPY requirements.txt /tmp/
RUN apt-get update  \
    && pip install --upgrade pip \
    # remove all opencv, problem with its import https://github.com/opencv/opencv-python/issues/884
    && pip install opencv-fixer==0.2.5 \
    && python -c "from opencv_fixer import AutoFix; AutoFix()" \
    # && pip uninstall $(pip list --format=freeze | grep opencv) -y \ 
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    # mmdetection 
    # && python -m mim install "mmengine==0.10.3" "mmcv-full==2.1.0" "mmdet==3.3.0" \
    && python -m mim install "mmengine==0.10.3" "mmcv-full" "mmdet==3.3.0" \
    # create non-root user
    && groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    # libgl1-mesa-glx
    && export DEBIAN_FRONTEND=noninteractive \
    && export TZ=Etc/UTC \
    && apt-get install -y sudo tmux x11-apps tree libgl1 \
    # install latex for scienceplots
    dvipng texlive-latex-extra texlive-fonts-recommended cm-super \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    # git configure
    && git config --global user.email "${USER_EMAIL}" \
    # git use main instead master branch name
    && git config --global init.defaultBranch main \
    # git avoid dubious ownership
    && git config --global --add safe.directory /workspaces/${WORKDIR_NAME}

# so zsh is installed for non-root user
USER $USERNAME
# install zsh
COPY zsh-in-docker.sh /tmp/zsh-in-docker.sh
RUN sh /tmp/zsh-in-docker.sh \
    -t amuse \
    -p git \
    -p extract \
    -p copybuffer

# # make zsh history persistent after rebuild
# RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.zsh_history" \
#     && touch /.zsh_history \
#     && chown -R $USERNAME /.zsh_history \
#     && echo "$SNIPPET" >> "/home/$USERNAME/.zshrc"
