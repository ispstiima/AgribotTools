FROM ubuntu:noble-20241118.1

# Install base utilities
RUN export DEBIAN_FRONTEND=noninteractive \
 && apt update \
 && apt install -y -qq --no-install-recommends \
        bash-completion sudo nano less parallel \
        curl wget ssh gnupg2 ca-certificates \
        htop btop bc xxd uuid-runtime net-tools \
        tar unzip xz-utils p7zip-full zip \
        software-properties-common lsb-release \
        git git-lfs locales \
        python3 \
        python3-pip \
        python3-venv \
        libgl1 \
 && rm -rf /var/lib/apt/archives /var/lib/apt/lists

# Setup 'ubuntu' user settings
RUN echo 'ubuntu:pass' | chpasswd \ 
 && echo 'alias ll="ls -al"' >> /home/ubuntu/.bashrc \
 && echo 'export GIT_PAGER=less' >> /home/ubuntu/.bashrc \
 && echo DONE
 
# Install git-aware-prompt
RUN wget -qO- 'https://github.com/jimeh/git-aware-prompt/archive/df8f218d4af51b1fac03f58bf0ddc776466312de.tar.gz' | tar xvz -C /opt \
 && mv /opt/git-aware-prompt-* /opt/git-aware-prompt \
 && echo 'export GITAWAREPROMPT=/opt/git-aware-prompt' >> /home/ubuntu/.bashrc \
 && echo 'source "${GITAWAREPROMPT}/main.sh"' >> /home/ubuntu/.bashrc \
 && echo 'export PS1="\${debian_chroot:+(\$debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\] \[$txtcyn\]\$git_branch\[$txtred\]\$git_dirty\[$txtrst\]\$ "' >> /home/ubuntu/.bashrc
 
 # Set up the LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT environment variable
 RUN echo 'export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/ubuntu/labelstudio' >> /home/ubuntu/.bashrc