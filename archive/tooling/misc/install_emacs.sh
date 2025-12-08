# SUDO=
SUDO=sudo

$SUDO apt-get update && $SUDO apt-get install -y emacs

# $SUDO apt-get remove emacs24 emacs24-nox

# $SUDO add-apt-repository ppa:kelleyk/emacs
# $SUDO apt-get update && $SUDO apt-get install -y emacs26-nox

cd $HOME && rm -rf .emacs.d && git clone https://github.com/deven-amd/.emacs.d.git

# EMACS_TARBALL=emacs-26.1.tar.gz
# EMACS_DOWNLOAD_URL=https://ftp.gnu.org/gnu/emacs/$EMACS_TARBALL
# EMACS_DOWNLOAD_DIR=$HOME/temp_emacs

# mkdir $EMACS_DOWNLOAD_DIR
# cd $EMACS_DOWNLOAD_DIR && wget $EMACS_DOWNLOAD_URL && tar -zxvf $EMACS_TARBALL
# cd $EMACS_DOWNLOAD_DIR && ./configure --with-xpm=no --with-gif=no --with-gnutls=no && ./make
