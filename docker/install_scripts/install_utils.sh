WORKDIR=$1

cd $WORKDIR

apt-get update

apt-get install -y --no-install-recommends \
	apt-transport-https \
	autoconf \
	automake \
	build-essential \
	curl \
	emacs24-nox \
	ffmpeg \
	git \
	libcurl4-openssl-dev \
	libtool \
	libssl-dev \
	lsb-release \
	mlocate \
	openjdk-8-jdk \
	openjdk-8-jre-headless \
	pkg-config \
	rsync \
	software-properties-common \
	sudo \
	swig \
	unzip \
	vim \
	wget \
	xz-utils \
	zip \
	zlib1g-dev

# Install ca-certificates, and update the certificate store.
apt-get install -y ca-certificates-java && update-ca-certificates -f
