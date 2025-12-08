#version=0.15.0
#version=0.19.2
#version=0.20.0
#version=0.23.2
#version=0.24.1
#version=0.26.1
#version=0.27.1
#version=0.29.1
#version=1.1.0
#version=1.2.1
version=2.0.0
#version=2.1.0
cd /root/
wget https://github.com/bazelbuild/bazel/releases/download/$version/bazel-$version-installer-linux-x86_64.sh
chmod a+x bazel-$version-installer-linux-x86_64.sh
./bazel-$version-installer-linux-x86_64.sh
