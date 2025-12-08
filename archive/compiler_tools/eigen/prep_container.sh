cd /home/rocm-user

git clone https://gitlab.com/deven-amd/eigen eigen-fork
cd eigen-fork
git remote add upstream https://gitlab.com/libeigen/eigen
git fetch upstream

cd /home/rocm-user
rm -rf eigen
ln -s eigen-fork eigen
