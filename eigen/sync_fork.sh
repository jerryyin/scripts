cd /home/rocm-user/eigen-fork

git fetch upstream

git checkout master
git merge --ff-only upstream/master
git push origin master
