cd /root/temp/horovod-fork

git fetch upstream

git checkout master
git merge --ff-only upstream/master
git push origin master
