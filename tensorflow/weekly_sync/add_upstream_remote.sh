# add the google upstream repo as a remote and fetch it
git remote add google_upstream https://github.com/tensorflow/tensorflow.git
git fetch google_upstream

# create a branch to merge the updates from the remote master
# git checkout develop-upstream
# git merge --ff-only origin/develop-upstream
git checkout -b develop-upstream-sync-`date +%y%m%d`


# merge the updates from remote master
git merge google_upstream/master --no-edit 
