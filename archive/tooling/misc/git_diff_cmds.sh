
base_branch=google_upstream/master

compare_branch=$1

compare_files=$2

git diff $base_branch $compare_branch -- $compare_files
