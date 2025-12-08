rm -rf eigen.patch.new
diff -Naur eigen/eigen-eigen-6913f0cf7d06 eigen_archive > eigen.patch.new.unsorted
/common/scripts/archive/sort_eigen_patch.py eigen.patch.foo eigen.patch.new.unsorted eigen.patch.new
rm -rf eigen.patch.new.unsorted
