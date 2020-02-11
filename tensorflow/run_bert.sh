# Hi Deven and Jack,
#
# Our internal BERT repo(https://github.com/ROCmSoftwarePlatform/BERT) has
#
# * a master branch(https://github.com/ROCmSoftwarePlatform/BERT/tree/master) 
#    which has model level changes, and scripts for training and testing
# * a develop-upstream branch(https://github.com/ROCmSoftwarePlatform/BERT/tree/develop-upstream)
#    which simply tracks the original BERT repo (https://github.com/google-research/bert)
# * a develop-upstream-with-scripts(https://github.com/ROCmSoftwarePlatform/BERT/tree/develop-upstream-with-scripts)
#     which is the original BERT with our training and testing scripts
#
# I have been using develop-upstream-with-scripts to collect the nongemm performance,
# with the reasoning that model level changes can add or remove kernels that affect the performance for any one using BERT on rocm.
# That being said these steps will work with the master branch as well. 
#
# From the root directory 
# * do a performance run with hcc calls logged (1500 steps with a small dataset)  
#    "export HCC_PROFILE=2"
#    "sh scripts/train_bert_large_perf_amd.sh"
# * pass the log of the perf run to rpt for a summary with all kernels
#    "/opt/rocm/hcc/bin/rpt --topn -1 bert_large_perf_train/wiki_00_ba4_seq512/wiki_00_ba4_seq512.txt >wiki_00_ba4_seq512_summary.txt"
# * then use the python script attached to a csv of the top nongemm kernels
#    "python3 nongemm_csv_amd.py --amd_path="wiki_00_ba4_seq512_summary.txt"


cd /root/BERT

#export HCC_PROFILE=2
export HIP_TRACE_API=1
export TF_CPP_MIN_VLOG_LEVEL=3

sh ./scripts/train_bert_large_perf_amd.sh


# /opt/rocm/hcc/bin/rpt --topn -1 bert_large_perf_train/wiki_00_ba4_seq512/wiki_00_ba4_seq512.txt >wiki_00_ba4_seq512_summary.txt

# python3 nongemm_csv_amd.py --amd_path="wiki_00_ba4_seq512_summary.txt"
