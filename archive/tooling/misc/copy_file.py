import os

file_name="ockl.amdgcn.bc"

from_loc="/common/"

to_locs=["/opt/rocm/hcc/lib", "/opt/rocm/hcc/rocdl/ockl", "/opt/rocm/lib", "/opt/rocm/opencl/lib/x86_64/bitcode"]


for to_loc in to_locs:
    from_file = os.path.join(from_loc, file_name)
    to_file = os.path.join(to_loc, file_name)
    bkup_file =  os.path.join(to_loc, file_name+".bkup")
    
    print ("----------------------------")
    
    # cmd = "ls -l {} {}".format(from_file, to_file)
    # cmd = "ls -l {}".format(bkup_file)
    # os.system(cmd)


    cmd = "cp {} {}".format(to_file, bkup_file)
    print (cmd)
    os.system(cmd)
    

    cmd = "cp {} {}".format(from_file, to_file)
    print (cmd)
    os.system(cmd)
    
