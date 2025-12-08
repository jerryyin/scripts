# ssh from_node to_node

# First generate the "/home/user/.ssh/id_rsa.pub" file on "from_node"
# (Do not enter a passphrase)

# ssh-keygen -t rsa

# Then create a /home/user/.ssh directory on to_node
# MAKE SURE via `ls -ld /home/user/.ssh` that it is owned by user/deven

# Then append the contents of the "/home/user/.ssh/id_rsa.pub" file on "from_node"
# to the contents of the "/home/user/.ssh/authorized_keys" file on "to_node

# cat ~/.ssh/id_rsa.pub | ssh -p 1234 deven@0.0.0.0 'cat >> .ssh/authorized_keys'

# Depending on your version of ssh you might also have to
# * Put the public key in .ssh/authorized_keys2
# * Change the permissions of .ssh to 700
# * Change the permissions of .ssh/authorized_keys2 to 640
