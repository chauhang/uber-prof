#! /bin/bash

# add the profile files into /etc/profile.d
cp "$CYCLECLOUD_SPEC_PATH"/files/deps_path.sh /etc/profile.d/

# make sure both are readable and executable
chmod a+rx /etc/profile.d/deps_path.sh
/shared/home/azureuser/.conda/bin/conda init bash
eval "$(/shared/home/azureuser/.conda/bin/conda shell.bash hook)"
chown -R azureuser:azureuser /shared/home/azureuser/.conda

cat > /etc/sudoers.d/azureuser << EOF
azureuser  ALL=(ALL) NOPASSWD: ALL
EOF
chmod 0440 /etc/sudoers.d/azureuser
