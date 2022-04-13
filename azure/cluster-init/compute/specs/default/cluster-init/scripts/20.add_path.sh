#! /bin/bash

# add the profile files into /etc/profile.d
cp "$CYCLECLOUD_SPEC_PATH"/files/deps_path.sh /etc/profile.d/

# make sure both are readable and executable
chmod a+rx /etc/profile.d/deps_path.sh
