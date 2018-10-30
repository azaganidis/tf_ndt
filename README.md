# tf_ndt
Tensorflow implementation of Normal Distributions Transform

Normal Distribution Transform works. Test with 
  python rosCloudToNDT.py
The file publishes the resulting NDT's on ROS. A listener can then do the registration/visualization.
https://github.com/azaganidis/se_ndt contains an example for visualization, but is undocumented.

Registration in tensorflow is working, but slow due to operations incompatible with cuda. Not recommended yet.
