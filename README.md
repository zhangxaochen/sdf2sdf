# sdf2sdf

An implementation of the Paper SDF-2-SDF: Highly Accurate 3D Object Reconstruction

## How to run

> ./sdf2sdf -om -param \<file\> -eval \<dir\>

Options

-om                 toggle whether mask the depth maps with omasks_*.png

-param \<file\>     specify the camera intrinsics param file

-eval \<dir\>       specify the path containing test data sequences

E.g.:

> ./sdf2sdf -om -param "sdf2sdf-kinect.param" -eval "Kinect_Bunny_Turntable"

test on folder "Kinect_Bunny_Turntable", with depth maps masked, using camera intrinsics file "sdf2sdf-kinect.param".

> ./sdf2sdf -param "sdf2sdf-syn.param" -eval "Synthetic_Kenny_Circle"

test on folder "Synthetic_Kenny_Circle", without masks, using camera intrinsics file "sdf2sdf-syn.param".