** You must follow the directions here to be able to compare to prior work!**

# TeraLasso Setup:

https://uk.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

The guidelines in that link should work for all computers.  I will give an example using my Mac:

In my Applications folder I have "MATLAB_R2023b"; right click on it to select "Show Package Contents".  Then, navigate to `extern/engines/python`.  Right click the folder and say "New Terminal at Folder".  In this terminal, activate the python environment you want to use.  Then, run `python -m pip install .`

It seems like with recent versions of Matlab (including my own) you can skip all this and just run `python -m pip install matlabengine` - however I have not tested that route.

# EiGLasso Setup

First, set up the Matlab Python Engine (by following the TeraLasso Setup section).  Now, we need to compile their C++ code to be interfaceable with Matlab.

https://github.com/SeyoungKimLab/EiGLasso/tree/main/EiGLasso_JMLR (Section 2)

On my Mac; Open the Matlab application, and cd into `other_algs/eiglasso/EiGLasso_JMLR`; once there, run `mex -output eiglasso_joint eiglasso_joint_mex.cpp -lmwlapack`.
