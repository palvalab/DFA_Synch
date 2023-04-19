MATLAB CODE

System requirements:
This code requires Matlab software. The code has been tested on versions R2018, R2020 and R2021 on Windows platforms (Windows 10 Enterprise and Windows Server 2019 Standard). 
Installation of Matlab should take less than 1 hour.

Instructions:
Download the dataset from https://doi.org/10.5061/dryad.vdncjsxzn and save the two files MEG_data.mat and SEEG_data.mat to a folder on your computer. In the matlab scripts, set the variable paths.rsdata to the path of this folder.
The scripts replicate the main findings of the study which are shown in Figure 2.


PYTHON CODE

System requirements:
This code requires Python with the libraries numpy, scipy, sys, os, statsmodels, typing, tqdm, and matplotlib. Most of these should be part of a standard python installation, the rest can be installed with "conda install" or "pip install". 
The library cupy can be used if a NVidea graphics card is present and the cuda toolkit installed. If not, computations will be carried out with numpy which will take slightly longer. 
The code was tested under python 3.10.6 with numpy 1.23.4 and cupy 11.3.0 on a Windows platform (Windows Server 2019 Standard). 
Installation of python should take less than 1 hour, with another 1-2 hours for the optional installation of cuda and cupy.

Instructions:
Open the script "dfa_synch_kuramoto.py" and set the variable "source directory" to the folder where the script and the folders "metadata" and "utils" are located. Run the script to replicate Figure 1C.
