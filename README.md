# Tracking the neurodevelopmental trajectory of beta band oscillations with OPM-MEG
Lukas Rier and Nathalie Rhodes et al. eLife 2024 (https://doi.org/10.7554/eLife.94561.1)

All data can be found on zenodo:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11126593.svg)](https://doi.org/10.5281/zenodo.11126593)


Copy your project_directories ("Children" and "Adults") into the folder containing your clone of this repository ("code" below).

Download FieldTrip version 20220906 at https://www.fieldtriptoolbox.org/download.php and save the toolbox into "code". Repeat the same for HMM toolboxes (see below).

Download and extract the zip folder to '/project_directory/data' and clone this repository to '/project_directory/scripts' to get the following file structure:

.   
|code   
|---|fieldtrip-20220906  
|---|HMM-MAR-master  
|---|HMM_bursts-master  
|---|Children   
|---|---|--> derivatives   
|---|---|--> sub-001   
|---|---|--> sub-002   
|---|---|--> ...   
|---|---|--> sub-027   
|---|Adults   
|---|---|--> derivatives   
|---|---|--> sub-101   
|---|---|--> sub-102   
|---|---|--> ...   
|---|---|--> sub-126  


Run_analyses.m, Run_analyses_adults.m, Run_HMM.m will produce all necessary outputs to reproduce the results presented in the paper.

Any RESULTS_*.m scripts and PaperPlots.m will reproduce the figures. A range of .mat files have been provided that will allow reproduction of the figures without running the entire analysis.

All analyses have been run on Matlab 2022b.

Some dependencies have been included here for ease:

- violin plot toolbox (https://github.com/bastibe/Violinplot-Matlab)
- gifti-1.8 for brain surface plots (https://vcs.ynic.york.ac.uk/hw1012/mpsych_gradient/tree/master/src/gifti-1.8)

HMM related dependencies can be fount at:
- https://github.com/ZSeedat/HMM_bursts/
- https://github.com/OHBA-analysis/HMM-MAR