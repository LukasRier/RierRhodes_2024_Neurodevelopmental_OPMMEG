# Tracking the neurodevelopmental trajectory of beta band oscillations with OPM-MEG
Lukas Rier and Nathalie Rhodes et al. eLife 2024 (https://doi.org/10.7554/eLife.94561.1)

All data can be found on zenodo:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11126593.svg)](https://doi.org/10.5281/zenodo.11126593)


Copy your project_directories ("Children" and "Adults") into the folder containing your clone of this repository ("code" below).

Download FieldTrip version 20199212 at https://www.fieldtriptoolbox.org/download.php and save the toolbox into "code".

Download and extract the zip folder to '/project_directory/data' and clone this repository to '/project_directory/scripts' to get the following file structure:

.   
|code   
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