# decals-cnn repository  
The project aims to create a CNN which can identify and classify tidal features in images of galaxies from the DECaLS survey.  

## People and Affiliations

### People involved:  

#### Primary contact  
contact: Alexander Gordon

email: Alexander.Gordon@ed.ac.uk

affiliation: Institute for Astronomy, School of Physics and Astronomy, University of Edinburgh

## Other Key Information

### Publications:   
A paper has been accepted for publication in [Monthly Notices of the Royal Astronomical Society](https://doi.org/10.1093/mnras/stae2169). A author accepted manuscript is available on [arXiv](https://doi.org/10.48550/arXiv.2404.06487).

### Disclaimer, copyright & licencing:  


## Installation and Directions for Use

### Installation
To install the repository navigate to the directory you wish to use and run
```console
git clone https://github.com/aj-gordon/decals-cnn
```

### Directions for Use
Most of the functionality can be run from either ```run.py``` or ```full_sample_run.py```. Command line arguments support the running of these files. 

Use ```run.py``` to perform training and testing. Use ```full_sample_run```.py for deployment.

## Software, Scripts, and Data

### Software:  
The following software (python and packages) are used:

* python: 3.10.9
* matplotlib: 3.7.2
* numpy: 1.24.3
* pandas: 2.1.0
* scikit-learn: 1.3.0
* tensorflow: 2.10.0

### Scripts:   
The following scripts are contained in the repository: 

* cnns
  * ```__init__.py```
  * ```decals_net.py``` - Tensorflow implementation of the network used in this project.
* utils
  * ```__init__.py```
  * ```data.py``` - A class to hold image and label data for loading into the network. Supports test, train, and validation splitting and augmentation of the data.
  * ```metrics.py``` - Some handy metrics to support analysis.
  * ```plot_hist.py``` - A handy method for plotting training history of the network. Supports the history being supplied as a dictionary or output from ```model.fit()```
  * ```preprocessor.py``` - A tensorflow model for preprocessing and augmenting the data.
  * ```read_args.py``` - For reading command line arguments.
  * ```save_to_csv.py``` - A handy method for writing outputs to file.
* ```full_sample_run.py``` - For deployement.
* ```run.py``` - For training and testing.

### Data: 
The Galaxy Zoo: DECaLS catalogue and images originate from Walmsley et al., 2022, and are readily available [here](https://zenodo.org/record/4573248).

Other images from the Legacy Survey are available from the [Legacy Survey](https://www.legacysurvey.org/) webpage.

The network code is available in [this repository](https://github.com/aj-gordon/decals-cnn). All further code and data for this article will be shared upon a reasonable request to the corresponding author.
