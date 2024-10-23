# iridophore- automeasure
Python code for automated measuring of iridophore crystals from TEM images


## Scripts
iridophore_crystal_extract.py: Python script for extracting and measuring iridophore crystals from TEM images
iridiophore_crystal_stats.py: Python script for statistical testing of iridophore crystals between iridophore types


## Getting Started
### Directories
Setup the following directories:
- /detection: Input images organized by magnifications and outputs for iridophore crystal extraction and measurements 
  - /images: Type1 and Type2 iridophore input images used for measurements 
  - /out: output CSV files containing measurements
  - /masks: iridophore masks creates in Fiji/ImageJ for noisy images

- /stats: Statistical test outputs

## Run
``
