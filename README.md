# iridophore-automeasure
Python code for automated measuring of iridophore crystals from TEM images

## Basic folder structure example
```
.
├── iridophore_crystal_extract.py  # script for extracting and measuring iridophore crystals from TEM images
├── iridiophore_crystal_stats.py   # script for statistical testing of crystals measurements between iridophore subtypes
├── stats                          # outputs of statistical test
└── detection                      # input and outputs of iridophore crystal detection
    ├── Type1                      # inputs are pre-organized by iridophore subtype and image magnification
    └── Type2             
         ├── 2500x        
         └── 5000x        
             ├── image             # iridophore input TEM images used for measurements 
             ├── masks             # iridophore masks for noisy images (optional, can be created in softwares such as Fiji/ImageJ)
             └── out               # outputs of detection
```

## Test data
Minumum test dataset provided as example (iridophore_testdata.tar.gz).
Full dataset used in manuscript available with request.
