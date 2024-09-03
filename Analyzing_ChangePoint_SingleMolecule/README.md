# DNA Force-Extension Analysis Jupyter Notebook

This Jupyter notebook presents a streamlined yet robust analytical pipeline for processing and interpreting DNA force-extension experiments conducted using optical tweezers. Specifically designed to handle raw experimental data in TDMS format, this notebook employs advanced polymer physics models (WLC for dsDNA, and FJC for ssDNA) to extract crucial biophysical parameters. The analysis pipeline encompasses several key components, including visualization of force-extension curves, quantification of ssDNA/dsDNA percentages catalysis by DNAp, and identification and characterization of discrete steps in DNAp processing events. 

## Methodology

### Data Acquisition and Preprocessing

1. **TDMS File Reading**: The notebook initiates by reading raw .tdms files using the `TdmsFile` function from the `nptdms` library.
2. **Region of Interest (ROI) Selection**: Users identify specific time ranges corresponding to polymerase (pol) events for focused analysis.

### Theoretical Models

1. **Twistable Worm-Like Chain (tWLC) Model**: 
   - Applicable for dsDNA
   - Parameters:
     - Contour length (Lc) = 2.85056 μm
     - Persistence length (Lp) = 56 nm
     - Twist rigidity (C) = 440 pN nm²
     - Stretching modulus (S) = 1500 pN
     - Twist-stretch coupling: g(F) = g₀ + g₁F, where g₀ = -637 pN nm, g₁ = 17 nm

2. **Freely Jointed Chain (FJC) Model**:
   - Applicable for ssDNA
   - Parameters:
     - Contour length (Lss) = 4.69504 μm
     - Kuhn length (b) = 1.5 nm
     - Stretching modulus (Sss) = 800 pN

### Data Analysis Pipeline

1. **Force-Extension Curve Generation**: Experimental data is plotted against theoretical tWLC and FJC models.
2. **ssDNA Percentage Calculation**: 
   ```
   ssDNA% = (EED - tWLC(F)) / (FJC(F) - SSB_factor(F) - tWLC(F))
   ```
   where EED is the end-to-end distance, and SSB_factor accounts for single-strand binding protein effects.
3. **Base Pair Conversion**: ssDNA percentages are converted to base pair counts based on the DNA construct length.
4. **Change-Point Detection**: Piecewise linear fitting is applied to identify discrete steps in the DNA processing events.

### Visualization and Output

1. **Interactive Plots**: The notebook generates several interactive matplotlib figures:
   - Force vs. Distance curves (experimental data overlaid with theoretical models)
   - Base pairs vs. Time plots
   - Step-fitted data visualizations
2. **Data Export**: Processed data is exported to Excel files for further analysis:
   - Raw data: `[original_filename]-cycle#[cycle_number]_rawData.xlsx`
   - Step-detected data: `[original_filename]-cycle#[cycle_number]_StepDetectedData.xlsx`

## Requirements

- Python 3.9.13
- JupyterLab 3.0.12
- Key Libraries:
  - matplotlib 3.3.4
  - pwlf 2.2.1
  - npTDMS 1.1.0
  - numpy 1.20.1
  - pandas 1.2.3
  - scipy 1.6.1
  - sympy

For a complete list of dependencies, refer to the `requirements.txt` file.

## Usage Instructions

1. Launch JupyterLab and open the notebook.
2. Execute cells sequentially, providing the following inputs when prompted:
   - Full path to the TDMS file
   - Cycle number of interest
   - Time range for the region of interest (ROI)
   - Number of segments for piecewise linear fitting
3. Analyze the generated plots and exported data files.

## Customization

The notebook allows for parameter customization to accommodate various experimental setups:

- Bead size (default: 1.76 μm)
- Exonuclease force (default: 50 pN)
- Polymerase force (default: 10 pN)
- SSB_factor (force-dependent)
- DNA construct length (default: 8393 bp)

Modify these parameters in the designated cells to tailor the analysis to specific experimental conditions.

## Results and Discussion

The notebook generates a comprehensive analysis of DNA force-extension behavior, including:

1. Comparative analysis of experimental data against theoretical polymer models
2. Quantification of ssDNA/dsDNA fractions throughout the experiment
3. Detection and characterization of discrete steps in DNA processing events
4. Statistical analysis of event durations, fragment lengths, and processing rates

These results provide insights into the mechanics of DNA-protein interactions and the kinetics of enzymatic processes at the single-molecule level.

## Limitations and Future Work

- Integration with machine learning algorithms for improved change-point detection and noise reduction could enhance the analysis pipeline.
-  We are exploring the adoption of more advanced Python-centric methodologies for data analysis, such as the [Bayesian changepoint detection for single-molecule analysis](https://github.com/longfuxu/bayesian_changepoint_detection_single_molecule).

## References

1. Gross, P., et al. (2011). Quantifying how DNA stretches, melts and changes twist under tension. Nature Physics, 7(9), 731-736.
2. Smith, S. B., Cui, Y., & Bustamante, C. (1996). Overstretching B-DNA: the elastic response of individual double-stranded and single-stranded DNA molecules. Science, 271(5250), 795-799.
3. Bustamante, C., Marko, J. F., Siggia, E. D., & Smith, S. (1994). Entropic elasticity of lambda-phage DNA. Science, 265(5178), 1599-1600.