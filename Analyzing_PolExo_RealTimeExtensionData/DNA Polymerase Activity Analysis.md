# DNA Polymerase Activity Analysis

## Overview
This Jupyter Notebook is designed to analyze the activity of DNA polymerase (Pol) and its exonuclease (exo) correction activity in real-time DNA primer extension assay. The analysis focuses on how the presence of Single-Strand Binding protein (SSB) affects both the polymerization and exonuclease activities of the enzyme. The notebook processes fluorescence intensity data over time, fitting two distinct linear models to segments of the data to extract rates of activity in the unit of Relative Fluorescence Units per second (RFU/s) as a function of SSB concentration.

## Rationale Behind the Analysis
### First Linear Fit: Initial Polymerase Activity
The first linear fit is applied to the initial 1.5 minutes of the fluorescence data. This period is critical as it represents the phase where the polymerase activity is most prominent, prior to significant exonuclease activity interference. The slope derived from this fit gives an estimate of the initial polymerase activity in RFU/s. This measure is crucial for understanding how efficiently the polymerase synthesizes DNA under varying conditions of SSB concentration, before any significant exonuclease activity begins.

### Second Linear Fit: Exonuclease Activity
The second linear fit is conducted from 1 minute after the observed minimum fluorescence intensity for the following 3.5 minutes. This timing is chosen based on the rationale that after reaching a minimum fluorescence level, the exonuclease activity of the polymerase becomes significant, leading to an increase in fluorescence due to the exonucleolytic removal of previously incorporated fluorescent nucleotides. This fit provides insight into the exonuclease (proofreading) activity of the polymerase, allowing for the comparison of polymerase fidelity under different SSB concentrations.
#### Example of the fit
![alt text](<example data plot/plot_wt_10_1.png>)

### Plotting Activity vs. SSB Concentration
The analysis generates plots that showcase both the polymerization and exonuclease activities of DNA polymerase as functions of SSB concentration. These plots are crucial for visualizing how SSB influences the balance between polymerization and proofreading activities of the polymerase, which can have significant implications for DNA replication fidelity and efficiency.
#### Pol Activity vs SSB Concentration
![alt text](<example data plot/PolActivity_SSBConc_1.5_1_4.5.png>)
#### Exo Activity vs SSB Concentration
![alt text](<example data plot/ExoActivity_SSBConc_1.5_1_4.5.png>)
## Example Data
An example dataset is provided with the notebook, demonstrating the application of the analysis to real experimental data. This dataset includes time-course fluorescence measurements obtained from real-time PCR experiments, where the concentration of SSB protein was varied to assess its impact on the activity of DNA polymerase.

The data structure consists of fluorescence intensity readings (RFU) measured over time (minutes), with different experimental conditions corresponding to various concentrations of SSB.