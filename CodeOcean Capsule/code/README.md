# Interplay Between DNA Polymerase and SSB Analysis Pipeline

This repository contains a pipeline for analyzing optical tweezers (OT) data and kymograph data related to the interplay between DNA Polymerase and SSB proteins. 

## Quick Start: Click the 'Reproduce Run' in the right side to check all the analyzed results.

## Scripts

- **`main.py`**: The entry point script that orchestrates the analysis pipeline. It runs the following scripts in sequence:
  - **`OTdataAnalyzer.py`**: Processes optical tweezers data (e.g., force and distance measurements) and saves results to the `/results` folder.
  - **`KymographAnalyzer.py`**: Analyzes kymograph data (e.g., imaging data of DNA Polymerase and SSB trajectories) and saves results to the `/results` folder.

## Instructions for Use in Code Ocean

1. **Directory Structure**:
- Place the scripts (`main.py`, `OTdataAnalyzer.py`, `KymographAnalyzer.py`) in the root directory of your Code Ocean capsule.
- Create a `data` folder in the root directory and place your input files (e.g., `OT data example.tdms` and `image data example.tdms`) inside it.
- The script will automatically create a `results` folder to store all output files (plots, TIFF, CSV, etc.).

2. **Run the Pipeline**:
- Click the "Reproduce Run" button in Code Ocean. This will execute `main.py`, which will:
1. Run `OTdataAnalyzer.py` to process the OT data and generate initial results (e.g., Excel file with processed data).
2. Run `KymographAnalyzer.py` to analyze the kymograph data using the OT results and generate additional outputs (e.g., plots and CSV files).
- All results will be saved in the `/results` folder.

3. **Verify Results**:
- Check the `/results` folder for output files, including `.png` plots, `.tiff` images, and `.csv` data files.
- Review the console output in Code Ocean for any error messages or completion confirmation.


## Open-Source Code
The open-sourced code for this project is available on GitHub at:  
[https://github.com/longfuxu/Interplay_Between_DNAPol_and_SSB](https://github.com/longfuxu/Interplay_Between_DNAPol_and_SSB)

## Contact
For questions or contributions, please open an issue on the GitHub repository or contact the author directly (longfuxu@berkeley.edu).
