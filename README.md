# Force Measurement Analysis and Kymograph Analysis

This repository contains two separate Python scripts for analyzing fluorescence microscopy images of DNA polymerase molecules and processing force measurement data. Both scripts process data in `.tdms` format, and used together, they can provide insights into the dynamics of DNA polymerase activity. An example trace is provided in the folder `example data analysis`

## Force Measurement Analysis

This Python jupyternotebook `1_CalculatingDNApTrace_OT.ipynb` performs data analysis on a Tdms file from force measurements. The analysis includes fitting data to the tWLC and FJC models, calculating ssDNA percent as a function of time, plotting DNA polymerase trace, calculating basepair changes, and exporting processed data to an Excel file.

### Dependencies

- python==3.9
- jupyterlab==3.0.12
- matplotlib==3.3.4
- more-itertools==8.7.0
- npTDMS==1.1.0
- numpy==1.20.1
- opencv-python==4.5.1.48
- pandas==1.2.3
- scipy==1.6.1
- tifffile==2021.3.5
- sympy

### Usage

1. Input the file name with an absolute path.
2. Run the script and follow the prompts to provide the starting and ending times for the events of interest (exo and pol).
3. Provide the cycle number of interest.
4. On the resulting plots, review the results and save output images and Excel files as desired.

### Output

The script generates several plots and processed data as follows:

1. Force extension curves for tWLC and FJC models.
2. Experimental data fit to tWLC and FJC Model.
3. Basepair changes as a function of time.
4. DNA polymerase traces.
5. Smoothed basepair changes as a function of time.

Additionally, an Excel file containing processed data such as time, ssDNA percentage, junction position, and basepairs will be generated.

# Kymograph Analysis

This script is used to analyze kymographs in fluorescence microscopy images of DNA polymerase molecules. It processes data in `.tdms` format, converts it into image data, helps visualize images and DNA polymerase trajectories on the kymographs, and  calculates distances between DNA polymerase and SSB (single-stranded DNA binding protein) trajectories. Data can then be saved in `.csv` format for further analysis.

## Requirements

- Python 3.9
- JupyterLab 3.0.12
- lumicks.pylake 0.8.1
- matplotlib 3.3.4
- more-itertools 8.7.0
- npTDMS 1.1.0
- numpy 1.20.1
- opencv-python 4.5.1.48
- pandas 1.2.3
- scipy 1.6.1
- tifffile 2021.3.5
- tabulate 0.8.6

## Steps

1. Read raw image data of a `.tdms` file with TdmsFile function.
2. Access `.tdms` file and convert it into image data.
3. Display the kymograph in the notebook.
4. Save the `.tdms` file as a `.tiff` image.
5. Split the image into different color channels and display the images.
6. Read and display the DNA polymerase trajectory calculated from force measurement data.
7. Overlay DNA polymerase trace on top of both DNA polymerase and SSB images.
8. Detect SSB trajectory bands using the lumicks.pylake package.
9. Refine and smooth SSB trajectories using the Savitzky-Golay filter.
10. Plot DNA polymerase and SSB trajectories together in separate subplots.
11. Calculate distances between DNA polymerase and SSB trajectories.
12. Save data in separate `.csv` files for easy retrieval later.

## Usage

- Modify the file path in the input() function for the kymograph `.tdms` file and the trace_file.
- Customize plot appearance and region of interest (ROI) as needed.
- Adjust the parameters for the Savitzky-Golay filter for optimal results.
- Run the script and visualize the results in the notebook.

## Output

- DNA polymerase traces with time and position data saved in `.csv` format.
- Smoothed SSB trajectories with time and position data saved in `.csv` format.
- Distances between DNA polymerase and SSB traces calculated and saved in `.csv` format.
- Plots saved in `.eps` and `.png` format.

# Note

All codes listed in this repository are developed by Longfu Xu (longfuxu.com) during the phD work in Gijs Wuite Group.

Please note that the code in this repository is custom written for internal lab use and still may contain bugs; external support cannot be guaranteed.

Developer:

Longfu Xu (longfuxu.com) . Maintenance, development, support. For questions or reports, e-mail: [longfu2.xu@gmail.com](mailto:l2.xu@vu.nl)