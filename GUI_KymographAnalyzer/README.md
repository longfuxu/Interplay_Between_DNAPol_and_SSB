# Multi-Modal Single-Molecule Analysis Platform

An integrated GUI application for analyzing single-molecule data from Optical Tweezers (OT) and Kymographs to study DNAp-SSB interactions.

## Features

### Comprehensive Analysis Workflow
- **OT Data Analysis**: Process and analyze Optical Tweezers data for tracking DNA polymerase movement
- **Kymograph Analysis**: Visualize and extract SSB protein trajectories from kymographs
- **Integrated DNAp-SSB Interaction**: Combine data from both modalities to analyze interactions

### OT Data Processing
- Load TDMS files from optical tweezers experiments
- Apply tWLC and FJC models for polymer physics
- Track ss/dsDNA junction representing DNA polymerase movement
- Piecewise linear segment fitting for detailed analysis of trajectories
- Exo and pol phase analysis

### Kymograph Processing
- Multi-format support for loading various image file types (TDMS, TIFF, HDF5)
- Enhanced visualization with contrast and channel control
- Advanced ROI selection for targeted analysis
- SSB trajectory detection and tracking

### DNAp-SSB Interaction Analysis
- Synchronize data from OT and kymograph sources
- Calculate and visualize distance between DNAp and SSB proteins
- Advanced data visualization tools for correlation analysis
- Comprehensive export options for further analysis

### Modern User Interface
- Intuitive tab-based navigation
- Real-time data visualization
- Integrated plotting tools
- Responsive design with modern controls

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/multi-modal-single-molecule-analysis.git
   cd multi-modal-single-molecule-analysis
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python -m main
   ```

## Usage Guide

### OT Data Analysis Tab
1. **Load Data**: Browse and select TDMS files containing OT data
2. **Select Time Ranges**: Click on the plot after selecting input boxes to set exo and pol phases
3. **Adjust Parameters**: Set model parameters for tWLC and FJC models
4. **Analyze**: Use the buttons to fit models, track junction movement, and perform segment fitting
5. **Export**: Save the processed data and plots for further analysis

### Kymograph Analysis Tab
1. **Load Data**: Select kymograph image files in supported formats
2. **Visualization**: Adjust contrast and select color channels for better visualization
3. **Select ROI**: Define regions of interest for focused analysis
4. **Detect Trajectories**: Identify and track SSB protein trajectories
5. **Analyze**: Perform quantitative analysis on the tracked trajectories

### DNAp-SSB Interaction Tab
1. **Select Data**: Load the processed OT data and kymograph trajectory data
2. **Synchronize**: Set time offset and distance scaling to align the datasets
3. **Analyze Interaction**: Calculate distances and correlations between DNAp and SSB
4. **Visualize**: Examine the combined trajectories and interaction metrics
5. **Export Results**: Save the integrated analysis for further processing

## Example Workflow

### Analyzing DNAp-SSB Interaction
1. Start by processing OT data in the OT Data Analysis tab
2. Extract and save DNAp trajectory data
3. Switch to the Data Loading tab to load a kymograph image
4. Process the kymograph to extract SSB trajectories
5. Go to the DNAp-SSB Interaction tab
6. Load both processed datasets and synchronize them
7. Analyze the interaction between DNAp and SSB
8. Export the combined results

## Troubleshooting

### Common Issues
- **File Loading Errors**: Ensure files are in supported formats and not corrupted
- **Model Fitting Issues**: Verify parameter values are within reasonable ranges
- **Trajectory Detection Problems**: Adjust contrast and ROI to improve detection
- **Memory Errors**: Try working with smaller regions or downsampled images

### Getting Help
If you encounter issues not covered in this documentation, please:
1. Check the console for error messages
2. Refer to the API documentation
3. File an issue on the GitHub repository

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This software builds upon methods developed for analyzing single-molecule data
- The kymograph analysis components are inspired by image analysis techniques in biophysics
- The OT data analysis methods are based on established polymer physics models
- Thanks to all contributors who have helped improve this software

## Citation

If you use this software in your research, please cite:
```
Author, A. et al. Multi-Modal Single-Molecule Analysis Platform: Integrating Optical Tweezers and Kymograph Data for DNAp-SSB Interaction Studies. Journal of Biophysics (2023).
``` 