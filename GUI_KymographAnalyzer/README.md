# Correlation Image Force Analyzer

A GUI application for analyzing correlation images and force data from T7DNAp-SSB experiments.

## Project Structure

The project is organized into a modular structure to make the code more maintainable:

```
code_base_modular/
├── main.py                  # Main script to run the application
├── gui/                     # GUI-related modules
│   ├── __init__.py          # Makes the directory a package
│   ├── main_window.py       # Main window class
│   └── ui_components.py     # UI component creation methods
└── methods/                 # Analysis methods
    ├── __init__.py          # Makes the directory a package
    ├── display_methods.py   # Image display methods
    ├── analysis_methods.py  # Data analysis methods
    ├── segmentation_methods.py # Segmentation analysis methods
    ├── file_io_methods.py   # File input/output methods
    └── utility_methods.py   # Utility functions
```

## Requirements

- Python 3.6+
- tkinter
- numpy
- pandas
- matplotlib
- scipy
- opencv-python (cv2)
- nptdms
- lumicks.pylake

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   ```

2. Install the required packages:
   ```
   pip install numpy pandas matplotlib scipy opencv-python nptdms lumicks.pylake
   ```

## Usage

Run the application using:

```
python main.py
```

## Features

- Load and display kymograph files
- Load and analyze trace files
- Region of Interest (ROI) selection
- Channel selection (Red, Green, Blue)
- Contrast adjustment
- Various analysis options:
  - Plot time vs end-to-end distance and fork front position
  - Force-image analysis
  - Intensity profile with unwinding data
  - SSB trajectory detection and analysis
  - DNAp-SSB distance calculation
  - Data segmentation
- Export data to CSV files
- Save high-resolution figures

## License

This project is licensed under the MIT License - see the LICENSE file for details. 