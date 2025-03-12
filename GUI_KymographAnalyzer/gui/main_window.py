"""
Main window class for the Kymograph Analyzer.
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import cv2
import pandas as pd
from nptdms import TdmsFile
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import lumicks.pylake as lk
import tifffile
import h5py
import traceback
import json
import time
import warnings
from scipy.signal import savgol_filter
try:
    # Import the FastPWLFit class for piecewise linear segment fitting
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from lib.fast_pwl_fit import FastPWLFit
except ImportError:
    print("Warning: FastPWLFit module not found. Segment fitting will be disabled.")
    FastPWLFit = None

# Add the parent directory to the path so we can import the methods
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import methods from the methods package
from methods.display_methods import (
    update_contrast_from_slider, display_single_channel, 
    display_image_left, save_high_res
)
from methods.file_io_methods import (
    select_kymo_file, read_kymo_file, select_trace_file, 
    read_trace_file, read_tdms_file, export_data, export_distance_data
)
from methods.analysis_methods import (
    plot_time_vs_eed_and_fork_front_popout, plot_time_vs_fork_front_reverse_popout,
    force_image_analyzer, force_image_analyzer_reverse, display_intensity_with_unwinding,
    detect_ssb_trajectories, plot_ssb_trajectories, plot_dnap_ssb
)
from methods.segmentation_methods import (
    first_derivative, segment_data, process_segments, segment_distance,
    export_segments, calculate_dnap_ssb_distance
)
from methods.utility_methods import (
    update_channel, on_entry_focus, on_entry_focus_out, 
    on_roi_entry, onselect_rect, on_image_click
)

# Import UI components
from gui.ui_components import (
    create_matplotlib_figure,
    create_channel_controls,
    create_roi_controls,
    create_parameter_controls,
    create_file_selector,
    create_plot_frame,
    create_tab_control,
    create_status_bar,
    set_modern_style
)

class KymographAnalyzer:
    """
    Main class for the Kymograph Analyzer application.
    """
    def __init__(self, master):
        """Initialize the Kymograph Analyzer application."""
        self.master = master
        self.master.title("Multi-Modal Single-Molecule Analysis")
        
        # Initialize variables before creating UI components
        self.initialize_variables()
        
        # Create modern-styled UI with tabs
        self.create_tab_control()
        
        # Set up individual tabs
        self.setup_ot_analysis_tab(self.tabs["ot_analysis"])
        self.setup_data_loading_tab(self.tabs["data_loading"])
        self.setup_visualization_tab(self.tabs["visualization"])
        self.setup_analysis_tab(self.tabs["analysis"])
        self.setup_results_tab(self.tabs["results"])
        self.setup_interaction_analysis_tab(self.tabs["interaction"])
        
        # Create a status bar at the bottom of the window
        self.create_status_bar()
        
        # Set initial status
        self.status_var.set("Ready")
        
    def initialize_variables(self):
        """Initialize instance variables"""
        # Data storage
        self.img = None
        self.time_per_line = None
        self.px_size = None
        self.trace = None
        self.traces = None
        self.smoothed_traces = None
        self.distance_data = None
        
        # Add aliases for methods that expect these names
        self.kymo_data = None
        self.trace_data = None
        
        # UI state
        self.selected_channel = 'Red'
        self.contrast_max = 98
        self.status_var = tk.StringVar(value="Ready")
        
        # ROI coordinates
        self.roi_coords = {'start_x': None, 'end_x': None, 'start_y': None, 'end_y': None}
        
        # Display state tracking
        self.kymograph_displayed = False
        self.junction_plot_displayed = False
        
        # Plot components
        self.kymo_fig = None
        self.kymo_ax = None
        self.kymo_canvas = None
        self.preview_fig = None
        self.preview_ax = None
        self.preview_canvas = None
        
    def create_tab_control(self):
        """Create a tabbed interface for the application"""
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.tabs = {
            "ot_analysis": ttk.Frame(self.notebook),
            "data_loading": ttk.Frame(self.notebook),
            "visualization": ttk.Frame(self.notebook),
            "analysis": ttk.Frame(self.notebook),
            "results": ttk.Frame(self.notebook),
            "interaction": ttk.Frame(self.notebook)
        }
        
        self.notebook.add(self.tabs["ot_analysis"], text="OT Data Analysis")
        self.notebook.add(self.tabs["data_loading"], text="Data Loading")
        self.notebook.add(self.tabs["visualization"], text="Visualization")
        self.notebook.add(self.tabs["analysis"], text="Analysis")
        self.notebook.add(self.tabs["results"], text="Results")
        self.notebook.add(self.tabs["interaction"], text="DNAp-SSB Interaction")
        
        # Configure tab grids
        for tab in self.tabs.values():
            tab.columnconfigure(0, weight=1)
            tab.rowconfigure(0, weight=1)
        
    def setup_data_loading_tab(self, tab):
        """Set up the data loading tab"""
        # Create a frame for the tab content
        main_frame = ttk.Frame(tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure grid
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=0)  # Input section
        main_frame.rowconfigure(1, weight=1)  # Preview section
        
        # Create input section
        input_frame = ttk.LabelFrame(main_frame, text="File Input")
        input_frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        
        # File selectors
        kymo_filetypes = [
            ("TDMS files", "*.tdms"),
            ("HDF5 files", "*.h5"),
            ("TIFF files", "*.tif *.tiff"),
            ("All files", "*.*")
        ]
        self.kymo_entry = create_file_selector(
            input_frame, "Kymograph File:", self.load_kymo_file, kymo_filetypes
        )
        
        trace_filetypes = [
            ("Excel files", "*.xlsx *.xls"),
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]
        self.ot_entry = create_file_selector(
            input_frame, "OT Processed Data:", self.load_trace_file, trace_filetypes
        )
        
        # Create preview sections
        preview_frame = ttk.LabelFrame(main_frame, text="Kymograph Preview")
        preview_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        trace_preview_frame = ttk.LabelFrame(main_frame, text="OT Data Preview")
        trace_preview_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        
        # Create kymograph preview
        self.kymo_frame, self.kymo_fig, self.kymo_ax, self.kymo_canvas = create_plot_frame(
            preview_frame, title="Kymograph", figsize=(4, 4)
        )
        
        # Create OT data preview
        self.preview_frame, self.preview_fig, self.preview_ax, self.preview_canvas = create_plot_frame(
            trace_preview_frame, title="OT Data", figsize=(4, 4)
        )
        
        # Add a button frame for basic controls
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        # Add refresh and next tab buttons
        ttk.Button(button_frame, text="Refresh Previews", 
                   command=self.refresh_previews).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Next: Visualization", 
                   command=lambda: self.notebook.select(1)).pack(side=tk.RIGHT, padx=5)
    
    def setup_visualization_tab(self, tab):
        """Set up the visualization tab"""
        # Create a frame for the tab content
        main_frame = ttk.Frame(tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure grid
        main_frame.columnconfigure(0, weight=0)  # Controls
        main_frame.columnconfigure(1, weight=1)  # Visualization
        main_frame.rowconfigure(0, weight=1)  # Main row
        
        # Create controls frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Channel controls
        self.channel_var, self.contrast_var = create_channel_controls(
            controls_frame, self.selected_channel, self.update_channel
        )
        
        # Add a callback to the contrast slider
        self.contrast_var.trace_add("write", self.update_contrast)
        
        # ROI controls
        self.roi_entries = create_roi_controls(
            controls_frame, self.apply_roi, add_minimap=True
        )
        
        # Add buttons for visualization
        buttons_frame = ttk.LabelFrame(controls_frame, text="Visualization Options")
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add buttons for various visualization options
        ttk.Button(buttons_frame, text="Show Full Image", 
                   command=self.show_kymograph_popup).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(buttons_frame, text="Save Image", 
                   command=self.save_kymograph_image).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(buttons_frame, text="Junction Forward", 
                   command=self.plot_junction_forward).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(buttons_frame, text="Junction Reverse", 
                   command=self.plot_junction_reverse).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(buttons_frame, text="Force-Image Analyzer", 
                   command=self.force_image_analyzer).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(buttons_frame, text="Intensity Profile", 
                   command=self.display_intensity_with_unwinding).pack(fill=tk.X, padx=5, pady=2)
        
        # Add navigation buttons to move between tabs
        nav_frame = ttk.Frame(controls_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(nav_frame, text="Previous: Data Loading", 
                   command=lambda: self.notebook.select(0)).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next: Analysis", 
                   command=lambda: self.notebook.select(2)).pack(side=tk.RIGHT, padx=5)
        
        # Create visualization frame
        visualization_frame = ttk.LabelFrame(main_frame, text="Kymograph Visualization")
        visualization_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        # Create matplotlib figure for kymograph
        self.main_kymo_frame, self.main_kymo_fig, self.main_kymo_ax, self.main_kymo_canvas = \
            create_matplotlib_figure(visualization_frame, "Kymograph View", "Time (px)", "Position (px)", figsize=(8, 6))
    
    def setup_analysis_tab(self, tab):
        """Set up the analysis tab"""
        # Create a frame for the tab content
        main_frame = ttk.Frame(tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure grid
        main_frame.columnconfigure(0, weight=0)  # Parameters
        main_frame.columnconfigure(1, weight=1)  # Analysis view
        main_frame.rowconfigure(0, weight=1)  # Main row
        
        # Create parameters frame
        params_frame = ttk.Frame(main_frame)
        params_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Parameter sections
        param_sections = {
            "Trajectory": {
            "Line Width": 5,
            "Pixel Threshold": 25,
            "Window Size": 6
            },
            "Smoothing": {
                "Window Size": 31,
                "Polynomial Order": 3
            },
            "Calibration": {
                "X Offset": -23,
                "Y Offset": 7,
                "Time Scale": 1.0,
                "Position Scale": 1.0
            }
        }
        
        self.param_entries = create_parameter_controls(params_frame, None, param_sections)
        
        # Add analysis buttons
        analysis_frame = ttk.LabelFrame(params_frame, text="Analysis Tools")
        analysis_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(analysis_frame, text="Detect Trajectories", 
                   command=self.detect_trajectories).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(analysis_frame, text="Plot Trajectories", 
                   command=self.plot_trajectories).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(analysis_frame, text="Filter Trajectories", 
                   command=self.filter_trajectories).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(analysis_frame, text="Calculate Distance", 
                   command=self.calculate_dnap_ssb_distance).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(analysis_frame, text="Segment Distance", 
                   command=self.segment_distance).pack(fill=tk.X, padx=5, pady=2)
        
        # Add export frame
        export_frame = ttk.LabelFrame(params_frame, text="Export")
        export_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(export_frame, text="Export Trajectories", 
                   command=lambda: self.export_data("trajectories")).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(export_frame, text="Export Distance Data", 
                   command=self.export_distance_data).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(export_frame, text="Export Segments", 
                   command=self.export_segments).pack(fill=tk.X, padx=5, pady=2)
        
        # Add navigation buttons to move between tabs
        nav_frame = ttk.Frame(params_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(nav_frame, text="Previous: Visualization", 
                   command=lambda: self.notebook.select(1)).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next: Results", 
                   command=lambda: self.notebook.select(3)).pack(side=tk.RIGHT, padx=5)
        
        # Create analysis view frame
        analysis_view_frame = ttk.LabelFrame(main_frame, text="Analysis View")
        analysis_view_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        # Create matplotlib figure for analysis
        self.analysis_frame, self.analysis_fig, self.analysis_ax, self.analysis_canvas = \
            create_matplotlib_figure(analysis_view_frame, "Analysis Results", "Time", "Position", figsize=(8, 6))
    
    def setup_results_tab(self, tab):
        """Set up the results tab"""
        # Create a frame for the tab content
        main_frame = ttk.Frame(tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure grid for three columns
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(0, weight=1)  # Plots row
        main_frame.rowconfigure(1, weight=0)  # Buttons row
        
        # Create visualization frames
        kymo_overlap_frame = ttk.LabelFrame(main_frame, text="Kymograph Overlap")
        kymo_overlap_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        plot_overlap_frame = ttk.LabelFrame(main_frame, text="Plot Overlap")
        plot_overlap_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        distance_frame = ttk.LabelFrame(main_frame, text="Distance Analysis")
        distance_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        
        # Create plotting areas
        kymo_overlap_frame, self.kymo_overlap_fig, self.kymo_overlap_ax, self.kymo_overlap_canvas = \
            create_plot_frame(kymo_overlap_frame, title="Kymograph with Trajectories", figsize=(4, 4))
        
        plot_overlap_frame, self.plot_overlap_fig, self.plot_overlap_ax, self.plot_overlap_canvas = \
            create_plot_frame(plot_overlap_frame, title="Traces with Events", figsize=(4, 4))
        
        distance_frame, self.distance_fig, self.distance_ax, self.distance_canvas = \
            create_plot_frame(distance_frame, title="DNAp-SSB Distance", figsize=(4, 4))
        
        # Initialize axes as off
        self.kymo_overlap_ax.axis('off')
        self.plot_overlap_ax.axis('off')
        self.distance_ax.axis('off')
        
        # Add buttons for each plot
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        
        # Create a buttons grid
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)
        
        # Kymograph Overlap buttons
        kymo_buttons = ttk.Frame(button_frame)
        kymo_buttons.grid(row=0, column=0, padx=5, pady=5)
        
        ttk.Button(kymo_buttons, text="Generate Overlap", 
                  command=self.plot_kymo_overlap).pack(side=tk.LEFT, padx=5)
        ttk.Button(kymo_buttons, text="Toggle Axes", 
                  command=lambda: self.toggle_axes(self.kymo_overlap_ax, self.kymo_overlap_canvas)).pack(side=tk.LEFT, padx=5)
        
        # Plot Overlap buttons
        plot_buttons = ttk.Frame(button_frame)
        plot_buttons.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Button(plot_buttons, text="Plot DNAp-SSB", 
                   command=self.plot_dnap_ssb).pack(side=tk.LEFT, padx=5)
        ttk.Button(plot_buttons, text="Toggle Axes", 
                  command=lambda: self.toggle_axes(self.plot_overlap_ax, self.plot_overlap_canvas)).pack(side=tk.LEFT, padx=5)
        
        # Distance Analysis buttons
        distance_buttons = ttk.Frame(button_frame)
        distance_buttons.grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Button(distance_buttons, text="Analyze Distance", 
                  command=self.analyze_distance).pack(side=tk.LEFT, padx=5)
        ttk.Button(distance_buttons, text="Toggle Axes", 
                  command=lambda: self.toggle_axes(self.distance_ax, self.distance_canvas)).pack(side=tk.LEFT, padx=5)
        
        # Add export all and navigation buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        
        ttk.Button(action_frame, text="Export All Results", 
                   command=self.export_all_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Previous: Analysis", 
                   command=lambda: self.notebook.select(2)).pack(side=tk.RIGHT, padx=5)
    
    def create_status_bar(self):
        """Create a status bar at the bottom of the window"""
        self.status_frame, self.status_var, self.progress_var, self.progress_bar = create_status_bar(self.master)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
    def refresh_previews(self):
        """Refresh the preview displays"""
        try:
            self.status_var.set("Refreshing displays...")
            self.master.update_idletasks()
            
            if self.img is not None:
                self.update_kymograph_display()
                self.status_var.set("Kymograph display refreshed")
            
            if self.trace is not None:
                self.update_preview_display()
                self.status_var.set("OT data display refreshed")
                
            if self.img is not None and self.trace is not None:
                self.status_var.set("All displays refreshed")
            elif self.img is None and self.trace is None:
                self.status_var.set("No data to display")
                
        except Exception as e:
            print(f"Error refreshing previews: {str(e)}")
            import traceback
            traceback.print_exc()
            self.status_var.set(f"Error refreshing displays: {str(e)}")
    
    # Data loading methods
    def load_kymo_file(self, filename):
        """Load kymograph file"""
        try:
            self.status_var.set(f"Loading {os.path.basename(filename)}...")
            self.master.update_idletasks()  # Force UI update
            
            # Handle different file types
            if filename.lower().endswith('.tdms'):
                # TDMS file handling
                metadata = TdmsFile.read_metadata(filename)
                self.width = metadata.properties['Pixels per line']
                self.px_size = float(metadata.properties['Scan Command.Scan Command.scanning_axes.0.pix_size_nm'])
                self.px_dwell_time = float(metadata.properties['Scan Command.PI Fast Scan Command.pixel_dwell_time_ms'])
                
                tdms_file = TdmsFile(filename)
                kymo_time = np.array([int(i) for i in tdms_file['Data']['Time (ms)'][:]])
                kymo_position = np.array([int(i) for i in tdms_file['Data']['Actual position X (um)'][:]])
                
                # Load channel data
                self.chn_r = np.array([int(i) for i in tdms_file['Data']['Pixel ch 1'][:]])
                self.chn_g = np.array([int(i) for i in tdms_file['Data']['Pixel ch 2'][:]])
                self.chn_b = np.array([int(i) for i in tdms_file['Data']['Pixel ch 3'][:]])
                
                height = len(kymo_time) / self.width
                self.time_per_line = kymo_time[-1] / height
                
                # Create RGB image
                self.img = np.vstack((self.chn_r, self.chn_g, self.chn_b)).T
                self.img = self.img.reshape((int(height), int(self.width), 3))
                self.img = self.img.transpose((1, 0, 2))
                self.img = self.img.astype(np.uint16)
                
            elif filename.lower().endswith(('.tif', '.tiff')):
                # TIFF file handling
                self.img = tifffile.imread(filename)
                
                # Convert to RGB if grayscale
                if len(self.img.shape) == 2:
                    self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
                    
                # Set default parameters
                self.width = self.img.shape[1]
                self.time_per_line = 1.0  # Default value
                self.px_size = 1.0  # Default value
                
            elif filename.lower().endswith('.h5'):
                # HDF5 file handling
                import h5py
                with h5py.File(filename, 'r') as f:
                    # Try to find image data in the file
                    if 'image' in f:
                        self.img = f['image'][:]
                    elif 'data' in f:
                        self.img = f['data'][:]
                    else:
                        # Try to find the first dataset that looks like an image
                        for key in f.keys():
                            if isinstance(f[key], h5py.Dataset) and len(f[key].shape) >= 2:
                                self.img = f[key][:]
                                break
                
                # Convert to RGB if grayscale
                self.trace = pd.read_csv(file_path)
            
            # Set alias for other methods
            self.trace_data = self.trace
            
            # Debug: Output the column names to help with debugging
            print(f"Original columns: {list(self.trace.columns)}")
            
            # Create a mapping dictionary - prioritizing standard input format
            column_map = {}
            
            # Time column
            if 'time' in self.trace.columns:
                column_map['time'] = 'time'
            elif 'Time' in self.trace.columns:
                column_map['Time'] = 'time'
                
            # Force column
            if 'force' in self.trace.columns:
                column_map['force'] = 'force'
            elif 'Force' in self.trace.columns:
                column_map['Force'] = 'force'
                
            # Position/Junction columns - these are crucial
            # Check for the specific column from the sample data
            if 'dsDNA_junction' in self.trace.columns:
                column_map['dsDNA_junction'] = 'junction_forward'
                print("Found dsDNA_junction column in data")
            elif 'ssDNA/dsDNA junction' in self.trace.columns:
                column_map['ssDNA/dsDNA junction'] = 'junction_forward'
                print("Found ssDNA/dsDNA junction column in data")
            elif 'junction_position' in self.trace.columns:
                column_map['junction_position'] = 'junction_forward'
                print("Found junction_position column in data")
            
            # EED column (if exists)
            if 'EED' in self.trace.columns:
                column_map['EED'] = 'EED'
                print("Found EED column in data")
                
            # Apply the column mapping
            self.trace = self.trace.rename(columns=column_map)
            print(f"After mapping columns: {list(self.trace.columns)}")
            
            # Create missing columns if needed
            if 'time' not in self.trace.columns:
                print("Creating time column")
                self.trace['time'] = np.arange(len(self.trace))
                
            if 'junction_forward' not in self.trace.columns:
                print("No junction_forward column found. Looking for alternatives...")
                # Try to find a suitable column to use
                if 'dsDNA_junction' in self.trace.columns:
                    print("Using dsDNA_junction column")
                    self.trace['junction_forward'] = self.trace['dsDNA_junction']
                elif 'basepairs_file' in self.trace.columns:
                    print("Using basepairs_file column")
                    self.trace['junction_forward'] = self.trace['basepairs_file']
                else:
                    # Try to find any numeric column that might be position data
                    numeric_cols = self.trace.select_dtypes(include=[np.number]).columns
                    position_candidates = [col for col in numeric_cols if 'position' in col.lower() or 'junction' in col.lower()]
                    
                    if position_candidates:
                        print(f"Using {position_candidates[0]} as junction_forward")
                        self.trace['junction_forward'] = self.trace[position_candidates[0]]
                    else:
                        # Create an empty column as a last resort
                        print("No suitable column found for junction_forward. Creating empty column.")
                        self.trace['junction_forward'] = np.zeros(len(self.trace))
                    
            # Create junction_reverse if not found
            if 'junction_reverse' not in self.trace.columns:
                if 'EED' in self.trace.columns and 'junction_forward' in self.trace.columns:
                    print("Creating junction_reverse from EED - junction_forward")
                    self.trace['junction_reverse'] = self.trace['EED'] - self.trace['junction_forward']
                else:
                    print("Creating junction_reverse as copy of junction_forward")
                    self.trace['junction_reverse'] = self.trace['junction_forward'].copy()
            
            # Update status
            self.status_var.set(f"Loaded trace file: {os.path.basename(file_path)}")
            
            # Print final columns
            print(f"Final columns: {list(self.trace.columns)}")
            print(f"Data sample: {self.trace[['time', 'junction_forward']].head()}")
            
            # Update the OT data preview
            self.update_preview_display()
            
            # Try to plot junction forward if possible
            try:
                self.plot_junction_forward()
            except Exception as e:
                print(f"Could not plot junction forward: {str(e)}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error loading trace file: {str(e)}")
            messagebox.showerror("Error", f"Failed to load trace file: {str(e)}")
            self.trace = None
    
    # Display update methods
    def update_kymograph_display(self):
        """Update kymograph display with the loaded image data"""
        try:
            if self.img is None:
                # No image loaded
                self.status_var.set("No kymograph loaded")
                return
                
            # Clear existing plot
            self.kymo_ax.clear()
            
            # Get the selected channel
            channel_idx = {'Red': 0, 'Green': 1, 'Blue': 2}[self.selected_channel]
            
            # Create a grayscale image from the selected channel
            img_channel = self.img[:, :, channel_idx].astype(np.float32)
            
            # Apply contrast enhancement
            p2, p98 = np.percentile(img_channel, (2, self.contrast_max))
            img_display = np.clip((img_channel - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
            
            # Display the image
            im = self.kymo_ax.imshow(img_display, cmap='gray', aspect='auto')
            
            # Only add colorbar on first display, not on refresh
            if not hasattr(self, 'kymo_colorbar'):
                self.kymo_colorbar = self.kymo_fig.colorbar(im, ax=self.kymo_ax, label='Intensity')
            
            # Set axis labels and title
            self.kymo_ax.set_xlabel('Time (pixels)')
            self.kymo_ax.set_ylabel('Position (pixels)')
            self.kymo_ax.set_title(f'Kymograph - {self.selected_channel} Channel')
            
            # Apply ROI if set
            if all(v is not None for v in self.roi_coords.values()):
                self.kymo_ax.axvline(x=self.roi_coords['start_x'], color='r', linestyle='--')
                self.kymo_ax.axvline(x=self.roi_coords['end_x'], color='r', linestyle='--')
                self.kymo_ax.axhline(y=self.roi_coords['start_y'], color='r', linestyle='--')
                self.kymo_ax.axhline(y=self.roi_coords['end_y'], color='r', linestyle='--')
            
            # Adjust the figure layout
            self.kymo_fig.tight_layout()
            
            # Draw the canvas
            self.kymo_canvas.draw()
            
            # Update status
            self.status_var.set(f"Kymograph displayed - {self.selected_channel} channel")
            
            # Signal that the image has been loaded successfully for other methods to use
            self.kymograph_displayed = True
            
        except Exception as e:
            print(f"Error in update_kymograph_display: {str(e)}")
            import traceback
            traceback.print_exc()
            self.status_var.set(f"Error displaying kymograph: {str(e)}")
    
    def update_preview_display(self):
        """Update OT data preview"""
        if self.trace is not None:
            try:
                self.preview_ax.clear()
                
                # Check if we have the required columns
                if 'time' not in self.trace.columns:
                    print("Warning: 'time' column not found in trace data")
                    return
                
                # Use junction_forward for position data if available
                if 'junction_forward' in self.trace.columns:
                    self.preview_ax.plot(self.trace['time'], self.trace['junction_forward'], 'b-', lw=1.5)
                    y_label = "Junction Position"
                elif 'position' in self.trace.columns:
                    self.preview_ax.plot(self.trace['time'], self.trace['position'], 'b-', lw=1.5)
                    y_label = "Position"
                else:
                    # Try to find any numeric column to plot
                    numeric_cols = self.trace.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 1 and 'time' in numeric_cols:
                        # Use the first non-time numeric column
                        plot_col = [col for col in numeric_cols if col != 'time'][0]
                        self.preview_ax.plot(self.trace['time'], self.trace[plot_col], 'b-', lw=1.5)
                        y_label = plot_col
                    else:
                        print("Warning: No suitable column found for plotting")
                        return
                
                # Add grid and labels
                self.preview_ax.grid(True, alpha=0.3)
                self.preview_ax.set_xlabel("Time")
                self.preview_ax.set_ylabel(y_label)
                self.preview_ax.set_title("OT Data Preview")
                
                # Make sure axes scales are appropriate
                self.preview_ax.relim()
                self.preview_ax.autoscale_view()
                
                # Update the canvas
                self.preview_fig.tight_layout()
                self.preview_canvas.draw()
                
                # Update status
                self.status_var.set("OT data preview updated")
                
            except Exception as e:
                print(f"Error updating preview display: {str(e)}")
                import traceback
                traceback.print_exc()
    
    # Channel and ROI methods
    def update_channel(self, channel):
        """Update selected channel"""
        self.selected_channel = channel
        self.update_kymograph_display()
    
    def apply_roi(self, x_left, x_right, y_top, y_bottom):
        """Apply ROI to kymograph"""
        self.roi_coords = {
            'start_x': x_left,
            'end_x': x_right,
            'start_y': y_top,
            'end_y': y_bottom
        }
        self.update_kymograph_display()
    
    # Analysis methods
    def detect_trajectories(self):
        """Detect SSB trajectories"""
        if self.img is None:
            tk.messagebox.showerror("Error", "No kymograph loaded")
            return
            
        try:
            # Get parameters
            line_width = int(self.param_entries["Line Width"].get())
            pixel_threshold = int(self.param_entries["Pixel Threshold"].get())
            window = int(self.param_entries["Window Size"].get())
            
            # Check if ROI is defined
            if not all(v is not None for v in self.roi_coords.values()):
                # If no ROI set, use the whole image
                self.roi_coords = {
                    'start_x': 0,
                    'end_x': self.img.shape[1],
                    'start_y': 0,
                    'end_y': self.img.shape[0]
                }
                self.status_var.set("No ROI set, using entire image")
            
            # Extract ROI
            roi = self.img[int(self.roi_coords['start_y']):int(self.roi_coords['end_y']),
                         int(self.roi_coords['start_x']):int(self.roi_coords['end_x'])]
            
            # Check if ROI is valid
            if roi.size == 0:
                tk.messagebox.showerror("Error", "Invalid ROI selection. Please set a valid ROI.")
                return
            
            # Update status
            self.status_var.set("Detecting trajectories...")
            self.master.update_idletasks()  # Force UI update
            
            # Detect trajectories
            self.traces = lk.track_greedy(roi, line_width=line_width,
                                        pixel_threshold=pixel_threshold, window=window)
                                        
            # Filter and refine
            if self.traces:
                self.traces = lk.filter_lines(self.traces, 3)
                self.traces = lk.refine_lines_centroid(self.traces, line_width=line_width)
                
                # Plot detected trajectories
                self.plot_trajectories()
                
                # Update status
                self.status_var.set(f"Detected {len(self.traces)} trajectories")
            else:
                tk.messagebox.showinfo("Info", "No trajectories detected with current parameters. Try adjusting parameters.")
                self.status_var.set("No trajectories detected")
            
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to detect trajectories: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
    
    def plot_trajectories(self):
        """Plot detected trajectories"""
        if self.traces is None:
            tk.messagebox.showerror("Error", "No trajectories detected")
            return
            
        try:
            # Store axes state (on or off)
            axes_visible = self.plot_overlap_ax.get_xaxis().get_visible() and self.plot_overlap_ax.get_yaxis().get_visible()
            
            # Clear previous plot
            self.plot_overlap_ax.clear()
            
            # Extract ROI if set, otherwise use the whole image
            if all(v is not None for v in self.roi_coords.values()):
                roi_img = self.img[int(self.roi_coords['start_y']):int(self.roi_coords['end_y']),
                                int(self.roi_coords['start_x']):int(self.roi_coords['end_x'])]
                self.plot_overlap_ax.imshow(roi_img, aspect="auto", vmax=50, cmap='gray')
            else:
                self.plot_overlap_ax.imshow(self.img, aspect="auto", vmax=50, cmap='gray')
            
            # Plot each trajectory
            for trace in self.traces:
                # Apply smoothing if trace has enough points
                if len(trace.time_idx) > 31:  # Need enough points for Savitzky-Golay filter
                    time_idx_smooth = savgol_filter(np.array(trace.time_idx), 31, 3)
                    coordinate_idx_smooth = savgol_filter(np.array(trace.coordinate_idx), 31, 3)
                else:
                    time_idx_smooth = np.array(trace.time_idx)
                    coordinate_idx_smooth = np.array(trace.coordinate_idx)
                
                # Plot the trace
                self.plot_overlap_ax.plot(time_idx_smooth, coordinate_idx_smooth, 'r-', linewidth=1.5)
            
            # Set plot labels
            self.plot_overlap_ax.set_title("Detected Trajectories")
            self.plot_overlap_ax.set_xlabel("Time (px)")
            self.plot_overlap_ax.set_ylabel("Position (px)")
            
            # Keep axes hidden if they were hidden before
            if not axes_visible:
                self.plot_overlap_ax.axis('off')
                
            # Update the canvas
            self.plot_overlap_canvas.draw()
            
            # Update status
            self.status_var.set(f"Displayed {len(self.traces)} trajectories")
            
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to plot trajectories: {str(e)}")
    
    def plot_junction_forward(self):
        """Plot junction forward analysis"""
        try:
            if self.trace is None:
                tk.messagebox.showerror("Error", "No trace data loaded")
                return
                
            # Make sure we have the required columns
            if 'time' not in self.trace.columns:
                raise ValueError("Time column not found in trace data")
                
            if 'junction_forward' not in self.trace.columns:
                raise ValueError("Junction forward column not found in trace data")
            
            # Get the data
            time_data = self.trace['time'].values
            junction_data = self.trace['junction_forward'].values
            
            # Check for invalid data
            if np.isnan(junction_data).any() or np.isinf(junction_data).any():
                print("Warning: NaN or Inf values found in junction data. Cleaning...")
                # Replace NaN/Inf with zeros or previous valid values
                junction_data = np.nan_to_num(junction_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Clear previous plot
            self.preview_ax.clear()
            
            # Plot the data
            self.preview_ax.plot(time_data, junction_data, 'b-', lw=2)
            
            # Add grid and labels
            self.preview_ax.grid(True, alpha=0.3)
            self.preview_ax.set_xlabel("Time (s)")
            self.preview_ax.set_ylabel("Junction Position")
            self.preview_ax.set_title("Forward Junction Position")
            
            # Make sure axes scales are appropriate
            self.preview_ax.relim()
            self.preview_ax.autoscale_view()
            
            # Layout adjustment
            self.preview_fig.tight_layout()
            
            # Update the canvas
            self.preview_canvas.draw()
            
            # Update status
            self.status_var.set("Displaying junction forward plot")
            
            # Set flag for successful plot
            self.junction_plot_displayed = True
            
            return True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in plot_junction_forward: {str(e)}")
            tk.messagebox.showerror("Error", f"Failed to plot junction forward: {str(e)}\n\nAvailable columns: {list(self.trace.columns)}")
            self.junction_plot_displayed = False
            return False
    
    def plot_junction_reverse(self):
        """Plot junction reverse analysis"""
        try:
            if self.trace is None:
                tk.messagebox.showerror("Error", "No trace data loaded")
                return
                
            # Make sure we have the required columns
            if 'time' not in self.trace.columns:
                raise ValueError("Time column not found in trace data")
                
            if 'junction_reverse' not in self.trace.columns:
                raise ValueError("Junction reverse column not found in trace data")
            
            # Get the data
            time_data = self.trace['time'].values
            junction_data = self.trace['junction_reverse'].values
            
            # Check for invalid data
            if np.isnan(junction_data).any() or np.isinf(junction_data).any():
                print("Warning: NaN or Inf values found in junction data. Cleaning...")
                # Replace NaN/Inf with zeros or previous valid values
                junction_data = np.nan_to_num(junction_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Clear previous plot
            self.preview_ax.clear()
            
            # Plot the data
            self.preview_ax.plot(time_data, junction_data, 'r-', lw=2)
            
            # Add grid and labels
            self.preview_ax.grid(True, alpha=0.3)
            self.preview_ax.set_xlabel("Time (s)")
            self.preview_ax.set_ylabel("Junction Position")
            self.preview_ax.set_title("Reverse Junction Position")
            
            # Make sure axes scales are appropriate
            self.preview_ax.relim()
            self.preview_ax.autoscale_view()
            
            # Layout adjustment
            self.preview_fig.tight_layout()
            
            # Update the canvas
            self.preview_canvas.draw()
            
            # Update status
            self.status_var.set("Displaying junction reverse plot")
            
            # Set flag for successful plot
            self.junction_plot_displayed = True
            
            return True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in plot_junction_reverse: {str(e)}")
            tk.messagebox.showerror("Error", f"Failed to plot junction reverse: {str(e)}\n\nAvailable columns: {list(self.trace.columns)}")
            self.junction_plot_displayed = False
            return False
    
    def plot_kymo_overlap(self):
        """Plot kymograph with detected trajectories overlaid"""
        if self.img is None or not hasattr(self, 'traces') or not self.traces:
            messagebox.showinfo("Information", "Please load kymograph and detect trajectories first.")
            return
            
        try:
            self.status_var.set("Creating kymograph overlap visualization...")
            self.master.update_idletasks()
            
            # Store axes state (on or off)
            axes_visible = self.kymo_overlap_ax.get_xaxis().get_visible() and self.kymo_overlap_ax.get_yaxis().get_visible()
            
            # Clear previous plot
            self.kymo_overlap_ax.clear()
            
            # Create a copy of the image for display
            if self.img.shape[2] == 3:
                # If RGB, select only the current channel
                channel_idx = {'Red': 0, 'Green': 1, 'Blue': 2}[self.selected_channel]
                display_img = self.img[:, :, channel_idx].copy()
            else:
                display_img = self.img.copy()
                
            # Normalize to 8-bit for display
            display_img = display_img.astype(np.float32)
            p2, p98 = np.percentile(display_img, (2, 98))
            display_img = np.clip((display_img - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
            
            # Convert to RGB for colored overlays
            if len(display_img.shape) == 2:
                display_rgb = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)
            else:
                display_rgb = display_img
            
            # Apply ROI if set
            if all(v is not None for v in self.roi_coords.values()):
                roi = display_rgb[int(self.roi_coords['start_y']):int(self.roi_coords['end_y']),
                               int(self.roi_coords['start_x']):int(self.roi_coords['end_x'])]
            else:
                roi = display_rgb
                
            # Display the kymograph
            self.kymo_overlap_ax.imshow(roi, aspect='auto')
            
            # Get calibration parameters
            x_offset = 0
            y_offset = 0
            time_scale = 1.0
            position_scale = 1.0
            
            if hasattr(self, 'param_entries'):
                try:
                    x_offset = float(self.param_entries["X Offset"].get())
                    y_offset = float(self.param_entries["Y Offset"].get())
                    time_scale = float(self.param_entries["Time Scale"].get())
                    position_scale = float(self.param_entries["Position Scale"].get())
                except:
                    pass
                    
            # Apply ROI offset if using ROI
            if all(v is not None for v in self.roi_coords.values()):
                roi_x_offset = self.roi_coords['start_x']
                roi_y_offset = self.roi_coords['start_y']
            else:
                roi_x_offset = 0
                roi_y_offset = 0
                
            # Plot each detected trajectory
            colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(self.traces))))
            
            for i, trace in enumerate(self.traces):
                # Get raw trajectory data
                time_idx = np.array(trace.time_idx)
                coordinate_idx = np.array(trace.coordinate_idx)
                
                # Apply calibration and ROI offset
                adjusted_time = (time_idx - roi_x_offset)
                adjusted_pos = (coordinate_idx - roi_y_offset)
                
                # Apply smoothing if enough points
                if len(time_idx) > 31:
                    window_size = min(31, len(time_idx) - (len(time_idx) % 2) - 1)  # Make sure it's odd
                    if window_size >= 3:
                        adjusted_time_smooth = savgol_filter(adjusted_time, window_size, 3)
                        adjusted_pos_smooth = savgol_filter(adjusted_pos, window_size, 3)
                        
                        # Plot smoothed trajectory
                        color = colors[i % len(colors)]
                        self.kymo_overlap_ax.plot(adjusted_time_smooth, adjusted_pos_smooth, 
                                               color=color, linewidth=2, label=f'SSB {i+1}')
                    else:
                        # Not enough points for smoothing
                        color = colors[i % len(colors)]
                        self.kymo_overlap_ax.plot(adjusted_time, adjusted_pos, 
                                               color=color, linewidth=2, label=f'SSB {i+1}')
                else:
                    # Not enough points for smoothing
                    color = colors[i % len(colors)]
                    self.kymo_overlap_ax.plot(adjusted_time, adjusted_pos, 
                                           color=color, linewidth=2, label=f'SSB {i+1}')
            
            # Add grid and labels
            self.kymo_overlap_ax.grid(False)  # Usually better without grid over an image
            self.kymo_overlap_ax.set_xlabel('Time (px)')
            self.kymo_overlap_ax.set_ylabel('Position (px)')
            self.kymo_overlap_ax.set_title('Kymograph with Detected Trajectories')
            
            # Add legend if not too many traces
            if len(self.traces) <= 10:
                self.kymo_overlap_ax.legend(loc='upper right', fontsize='small')
            
            # Keep axes hidden if they were hidden before
            if not axes_visible:
                self.kymo_overlap_ax.axis('off')
                
            # Update canvas
            self.kymo_overlap_canvas.draw()
            
            # Update status
            self.status_var.set(f"Kymograph overlap with {len(self.traces)} trajectories displayed")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to plot kymograph overlap: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
    
    def analyze_distance(self):
        """Analyze distance between DNAp and SSB"""
        if not hasattr(self, 'traces') or not self.traces or self.trace is None:
            messagebox.showinfo("Information", "Please load both kymograph and trace files, and detect SSB trajectories first.")
            return
            
        try:
            # Calculate distance data if not already done
            if not hasattr(self, 'distance_data') or not self.distance_data:
                distances = self.calculate_dnap_ssb_distance(plot_result=False)
                if not distances or len(distances) == 0:
                    return
            else:
                distances = self.distance_data
                
            # Update status
            self.status_var.set("Analyzing DNAp-SSB distance...")
            
            # Create a graph showing the distance over time
            self.distance_ax.clear()
            
            # Get color map for multiple traces
            colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(distances))))
            
            # Plot each trace with different color
            for i, distance_data in enumerate(distances):
                time = distance_data['ssb_time']
                distance = distance_data['distance']
                color = colors[i % len(colors)]
                
                # Plot raw data points with lower opacity
                self.distance_ax.scatter(time, distance, color=color, alpha=0.3, s=10, label=f'SSB {i+1} Raw')
                
                # Smooth the distance data if there are enough points
                if len(time) >= 5:
                    window_size = min(21, len(time) - (len(time) % 2) - 1)  # Ensure odd length
                    if window_size >= 3:
                        smoothed_distance = savgol_filter(distance, window_size, 3)
                        self.distance_ax.plot(time, smoothed_distance, color=color, linewidth=2, label=f'SSB {i+1} Smooth')
                else:
                    # For very short traces, just connect the dots
                    self.distance_ax.plot(time, distance, color=color, linewidth=2, label=f'SSB {i+1}')
            
            # Add horizontal line at y=0 to indicate when DNAp passes SSB
            self.distance_ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            # Add text annotation explaining the sign of distance
            self.distance_ax.text(0.02, 0.02, "Negative: DNAp ahead of SSB\nPositive: SSB ahead of DNAp", 
                        transform=self.distance_ax.transAxes, fontsize=8, 
                        bbox=dict(facecolor='white', alpha=0.7))
            
            # Set labels and title
            self.distance_ax.set_xlabel('Time (s)')
            self.distance_ax.set_ylabel('Distance (μm)')
            self.distance_ax.set_title('DNAp-SSB Distance Analysis')
            
            # Add grid for readability
            self.distance_ax.grid(True, alpha=0.3)
            
            # Add legend if not too many traces
            if len(distances) <= 5:
                self.distance_ax.legend(loc='best', fontsize='small')
            
            # Show axes
            self.distance_ax.axis('on')
            
            # Update the canvas
            self.distance_fig.tight_layout()
            self.distance_canvas.draw()
            
            # Update status
            self.status_var.set(f"Distance analysis complete for {len(distances)} SSB trajectories")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze distance: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
    
    def show_kymograph_popup(self):
        """Show kymograph in a popup window"""
        if self.img is None:
            messagebox.showerror("Error", "No kymograph loaded")
            return
            
        # Create popup window
        popup = tk.Toplevel(self.master)
        popup.title("Kymograph Image")
        popup.geometry("800x600")
        
        # Create matplotlib figure
        fig = Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111)
        ax.imshow(self.img.astype('uint16'), vmax=255)
        ax.set_xlabel("Time (px)")
        ax.set_ylabel("Position (px)")
        ax.set_title("Kymograph")
        
        # Add the figure to the popup
        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar_frame = ttk.Frame(popup)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

    def load_kymograph(self):
        """Load kymograph file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            self.img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if self.img is None:
                # Try using tifffile if OpenCV fails
                self.img = tifffile.imread(file_path)
            
            # RGB conversion if needed
            if len(self.img.shape) == 2:
                self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
            
            # Reset ROI coordinates
            self.roi_coords = {
                'start_x': 0,
                'end_x': self.img.shape[1],
                'start_y': 0,
                'end_y': self.img.shape[0]
            }
            
            # Update status
            self.status_var.set(f"Loaded kymograph: {os.path.basename(file_path)}")
            
            # Update the display
            self.update_kymograph_display()
            
            # Show the kymograph in a popup window
            self.show_kymograph_popup()
            
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to load kymograph file: {str(e)}")

    def save_kymograph_image(self):
        """Save kymograph as an image"""
        if self.img is None:
            messagebox.showerror("Error", "No kymograph loaded")
            return

        try:
            file_path = filedialog.asksaveasfilename(
                filetypes=[("TIFF files", "*.tif *.tiff"), ("PNG files", "*.png"), ("All files", "*.*")],
                defaultextension=".tif",
                initialfile="kymograph"
            )
            if not file_path:
                return
            
            # Save based on file extension
            if file_path.lower().endswith(('.png')):
                plt.imsave(file_path, self.img)
            else:  # Default to TIFF
                tifffile.imwrite(file_path, self.img)
                
            messagebox.showinfo("Success", f"Kymograph saved to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save kymograph: {str(e)}")

    def filter_trajectories(self):
        """Filter trajectories based on user input"""
        if self.traces is None:
            tk.messagebox.showerror("Error", "No trajectories to filter")
            return

        try:
            # Create a dialog for filtering
            filter_dialog = tk.Toplevel(self.master)
            filter_dialog.title("Filter Trajectories")
            filter_dialog.geometry("400x300")
            filter_dialog.transient(self.master)
            filter_dialog.grab_set()  # Make dialog modal
            
            # Create filtering options
            ttk.Label(filter_dialog, text="Minimum Trajectory Length:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
            min_length_var = tk.StringVar(value="5")
            ttk.Entry(filter_dialog, textvariable=min_length_var, width=10).grid(row=0, column=1, padx=10, pady=10)
            
            ttk.Label(filter_dialog, text="Minimum Distance (pixels):").grid(row=1, column=0, padx=10, pady=10, sticky="w")
            min_distance_var = tk.StringVar(value="10")
            ttk.Entry(filter_dialog, textvariable=min_distance_var, width=10).grid(row=1, column=1, padx=10, pady=10)
            
            ttk.Label(filter_dialog, text="Maximum Intensity:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
            max_intensity_var = tk.StringVar(value="255")
            ttk.Entry(filter_dialog, textvariable=max_intensity_var, width=10).grid(row=2, column=1, padx=10, pady=10)
            
            # Function to apply the filter
            def apply_filter():
                try:
                    min_length = int(min_length_var.get())
                    min_distance = float(min_distance_var.get())
                    max_intensity = float(max_intensity_var.get())
                    
                    # Filter the trajectories
                    original_count = len(self.traces)
                    filtered_traces = []
                    
                    for trace in self.traces:
                        # Check if trajectory meets criteria
                        if (len(trace.time_idx) >= min_length and
                            abs(trace.coordinate_idx[-1] - trace.coordinate_idx[0]) >= min_distance):
                            filtered_traces.append(trace)
                    
                    # Update the traces
                    self.traces = filtered_traces
                    
                    # Show result
                    tk.messagebox.showinfo("Filter Applied", 
                                          f"Filtered trajectories from {original_count} to {len(self.traces)}")
                    
                    # Close dialog and update display
                    filter_dialog.destroy()
                    self.plot_trajectories()
                    
                except ValueError as e:
                    tk.messagebox.showerror("Error", f"Invalid input: {str(e)}")
            
            # Buttons
            button_frame = ttk.Frame(filter_dialog)
            button_frame.grid(row=3, column=0, columnspan=2, pady=20)
            
            ttk.Button(button_frame, text="Apply", command=apply_filter).pack(side=tk.LEFT, padx=10)
            ttk.Button(button_frame, text="Cancel", command=filter_dialog.destroy).pack(side=tk.LEFT, padx=10)
            
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to filter trajectories: {str(e)}")

    def toggle_axes(self, ax, canvas):
        """Toggle axes visibility"""
        if ax.get_xaxis().get_visible() and ax.get_yaxis().get_visible():
            ax.axis('off')
            canvas.draw()
        else:
            ax.axis('on')
            canvas.draw()

    def debug_ui_components(self):
        """Print debug information about UI components and data"""
        print("\n===== DEBUGGING INFORMATION =====")
        print(f"Python version: {sys.version}")
        
        # Package versions
        try:
            print(f"NumPy version: {np.__version__}")
            print(f"Pandas version: {pd.__version__}")
            import matplotlib as mpl
            print(f"Matplotlib version: {mpl.__version__}")
            print(f"CV2 version: {cv2.__version__}")
        except Exception as e:
            print(f"Error getting package versions: {e}")
        
        # Canvas information
        if hasattr(self, 'kymo_canvas') and self.kymo_canvas is not None:
            print(f"\nKymograph canvas exists: {self.kymo_canvas}")
            print(f"Kymograph canvas packed: {self.kymo_canvas.get_tk_widget().winfo_ismapped()}")
        else:
            print("\nKymograph canvas does not exist")
            
        if hasattr(self, 'preview_canvas') and self.preview_canvas is not None:
            print(f"\nPreview canvas exists: {self.preview_canvas}")
            print(f"Preview canvas packed: {self.preview_canvas.get_tk_widget().winfo_ismapped()}")
        else:
            print("\nPreview canvas does not exist")
        
        # Image information
        if hasattr(self, 'img') and self.img is not None:
            print(f"\nImage loaded: Yes")
            print(f"Image shape: {self.img.shape}")
            print(f"Image type: {type(self.img)}")
            print(f"Image data type: {self.img.dtype}")
            print(f"Image min value: {np.min(self.img)}")
            print(f"Image max value: {np.max(self.img)}")
            print(f"Kymograph displayed: {getattr(self, 'kymograph_displayed', False)}")
        else:
            print("\nNo image loaded")
        
        # Trace information
        if hasattr(self, 'trace') and self.trace is not None:
            print(f"\nTrace loaded: Yes")
            print(f"Trace columns: {list(self.trace.columns)}")
            print(f"Trace length: {len(self.trace)}")
            print(f"Trace head: \n{self.trace.head()}")
            print(f"Junction plot displayed: {getattr(self, 'junction_plot_displayed', False)}")
            
            # Check for problematic values in the junction columns
            if 'junction_forward' in self.trace.columns:
                jf = self.trace['junction_forward']
                print(f"junction_forward min: {jf.min()}, max: {jf.max()}")
                print(f"NaN values in junction_forward: {jf.isna().sum()}")
                print(f"Inf values in junction_forward: {np.isinf(jf).sum()}")
        else:
            print("\nNo trace loaded")
            
        # ROI information
        print(f"\nROI coordinates: {self.roi_coords}")
        
        # Status information
        print(f"\nCurrent status: {self.status_var.get()}")
        
        print("===== END DEBUGGING INFO =====\n")
        
        # Display this information in a message box as well
        debug_info = f"Image: {'Yes' if hasattr(self, 'img') and self.img is not None else 'No'}\n"
        debug_info += f"Image shape: {getattr(self.img, 'shape', 'N/A')}\n"
        debug_info += f"Trace: {'Yes' if hasattr(self, 'trace') and self.trace is not None else 'No'}\n"
        if hasattr(self, 'trace') and self.trace is not None:
            debug_info += f"Trace columns: {list(self.trace.columns)}\n"
        debug_info += f"Kymograph displayed: {getattr(self, 'kymograph_displayed', False)}\n"
        debug_info += f"Junction plot displayed: {getattr(self, 'junction_plot_displayed', False)}\n"
        
        messagebox.showinfo("Debug Information", debug_info) 

    def export_all_results(self):
        """Export all analysis results to a directory"""
        try:
            # Ask user for the directory to save results
            output_dir = filedialog.askdirectory(
                title="Select Output Directory for All Results"
            )
            
            if not output_dir:
                return
                
            # Update status
            self.status_var.set("Exporting all results...")
            self.master.update_idletasks()  # Force UI update
            
            # Create a base filename from the input files
            kymo_filename = os.path.basename(self.kymo_entry.get()) if self.kymo_entry.get() else "kymograph"
            trace_filename = os.path.basename(self.ot_entry.get()) if self.ot_entry.get() else "trace"
            base_name = f"{os.path.splitext(kymo_filename)[0]}_{os.path.splitext(trace_filename)[0]}"
            
            # Export kymograph image if available
            if self.img is not None:
                # Save full kymograph image
                plt.figure(figsize=(10, 6))
                plt.imshow(self.img)
                plt.title("Kymograph")
                plt.xlabel("Time (px)")
                plt.ylabel("Position (px)")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{base_name}_kymograph.png"), dpi=300)
                plt.close()
                
                # Save individual channels
                for i, channel in enumerate(['Red', 'Green', 'Blue']):
                    plt.figure(figsize=(10, 6))
                    plt.imshow(self.img[:, :, i], cmap='gray')
                    plt.title(f"Kymograph - {channel} Channel")
                    plt.xlabel("Time (px)")
                    plt.ylabel("Position (px)")
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"{base_name}_kymograph_{channel.lower()}.png"), dpi=300)
                    plt.close()
            
            # Export trace data if available
            if hasattr(self, 'trace') and self.trace is not None:
                self.trace.to_csv(os.path.join(output_dir, f"{base_name}_dnap_trace.csv"), index=False)
            
            # Export SSB trajectory data if available
            if hasattr(self, 'traces') and self.traces is not None:
                for i, trace in enumerate(self.traces):
                    if isinstance(trace, pd.DataFrame):
                        trace.to_csv(os.path.join(output_dir, f"{base_name}_ssb_trace_{i+1}.csv"), index=False)
                    else:
                        # If trace is not a DataFrame, try to convert it
                        try:
                            df = pd.DataFrame({
                                'Time': [t for t in trace.time_idx],
                                'Position': [p for p in trace.coordinate_idx]
                            })
                            df.to_csv(os.path.join(output_dir, f"{base_name}_ssb_trace_{i+1}.csv"), index=False)
                        except Exception as e:
                            print(f"Failed to export trace {i+1}: {e}")
            
            # Export distance data if available
            if hasattr(self, 'distance_data') and self.distance_data is not None:
                for i, distance_data in enumerate(self.distance_data):
                    if isinstance(distance_data, dict):
                        df = pd.DataFrame({
                            'Time': distance_data['ssb_time'],
                            'SSB_Position': distance_data['ssb_pos'],
                            'DNAp_Position': distance_data['dnap_pos'],
                            'Distance': distance_data['distance']
                        })
                        df.to_csv(os.path.join(output_dir, f"{base_name}_distance_{i+1}.csv"), index=False)
            
            # Export segmented data if available
            if hasattr(self, 'segmented_data') and self.segmented_data is not None:
                self.segmented_data.to_csv(os.path.join(output_dir, f"{base_name}_segmented_data.csv"), index=False)
            
            # Export plots from the results tab
            # Kymograph Overlap
            if hasattr(self, 'kymo_overlap_fig') and self.kymo_overlap_fig is not None:
                self.kymo_overlap_fig.savefig(os.path.join(output_dir, f"{base_name}_kymo_overlap.png"), dpi=300)
            
            # Plot Overlap
            if hasattr(self, 'plot_overlap_fig') and self.plot_overlap_fig is not None:
                self.plot_overlap_fig.savefig(os.path.join(output_dir, f"{base_name}_plot_overlap.png"), dpi=300)
            
            # Distance Analysis
            if hasattr(self, 'distance_fig') and self.distance_fig is not None:
                self.distance_fig.savefig(os.path.join(output_dir, f"{base_name}_distance_analysis.png"), dpi=300)
            
            # Create a README file with export information
            with open(os.path.join(output_dir, f"{base_name}_README.txt"), 'w') as f:
                f.write(f"Kymograph Analyzer Export\n")
                f.write(f"========================\n\n")
                f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Input files:\n")
                f.write(f"  Kymograph: {self.kymo_entry.get()}\n")
                f.write(f"  Trace: {self.ot_entry.get()}\n\n")
                f.write(f"Exported files:\n")
                f.write(f"  Kymograph images: *_kymograph*.png\n")
                f.write(f"  DNAp trace data: *_dnap_trace.csv\n")
                f.write(f"  SSB trace data: *_ssb_trace_*.csv\n")
                f.write(f"  Distance data: *_distance_*.csv\n")
                f.write(f"  Segmented data: *_segmented_data.csv\n")
                f.write(f"  Analysis plots: *_kymo_overlap.png, *_plot_overlap.png, *_distance_analysis.png\n\n")
                f.write(f"Analysis parameters:\n")
                if hasattr(self, 'param_entries') and self.param_entries is not None:
                    for key, entry in self.param_entries.items():
                        try:
                            f.write(f"  {key}: {entry.get()}\n")
                        except:
                            pass
            
            self.status_var.set(f"Export complete: {len(os.listdir(output_dir))} files saved to {output_dir}")
            messagebox.showinfo("Export Complete", f"All results exported to:\n{output_dir}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export all results: {str(e)}")
            self.status_var.set(f"Export error: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_contrast(self, *args):
        """Update contrast based on slider value"""
        try:
            self.contrast_max = self.contrast_var.get()
            # Update the kymograph display if an image is loaded
            if self.img is not None:
                self.update_kymograph_display()
        except Exception as e:
            print(f"Error updating contrast: {e}")

    def force_image_analyzer(self):
        """Analyze force-image correlation"""
        if not self.kymo_data or not self.trace_data:
            messagebox.showwarning("Warning", "Please load both kymograph and trace data first.")
            return
            
        try:
            self.status_var.set("Analyzing force-image correlation...")
            # Implementation would go here in a real application
            messagebox.showinfo("Info", "Force-image correlation analysis not implemented in this demo.")
            self.status_var.set("Ready")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to analyze force-image correlation: {str(e)}")
            
    def display_intensity_with_unwinding(self):
        """Display intensity profile with unwinding data"""
        if not self.kymo_data:
            messagebox.showwarning("Warning", "Please load kymograph data first.")
            return
            
        try:
            self.status_var.set("Generating intensity profile...")
            # Implementation would go here in a real application
            messagebox.showinfo("Info", "Intensity profile display not implemented in this demo.")
            self.status_var.set("Ready")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to display intensity profile: {str(e)}")
            
    def plot_junction_forward(self):
        """Plot junction in forward direction"""
        if not self.kymo_data or not self.trace_data:
            messagebox.showwarning("Warning", "Please load both kymograph and trace data first.")
            return
            
        try:
            self.status_var.set("Plotting junction (forward)...")
            # Implementation would go here in a real application
            messagebox.showinfo("Info", "Junction forward plotting not implemented in this demo.")
            self.status_var.set("Ready")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to plot junction: {str(e)}")
            
    def plot_junction_reverse(self):
        """Plot junction in reverse direction"""
        if not self.kymo_data or not self.trace_data:
            messagebox.showwarning("Warning", "Please load both kymograph and trace data first.")
            return
            
        try:
            self.status_var.set("Plotting junction (reverse)...")
            # Implementation would go here in a real application
            messagebox.showinfo("Info", "Junction reverse plotting not implemented in this demo.")
            self.status_var.set("Ready")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to plot junction: {str(e)}")
            
    def save_kymograph_image(self):
        """Save the current kymograph image"""
        if not self.kymo_data:
            messagebox.showwarning("Warning", "No kymograph loaded to save.")
            return
            
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            if not filename:
                return
                
            self.status_var.set(f"Saving kymograph to {filename}...")
            # Implementation would go here in a real application
            messagebox.showinfo("Info", f"Kymograph would be saved to {filename} (not implemented in this demo).")
            self.status_var.set("Ready")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to save kymograph: {str(e)}")

    def calculate_dnap_ssb_distance(self, plot_result=True):
        """Calculate the distance between DNAp and SSB trajectories"""
        if not hasattr(self, 'traces') or not self.traces or self.trace is None:
            messagebox.showinfo("Information", "Please load both kymograph and trace files, and detect SSB trajectories first.")
            return None
            
        try:
            self.status_var.set("Calculating DNAp-SSB distance...")
            self.master.update_idletasks()
            
            # Get calibration parameters
            x_offset = float(self.param_entries["X Offset"].get())
            y_offset = float(self.param_entries["Y Offset"].get())
            time_scale = float(self.param_entries["Time Scale"].get())
            position_scale = float(self.param_entries["Position Scale"].get())
            
            # Create a list to store distance data for each SSB
            all_distances = []
            
            # For each SSB trajectory
            for trace_idx, ssb_trace in enumerate(self.traces):
                # Extract SSB trajectory data
                ssb_time = np.array(ssb_trace.time_idx)
                ssb_pos = np.array(ssb_trace.coordinate_idx)
                
                # Apply calibration
                ssb_time_calibrated = ssb_time * time_scale + x_offset
                ssb_pos_calibrated = ssb_pos * position_scale + y_offset
                
                # Interpolate DNAp trace to match SSB time points
                dnap_time = self.trace['time'].values
                dnap_pos = self.trace['junction_forward'].values
                
                # Create interpolation function for DNAp
                if len(dnap_time) > 1:
                    dnap_interp = interp1d(dnap_time, dnap_pos, bounds_error=False, fill_value="extrapolate")
                    
                    # Get DNAp positions at SSB time points
                    dnap_pos_at_ssb_times = dnap_interp(ssb_time_calibrated)
                    
                    # Calculate distance
                    distance = dnap_pos_at_ssb_times - ssb_pos_calibrated
                    
                    # Store distance data
                    distance_data = {
                        'ssb_time': ssb_time_calibrated,
                        'ssb_pos': ssb_pos_calibrated,
                        'dnap_pos': dnap_pos_at_ssb_times,
                        'distance': distance
                    }
                    all_distances.append(distance_data)
            
            # Store the distance data
            self.distance_data = all_distances
            
            # Plot the results if requested
            if plot_result and all_distances:
                self.analyze_distance()
                
            self.status_var.set(f"Calculated distance for {len(all_distances)} SSB trajectories")
            return all_distances
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to calculate distance: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            return None

    def segment_distance(self):
        """Segment the distance data to identify different regimes"""
        if not hasattr(self, 'distance_data') or not self.distance_data:
            messagebox.showinfo("Information", "Please calculate the DNAp-SSB distance first.")
            return
            
        try:
            self.status_var.set("Segmenting distance data...")
            self.master.update_idletasks()
            
            # Select which SSB trajectory to segment (use the first one by default)
            distance_data = self.distance_data[0]
            
            # Get time and distance data
            time = distance_data['ssb_time']
            distance = distance_data['distance']
            
            # Calculate the first derivative
            derivative = first_derivative(time, distance)
            
            # Detect segments based on the derivative
            segments = segment_data(time, distance, derivative)
            
            # Process the segments to calculate statistics
            self.segmented_data = process_segments(segments, time, distance)
            
            # Display the segmentation results in a new window
            self.display_segmentation_results()
            
            self.status_var.set("Distance data segmentation complete")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to segment distance data: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            
    def display_segmentation_results(self):
        """Display the segmentation results"""
        if not hasattr(self, 'segmented_data') or self.segmented_data is None:
            return
            
        try:
            # Create a new figure for segmentation results
            fig = Figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            
            # Get the first distance data set
            distance_data = self.distance_data[0]
            time = distance_data['ssb_time']
            distance = distance_data['distance']
            
            # Plot the raw data
            ax.scatter(time, distance, color='gray', alpha=0.5, label='Raw Data')
            
            # Plot each segment with a different color
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.segmented_data)))
            
            for i, segment in enumerate(self.segmented_data.itertuples()):
                segment_mask = (time >= segment.start_time) & (time <= segment.end_time)
                segment_time = time[segment_mask]
                segment_distance = distance[segment_mask]
                
                ax.plot(segment_time, segment_distance, 
                        color=colors[i], linewidth=2, 
                        label=f'Segment {i+1}: {segment.regime}')
                
            # Add horizontal line at y=0 
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            # Add grid and labels
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Distance (μm)')
            ax.set_title('Distance Segmentation Results')
            
            # Add legend
            ax.legend(loc='best')
            
            # Create a popup window for the plot
            popup = tk.Toplevel(self.master)
            popup.title("Distance Segmentation Results")
            popup.geometry("1000x800")
            
            # Display the figure in the popup
            canvas = FigureCanvasTkAgg(fig, master=popup)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar_frame = ttk.Frame(popup)
            toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            
            # Add a frame for displaying the segment data table
            table_frame = ttk.LabelFrame(popup, text="Segment Statistics")
            table_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Create a treeview for the table
            cols = list(self.segmented_data.columns)
            tree = ttk.Treeview(table_frame, columns=cols, show='headings')
            
            # Add the headings
            for col in cols:
                tree.heading(col, text=col.replace('_', ' ').title())
                tree.column(col, width=100, anchor='center')
            
            # Add the data
            for row in self.segmented_data.itertuples(index=False):
                values = [f"{getattr(row, col):.2f}" if isinstance(getattr(row, col), float) else str(getattr(row, col)) 
                          for col in cols]
                tree.insert('', 'end', values=values)
            
            # Add scrollbar
            scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            tree.pack(fill=tk.X, expand=True)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to display segmentation results: {str(e)}")

    def export_segments(self):
        """Export segmented data to CSV"""
        if not hasattr(self, 'segmented_data') or self.segmented_data is None:
            messagebox.showinfo("Information", "Please segment the distance data first.")
            return
            
        try:
            # Ask user for the file to save
            file_path = filedialog.asksaveasfilename(
                defaultextension='.csv',
                filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
                title="Export Segmented Data"
            )
            
            if not file_path:
                return  # User cancelled
                
            # Export to CSV
            self.segmented_data.to_csv(file_path, index=False)
            
            # Update status
            self.status_var.set(f"Segmented data exported to {file_path}")
            messagebox.showinfo("Export Complete", f"Segmented data successfully exported to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export segmented data: {str(e)}")
            self.status_var.set(f"Error: {str(e)}") 

    def export_distance_data(self):
        """Export distance data to CSV"""
        if not hasattr(self, 'distance_data') or not self.distance_data:
            messagebox.showinfo("Information", "Please calculate the DNAp-SSB distance first.")
            return
            
        try:
            # Ask user for the output directory
            output_dir = filedialog.askdirectory(
                title="Select Directory for Distance Data Export"
            )
            
            if not output_dir:
                return  # User cancelled
                
            # Get base name from input files
            kymo_filename = os.path.basename(self.kymo_entry.get()) if hasattr(self, 'kymo_entry') and self.kymo_entry.get() else "kymograph"
            base_name = os.path.splitext(kymo_filename)[0]
            
            # Export each distance data set
            for i, distance_data in enumerate(self.distance_data):
                # Create a dataframe from the distance data
                df = pd.DataFrame({
                    'Time': distance_data['ssb_time'],
                    'SSB_Position': distance_data['ssb_pos'],
                    'DNAp_Position': distance_data['dnap_pos'],
                    'Distance': distance_data['distance']
                })
                
                # Export to CSV
                file_path = os.path.join(output_dir, f"{base_name}_distance_data_{i+1}.csv")
                df.to_csv(file_path, index=False)
                
                # Generate a figure for this distance data
                plt.figure(figsize=(10, 6))
                plt.scatter(distance_data['ssb_time'], distance_data['distance'], alpha=0.5)
                plt.plot(distance_data['ssb_time'], distance_data['distance'], linewidth=1.5)
                plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                plt.grid(True, alpha=0.3)
                plt.xlabel('Time (s)')
                plt.ylabel('Distance (μm)')
                plt.title(f'DNAp-SSB Distance (Trajectory {i+1})')
                
                # Save the figure
                fig_path = os.path.join(output_dir, f"{base_name}_distance_plot_{i+1}.png")
                plt.savefig(fig_path, dpi=300)
                plt.close()
            
            # Update status
            self.status_var.set(f"Distance data exported to {output_dir}")
            messagebox.showinfo("Export Complete", 
                               f"Exported {len(self.distance_data)} distance datasets to:\n{output_dir}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to export distance data: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")

    def plot_dnap_ssb(self):
        """Plot DNAp and SSB traces together"""
        if not hasattr(self, 'traces') or not self.traces or self.trace is None:
            messagebox.showinfo("Information", "Please load both kymograph and trace files, and detect SSB trajectories first.")
            return
            
        try:
            self.status_var.set("Plotting DNAp and SSB traces...")
            self.master.update_idletasks()
            
            # Store axes state (on or off)
            axes_visible = self.plot_overlap_ax.get_xaxis().get_visible() and self.plot_overlap_ax.get_yaxis().get_visible()
            
            # Clear previous plot
            self.plot_overlap_ax.clear()
            
            # Get calibration parameters
            x_offset = float(self.param_entries["X Offset"].get()) if hasattr(self, 'param_entries') else 0
            y_offset = float(self.param_entries["Y Offset"].get()) if hasattr(self, 'param_entries') else 0
            time_scale = float(self.param_entries["Time Scale"].get()) if hasattr(self, 'param_entries') else 1.0
            position_scale = float(self.param_entries["Position Scale"].get()) if hasattr(self, 'param_entries') else 1.0
            
            # Plot DNAp trace
            dnap_time = self.trace['time'].values
            dnap_pos = self.trace['junction_forward'].values
            self.plot_overlap_ax.plot(dnap_time, dnap_pos, 'g-', linewidth=2, label='DNAp')
            
            # Plot each SSB trajectory with calibration
            for i, trace in enumerate(self.traces):
                # Apply calibration to SSB trajectories
                ssb_time = np.array(trace.time_idx) * time_scale + x_offset
                ssb_pos = np.array(trace.coordinate_idx) * position_scale + y_offset
                
                # Apply smoothing if trace has enough points
                if len(trace.time_idx) > 31:
                    window_size = min(31, len(trace.time_idx) - (len(trace.time_idx) % 2) - 1)  # Make sure it's odd
                    if window_size >= 3:
                        ssb_pos_smooth = savgol_filter(ssb_pos, window_size, 3)
                        self.plot_overlap_ax.plot(ssb_time, ssb_pos_smooth, 'r-', linewidth=1.5, label=f'SSB {i+1}' if i==0 else None)
                    else:
                        self.plot_overlap_ax.plot(ssb_time, ssb_pos, 'r-', linewidth=1.5, label=f'SSB {i+1}' if i==0 else None)
                else:
                    self.plot_overlap_ax.plot(ssb_time, ssb_pos, 'r-', linewidth=1.5, label=f'SSB {i+1}' if i==0 else None)
            
            # Add grid and labels
            self.plot_overlap_ax.grid(True, alpha=0.3)
            self.plot_overlap_ax.set_xlabel('Time (s)')
            self.plot_overlap_ax.set_ylabel('Position')
            self.plot_overlap_ax.set_title('DNAp (green) and SSB (red) Trajectories')
            
            # Add legend
            self.plot_overlap_ax.legend(loc='best')
            
            # Keep axes hidden if they were hidden before
            if not axes_visible:
                self.plot_overlap_ax.axis('off')
            
            # Update canvas
            self.plot_overlap_canvas.draw()
            
            # Update status
            self.status_var.set("DNAp and SSB trajectories plotted")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to plot DNAp and SSB traces: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")

    def load_trace_file(self, filename):
        """Load trace file with force and position data"""
        try:
            # Update status
            self.status_var.set(f"Loading {os.path.basename(filename)}...")
            self.master.update_idletasks()  # Force UI update
            
            # Load the data based on file extension
            if filename.endswith(('.xlsx', '.xls')):
                self.trace = pd.read_excel(filename)
            else:
                self.trace = pd.read_csv(filename)
            
            # Set alias for other methods
            self.trace_data = self.trace
            
            # Debug: Output the column names to help with debugging
            print(f"Original columns: {list(self.trace.columns)}")
            
            # Create a mapping dictionary - prioritizing standard input format
            column_map = {}
            
            # Time column
            if 'time' in self.trace.columns:
                column_map['time'] = 'time'
            elif 'Time' in self.trace.columns:
                column_map['Time'] = 'time'
                
            # Force column
            if 'force' in self.trace.columns:
                column_map['force'] = 'force'
            elif 'Force' in self.trace.columns:
                column_map['Force'] = 'force'
                
            # Position/Junction columns - these are crucial
            # Check for the specific column from the sample data
            if 'dsDNA_junction' in self.trace.columns:
                column_map['dsDNA_junction'] = 'junction_forward'
                print("Found dsDNA_junction column in data")
            elif 'ssDNA/dsDNA junction' in self.trace.columns:
                column_map['ssDNA/dsDNA junction'] = 'junction_forward'
                print("Found ssDNA/dsDNA junction column in data")
            elif 'junction_position' in self.trace.columns:
                column_map['junction_position'] = 'junction_forward'
                print("Found junction_position column in data")
            
            # EED column (if exists)
            if 'EED' in self.trace.columns:
                column_map['EED'] = 'EED'
                print("Found EED column in data")
                
            # Apply the column mapping
            self.trace = self.trace.rename(columns=column_map)
            print(f"After mapping columns: {list(self.trace.columns)}")
            
            # Create missing columns if needed
            if 'time' not in self.trace.columns:
                print("Creating time column")
                self.trace['time'] = np.arange(len(self.trace))
                
            if 'junction_forward' not in self.trace.columns:
                print("No junction_forward column found. Looking for alternatives...")
                # Try to find a suitable column to use
                if 'dsDNA_junction' in self.trace.columns:
                    print("Using dsDNA_junction column")
                    self.trace['junction_forward'] = self.trace['dsDNA_junction']
                elif 'basepairs_file' in self.trace.columns:
                    print("Using basepairs_file column")
                    self.trace['junction_forward'] = self.trace['basepairs_file']
                else:
                    # Try to find any numeric column that might be position data
                    numeric_cols = self.trace.select_dtypes(include=[np.number]).columns
                    position_candidates = [col for col in numeric_cols if 'position' in col.lower() or 'junction' in col.lower()]
                    
                    if position_candidates:
                        print(f"Using {position_candidates[0]} as junction_forward")
                        self.trace['junction_forward'] = self.trace[position_candidates[0]]
                    else:
                        # Create an empty column as a last resort
                        print("No suitable column found for junction_forward. Creating empty column.")
                        self.trace['junction_forward'] = np.zeros(len(self.trace))
                    
            # Create junction_reverse if not found
            if 'junction_reverse' not in self.trace.columns:
                if 'EED' in self.trace.columns and 'junction_forward' in self.trace.columns:
                    print("Creating junction_reverse from EED - junction_forward")
                    self.trace['junction_reverse'] = self.trace['EED'] - self.trace['junction_forward']
                else:
                    print("Creating junction_reverse as copy of junction_forward")
                    self.trace['junction_reverse'] = self.trace['junction_forward'].copy()
            
            # Update status
            self.status_var.set(f"Loaded trace file: {os.path.basename(filename)}")
            
            # Print final columns
            print(f"Final columns: {list(self.trace.columns)}")
            print(f"Data sample: {self.trace[['time', 'junction_forward']].head()}")
            
            # Update the OT data preview
            self.update_preview_display()
            
            # Try to plot junction forward if possible
            try:
                self.plot_junction_forward()
            except Exception as e:
                print(f"Could not plot junction forward: {str(e)}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error loading trace file: {str(e)}")
            messagebox.showerror("Error", f"Failed to load trace file: {str(e)}")
            self.trace = None

    def setup_ot_analysis_tab(self, tab):
        """Set up the Optical Tweezers data analysis tab"""
        # Main frame for OT analysis
        main_frame = ttk.Frame(tab)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        main_frame.columnconfigure(1, weight=3)
        main_frame.rowconfigure(0, weight=1)
        
        # Input parameters frame (left side)
        input_frame = ttk.LabelFrame(main_frame, text="Input Parameters")
        input_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Plots frame (right side)
        plot_frame = ttk.LabelFrame(main_frame, text="Plots")
        plot_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)
        
        # File selection
        file_frame = ttk.Frame(input_frame)
        file_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=5)
        
        ttk.Label(file_frame, text="TDMS File:").pack(side=tk.LEFT)
        self.ot_filename_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.ot_filename_var, width=40).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_ot_file).pack(side=tk.LEFT)
        
        # Basic parameters
        params_frame = ttk.Frame(input_frame)
        params_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5)
        
        # Cycle Number
        ttk.Label(params_frame, text="Cycle Number:").grid(row=0, column=0, sticky="w", pady=2)
        self.cycle_var = tk.StringVar(value="01")
        ttk.Entry(params_frame, textvariable=self.cycle_var, width=10).grid(row=0, column=1, sticky="w", padx=5)
        
        # Total Basepairs
        ttk.Label(params_frame, text="Total Basepairs:").grid(row=0, column=2, sticky="w", pady=2)
        self.total_basepairs_var = tk.StringVar(value="8393")
        ttk.Entry(params_frame, textvariable=self.total_basepairs_var, width=10).grid(row=0, column=3, sticky="w", padx=5)
        
        # Exo Start Time
        ttk.Label(params_frame, text="Exo Start Time (ms):").grid(row=1, column=0, sticky="w", pady=2)
        self.time_from_exo_var = tk.StringVar()
        exo_start_entry = ttk.Entry(params_frame, textvariable=self.time_from_exo_var, width=10)
        exo_start_entry.grid(row=1, column=1, sticky="w", padx=5)
        exo_start_entry.bind("<FocusIn>", lambda e: self.set_active_time_entry("exo_start"))
        
        # Exo End Time
        ttk.Label(params_frame, text="Exo End Time (ms):").grid(row=1, column=2, sticky="w", pady=2)
        self.time_to_exo_var = tk.StringVar()
        exo_end_entry = ttk.Entry(params_frame, textvariable=self.time_to_exo_var, width=10)
        exo_end_entry.grid(row=1, column=3, sticky="w", padx=5)
        exo_end_entry.bind("<FocusIn>", lambda e: self.set_active_time_entry("exo_end"))
        
        # Pol Start Time
        ttk.Label(params_frame, text="Pol Start Time (ms):").grid(row=2, column=0, sticky="w", pady=2)
        self.time_from_pol_var = tk.StringVar()
        pol_start_entry = ttk.Entry(params_frame, textvariable=self.time_from_pol_var, width=10)
        pol_start_entry.grid(row=2, column=1, sticky="w", padx=5)
        pol_start_entry.bind("<FocusIn>", lambda e: self.set_active_time_entry("pol_start"))
        
        # Pol End Time
        ttk.Label(params_frame, text="Pol End Time (ms):").grid(row=2, column=2, sticky="w", pady=2)
        self.time_to_pol_var = tk.StringVar()
        pol_end_entry = ttk.Entry(params_frame, textvariable=self.time_to_pol_var, width=10)
        pol_end_entry.grid(row=2, column=3, sticky="w", padx=5)
        pol_end_entry.bind("<FocusIn>", lambda e: self.set_active_time_entry("pol_end"))
        
        # Force parameters
        ttk.Label(params_frame, text="Exo Force (pN):").grid(row=4, column=0, sticky="w", pady=2)
        self.exo_force_var = tk.StringVar(value="45")
        ttk.Entry(params_frame, textvariable=self.exo_force_var, width=10).grid(row=4, column=1, sticky="w", padx=5)
        
        ttk.Label(params_frame, text="Pol Force (pN):").grid(row=4, column=2, sticky="w", pady=2)
        self.pol_force_var = tk.StringVar(value="10")
        ttk.Entry(params_frame, textvariable=self.pol_force_var, width=10).grid(row=4, column=3, sticky="w", padx=5)
        
        ttk.Label(params_frame, text="Bead Size:").grid(row=5, column=0, sticky="w", pady=2)
        self.bead_size_var = tk.StringVar(value="1.76")
        ttk.Entry(params_frame, textvariable=self.bead_size_var, width=10).grid(row=5, column=1, sticky="w", padx=5)
        
        # Segment Number
        ttk.Label(params_frame, text="Segment Number:").grid(row=5, column=2, sticky="w", pady=2)
        self.segment_number_var = tk.StringVar(value="20")
        ttk.Entry(params_frame, textvariable=self.segment_number_var, width=10).grid(row=5, column=3, sticky="w", padx=5)

        # SSB Factor
        ttk.Label(params_frame, text="SSB Factor:").grid(row=6, column=0, sticky="w", pady=2)
        self.ssb_factor_var = tk.StringVar(value="0.03")
        ttk.Entry(params_frame, textvariable=self.ssb_factor_var, width=10).grid(row=6, column=1, sticky="w", padx=5)
        
        # Model parameters frame
        model_frame = ttk.LabelFrame(input_frame, text="Model Parameters", padding="5")
        model_frame.grid(row=2, column=0, columnspan=3, pady=10, sticky="nsew")
        
        # tWLC parameters
        ttk.Label(model_frame, text="tWLC Parameters:").grid(row=0, column=0, columnspan=2, sticky="w")
        self.C_var = tk.StringVar(value="440")
        self.g0_var = tk.StringVar(value="-637")
        self.g1_var = tk.StringVar(value="17")
        self.Lc_var = tk.StringVar(value="2.85056")
        self.Lp_var = tk.StringVar(value="56")
        self.S_var = tk.StringVar(value="1500")
        
        params = [
            ("C:", self.C_var), ("g0:", self.g0_var), ("g1:", self.g1_var),
            ("Lc:", self.Lc_var), ("Lp:", self.Lp_var), ("S:", self.S_var)
        ]
        for i, (label, var) in enumerate(params):
            ttk.Label(model_frame, text=label).grid(row=i+1, column=0, sticky="w")
            ttk.Entry(model_frame, textvariable=var, width=10).grid(row=i+1, column=1, padx=5)
            
        # FJC parameters
        ttk.Label(model_frame, text="FJC Parameters:").grid(row=0, column=2, columnspan=2, sticky="w")
        self.Lss_var = tk.StringVar(value="4.69504")
        self.b_var = tk.StringVar(value="1.5")
        self.Sss_var = tk.StringVar(value="800")
        
        fjc_params = [
            ("Lss:", self.Lss_var), ("b:", self.b_var), ("Sss:", self.Sss_var)
        ]
        for i, (label, var) in enumerate(fjc_params):
            ttk.Label(model_frame, text=label).grid(row=i+1, column=2, sticky="w")
            ttk.Entry(model_frame, textvariable=var, width=10).grid(row=i+1, column=3, padx=5)
        
        # Buttons frame
        buttons_frame = ttk.Frame(input_frame)
        buttons_frame.grid(row=3, column=0, columnspan=3, pady=10, sticky="ew")
        buttons_frame.grid_columnconfigure(0, weight=1)
        buttons_frame.grid_columnconfigure(1, weight=1)
        
        # Model fitting button
        ttk.Button(buttons_frame, text="Fit to Model", command=self.fit_to_model).grid(row=0, column=0, columnspan=2, sticky="ew", pady=2, padx=2)
        
        # Track junction buttons
        track_buttons_frame = ttk.Frame(buttons_frame)
        track_buttons_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)
        track_buttons_frame.grid_columnconfigure(0, weight=1)
        track_buttons_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Button(track_buttons_frame, text="Track ss/dsDNA Junction (forward)", 
                  command=lambda: self.track_junction(reverse=False)).grid(row=0, column=0, sticky="ew", padx=2)
        ttk.Button(track_buttons_frame, text="Track ss/dsDNA Junction (reverse)", 
                  command=lambda: self.track_junction(reverse=True)).grid(row=0, column=1, sticky="ew", padx=2)
        
        # DNA polymerase trace button
        ttk.Button(buttons_frame, text="Save DNA Polymerase Trace", 
                  command=self.plot_dna_polymerase_trace).grid(row=2, column=0, columnspan=2, sticky="ew", pady=2, padx=2)
        
        # Linear Segment Fitting button
        ttk.Button(buttons_frame, text="Linear Segment Fitting", 
                  command=self.linear_segment_fitting).grid(row=3, column=0, columnspan=2, sticky="ew", pady=2, padx=2)
        
        # Save Data button
        ttk.Button(buttons_frame, text="Save Data", 
                  command=self.save_ot_data).grid(row=4, column=0, columnspan=2, sticky="ew", pady=2, padx=2)
        
        # Configure plot frame
        self.ot_fig = plt.Figure(figsize=(12, 8))
        self.ot_canvas = FigureCanvasTkAgg(self.ot_fig, master=plot_frame)
        self.ot_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Add toolbar
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        toolbar = NavigationToolbar2Tk(self.ot_canvas, toolbar_frame)
        toolbar.update()
    
    def set_active_time_entry(self, entry_type):
        """Set the active time entry field for click selection"""
        self.active_time_entry = entry_type
        
    def browse_ot_file(self):
        """Browse for an OT data file (TDMS format)"""
        filename = filedialog.askopenfilename(filetypes=[("TDMS files", "*.tdms")])
        if filename:
            self.ot_filename_var.set(filename)
            self.load_and_plot_ot_data()
    
    def load_and_plot_ot_data(self):
        """Load and plot OT data from a TDMS file"""
        try:
            # Update status
            self.status_var.set(f"Loading OT data file...")
            self.master.update_idletasks()
            
            # Read TDMS file
            from nptdms import TdmsFile
            tdms_file = TdmsFile(self.ot_filename_var.get())
            
            # Extract data
            self.time_data = np.array(tdms_file['FD Data']['Time (ms)'][:])
            self.force_data = np.array(tdms_file['FD Data']['Force Channel 0 (pN)'][:])
            self.distance_data = np.array(tdms_file['FD Data']['Distance 1 (um)'][:])
            
            # Clear previous plots
            self.ot_fig.clear()
            
            # Create initial force vs time plot
            ax = self.ot_fig.add_subplot(111)
            ax.plot(self.time_data, self.force_data)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Force (pN)')
            ax.set_title('Click to select time points (select input box first)')
            
            # Add cursor
            from matplotlib.widgets import Cursor
            cursor = Cursor(ax, useblit=True, color='red', linewidth=1)
            
            # Connect click event
            self.ot_canvas.mpl_connect('button_press_event', self.on_click)
            
            self.ot_fig.tight_layout()
            self.ot_canvas.draw()
            
            # Update status
            self.status_var.set(f"OT data loaded successfully: {os.path.basename(self.ot_filename_var.get())}")
            
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred while loading the file: {str(e)}")
            self.status_var.set(f"Error loading OT data: {str(e)}")
    
    def on_click(self, event):
        """Handle mouse clicks on the plot for time selection"""
        if event.inaxes and self.active_time_entry:
            if self.active_time_entry == "exo_start":
                self.time_from_exo_var.set(f"{event.xdata:.1f}")
            elif self.active_time_entry == "exo_end":
                self.time_to_exo_var.set(f"{event.xdata:.1f}")
            elif self.active_time_entry == "pol_start":
                self.time_from_pol_var.set(f"{event.xdata:.1f}")
            elif self.active_time_entry == "pol_end":
                self.time_to_pol_var.set(f"{event.xdata:.1f}")
    
    def tWLC(self, F):
        """Implementation of the tWLC model for dsDNA"""
        try:
            C = float(self.C_var.get())
            g0 = float(self.g0_var.get())
            g1 = float(self.g1_var.get())
            Lc = float(self.Lc_var.get())
            Lp = float(self.Lp_var.get())
            S = float(self.S_var.get())
            return Lc * (1 - 0.5 * (4.1/(F*Lp))**0.5 + C*F/(-(g0+g1*F)**2 + S*C))
        except Exception as e:
            messagebox.showerror("Error", f"Error in tWLC calculation: {str(e)}")
            return np.zeros_like(F)
    
    def FJC(self, F):
        """Implementation of the FJC model for ssDNA"""
        try:
            Lss = float(self.Lss_var.get())
            b = float(self.b_var.get())
            Sss = float(self.Sss_var.get())
            
            # Numerical implementation of hyperbolic cotangent
            def numerical_coth(x):
                return 1.0 / np.tanh(np.array(x, dtype=float))
                
            EEDss = []
            for Fext in F:
                x = Lss * (numerical_coth(Fext * b / 4.1) - 4.1 / (Fext * b)) * (1 + Fext / Sss)
                EEDss.append(x)
            return np.array(EEDss)
        except Exception as e:
            messagebox.showerror("Error", f"Error in FJC calculation: {str(e)}")
            return np.zeros_like(F)
    
    def get_exo_pol_analysis_data(self):
        """Get separate analysis data for exo and pol phases"""
        try:
            # Check if data is loaded
            if self.time_data is None or len(self.time_data) == 0:
                messagebox.showerror("Error", "No data loaded. Please load a TDMS file first.")
                return None, None, None, None, None, None
            
            # Check required parameters
            for var_name in ['time_from_exo_var', 'time_to_exo_var', 'time_from_pol_var', 'time_to_pol_var']:
                if not getattr(self, var_name).get():
                    messagebox.showerror("Error", f"Please set all time ranges for exo and pol phases.")
                    return None, None, None, None, None, None
            
            # Get exo data
            time_from_exo = float(self.time_from_exo_var.get())
            time_to_exo = float(self.time_to_exo_var.get())
            indtemp_exo = np.where((self.time_data <= time_to_exo) & (self.time_data >= time_from_exo))
            
            time_range_exo = self.time_data[indtemp_exo]
            force_range_exo = self.force_data[indtemp_exo]
            distance_range_exo = self.distance_data[indtemp_exo]
            
            # Get pol data
            time_from_pol = float(self.time_from_pol_var.get())
            time_to_pol = float(self.time_to_pol_var.get())
            indtemp_pol = np.where((self.time_data <= time_to_pol) & (self.time_data >= time_from_pol))
            
            time_range_pol = self.time_data[indtemp_pol]
            force_range_pol = self.force_data[indtemp_pol]
            distance_range_pol = self.distance_data[indtemp_pol]
            
            # Combine for all data
            time_range_all = np.append(time_range_exo, time_range_pol)
            force_range_all = np.append(force_range_exo, force_range_pol)
            distance_range_all = np.append(distance_range_exo, distance_range_pol)
            
            # Calculate parameters
            bead_size = float(self.bead_size_var.get())
            ssb_factor = float(self.ssb_factor_var.get())
            exo_force = float(self.exo_force_var.get())
            pol_force = float(self.pol_force_var.get())
            
            # Calculate reference values for dsDNA and ssDNA under exo and pol forces
            dsDNA_exo_ref = self.tWLC(exo_force)
            dsDNA_pol_ref = self.tWLC(pol_force)
            
            # Calculate ssDNA reference values with numerical implementation
            Lss = float(self.Lss_var.get())
            b = float(self.b_var.get())
            Sss = float(self.Sss_var.get())
            
            # Numerical implementation of hyperbolic cotangent
            def numerical_coth(x):
                return 1.0 / np.tanh(np.array(x, dtype=float))
            
            ssDNA_exo_ref = Lss * (numerical_coth(exo_force * b / 4.1) - 4.1 / (exo_force * b)) * (1 + exo_force / Sss)
            ssDNA_pol_ref = Lss * (numerical_coth(pol_force * b / 4.1) - 4.1 / (pol_force * b)) * (1 + pol_force / Sss)
            
            # Calculate ssDNA percentages
            ssDNA_exo_percentage = (distance_range_exo - bead_size - dsDNA_exo_ref)/(ssDNA_exo_ref - ssb_factor - dsDNA_exo_ref)
            ssDNA_pol_percentage = (distance_range_pol - bead_size - dsDNA_pol_ref)/(ssDNA_pol_ref - ssb_factor - dsDNA_pol_ref)
            ssDNA_all_percentage = np.append(ssDNA_exo_percentage, ssDNA_pol_percentage)
            
            # Calculate basepairs
            total_basepairs = float(self.total_basepairs_var.get())
            basepairs = (1 - ssDNA_all_percentage) * total_basepairs
            
            # Store data for later reference
            self.time_range_exo = time_range_exo
            self.force_range_exo = force_range_exo
            self.distance_range_exo = distance_range_exo
            self.time_range_pol = time_range_pol
            self.force_range_pol = force_range_pol
            self.distance_range_pol = distance_range_pol
            self.ssDNA_exo_percentage = ssDNA_exo_percentage
            self.ssDNA_pol_percentage = ssDNA_pol_percentage
            self.ssDNA_exo_ref = ssDNA_exo_ref
            self.ssDNA_pol_ref = ssDNA_pol_ref
            self.dsDNA_exo_ref = dsDNA_exo_ref
            self.dsDNA_pol_ref = dsDNA_pol_ref
            
            # Calculate junction positions for both exo and pol phases
            junction_position_exo = self.get_junction_position(
                distance_range_exo, 
                ssDNA_exo_percentage, 
                ssDNA_exo_ref, 
                dsDNA_exo_ref, 
                bead_size, 
                False  # Forward direction
            )
            
            junction_position_pol = self.get_junction_position(
                distance_range_pol, 
                ssDNA_pol_percentage, 
                ssDNA_pol_ref, 
                dsDNA_pol_ref, 
                bead_size, 
                False  # Forward direction
            )
            
            # Combine junction positions
            junction_position_all = np.append(junction_position_exo, junction_position_pol)
            
            # Store calculated positions
            self.junction_position_exo = junction_position_exo
            self.junction_position_pol = junction_position_pol
            
            return time_range_all, force_range_all, distance_range_all, ssDNA_all_percentage, basepairs, junction_position_all
            
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred during exo/pol analysis: {str(e)}")
            return None, None, None, None, None, None
    
    def get_junction_position(self, distance_range, ssDNA_percentage, ssDNA_ref, dsDNA_ref, bead_size, reverse=False):
        """
        Calculate ss/dsDNA junction position based on the direction (forward or reverse)
        SSB correction factor is included in the ssDNA_percentage calculation, but we need to apply it to ssDNA_ref too
        """
        # Get SSB factor
        ssb_factor = float(self.ssb_factor_var.get())
        # Apply SSB correction to ssDNA_ref
        corrected_ssDNA_ref = ssDNA_ref - ssb_factor
        
        if reverse:
            # Reverse direction calculation
            return (distance_range - bead_size) - (ssDNA_percentage * corrected_ssDNA_ref) * (distance_range - bead_size) / ((ssDNA_percentage * corrected_ssDNA_ref) + (1 - ssDNA_percentage) * dsDNA_ref)
        else:
            # Forward direction calculation
            return (ssDNA_percentage * corrected_ssDNA_ref) * (distance_range - bead_size) / ((ssDNA_percentage * corrected_ssDNA_ref) + (1 - ssDNA_percentage) * dsDNA_ref)
    
    def fit_to_model(self):
        """Fit the data to the tWLC and FJC models"""
        try:
            # Update status
            self.status_var.set("Fitting data to models...")
            self.master.update_idletasks()
            
            if self.time_data is None:
                messagebox.showerror("Error", "Please load a TDMS file first")
                self.status_var.set("Error: No data loaded")
                return
            
            # Use exo/pol data instead of ROI data
            if (self.time_from_exo_var.get() and self.time_to_exo_var.get() and 
                self.time_from_pol_var.get() and self.time_to_pol_var.get()):
                time_range, force_range, distance_range, ssDNA_percentage, basepairs, _ = self.get_exo_pol_analysis_data()
                if time_range is None:  # Error occurred in get_exo_pol_analysis_data
                    self.status_var.set("Error in exo/pol data analysis")
                    return
            else:
                messagebox.showerror("Error", "Please set Exo and Pol time ranges first")
                self.status_var.set("Error: Missing time ranges")
                return
            
            # Clear previous plots
            self.ot_fig.clear()
            
            # Create subplots in the top half
            gs = self.ot_fig.add_gridspec(2, 2, height_ratios=[1, 0.1])
            ax1 = self.ot_fig.add_subplot(gs[0, 0])
            ax2 = self.ot_fig.add_subplot(gs[0, 1])
            
            # Plot 1: Models and experimental data
            Force = np.linspace(0.1, 68, 1000)
            ax1.plot(self.tWLC(Force), Force, 'r-', label='WLC Model')
            ax1.plot(self.FJC(Force), Force, 'b-', label='FJC Model')
            ax1.plot(distance_range - float(self.bead_size_var.get()), force_range, 'k.', label='Experimental Data')
            ax1.set_xlabel('Distance (μm)')
            ax1.set_ylabel('Force (pN)')
            ax1.legend()
            ax1.set_title('Model Fitting')
            
            # Plot 2: Base pairs changes
            # Use Savitzky-Golay filter to smooth basepairs
            bp_filter = savgol_filter(basepairs, 31, 3)
            
            # Plot basepair changes as a function of time
            ax2.plot(time_range/1000, basepairs, color='lightgrey', linewidth=1)
            ax2.plot(time_range/1000, bp_filter, color='green', linewidth=1, markersize=1, label='Basepairs')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Base pairs')
            ax2.set_title('DNA polymerase catalyzing DNA')
            
            # Adjust spacing between subplots
            self.ot_fig.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.9, wspace=0.3)
            self.ot_canvas.draw()
            
            # Update status
            self.status_var.set("Model fitting completed successfully")
            
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred during model fitting: {str(e)}")
            self.status_var.set(f"Error in model fitting: {str(e)}")
    
    def track_junction(self, reverse=False):
        """Track the ss/dsDNA junction position"""
        try:
            # Update status
            self.status_var.set("Tracking ss/dsDNA junction position...")
            self.master.update_idletasks()
            
            if self.time_data is None:
                messagebox.showerror("Error", "Please load a TDMS file first")
                self.status_var.set("Error: No data loaded")
                return
            
            # Check if we have exo/pol data
            if not (self.time_from_exo_var.get() and self.time_to_exo_var.get() and 
                    self.time_from_pol_var.get() and self.time_to_pol_var.get()):
                messagebox.showerror("Error", "Please set Exo and Pol time ranges first")
                self.status_var.set("Error: Missing time ranges")
                return
                
            bead_size = float(self.bead_size_var.get())
            
            # If reverse is True, we need to recalculate the junction positions with reverse=True
            if reverse:
                # Get data without junction positions first
                time_range_all, force_range_all, distance_range_all, ssDNA_all_percentage, basepairs, _ = self.get_exo_pol_analysis_data()
                if time_range_all is None:  # Error occurred in get_exo_pol_analysis_data
                    self.status_var.set("Error in exo/pol data analysis")
                    return
                    
                # Calculate junction positions for reverse direction
                junction_position_exo = self.get_junction_position(
                    self.distance_range_exo, 
                    self.ssDNA_exo_percentage, 
                    self.ssDNA_exo_ref, 
                    self.dsDNA_exo_ref, 
                    bead_size, 
                    True  # Reverse direction
                )
                
                junction_position_pol = self.get_junction_position(
                    self.distance_range_pol, 
                    self.ssDNA_pol_percentage, 
                    self.ssDNA_pol_ref, 
                    self.dsDNA_pol_ref, 
                    bead_size, 
                    True  # Reverse direction
                )
                
                junction_position_all = np.append(junction_position_exo, junction_position_pol)
                time_range_all = np.append(self.time_range_exo, self.time_range_pol)
                distance_range_all = np.append(self.distance_range_exo, self.distance_range_pol)
            else:
                # For forward direction, we can use the junction positions directly from get_exo_pol_analysis_data
                time_range_all, force_range_all, distance_range_all, ssDNA_all_percentage, basepairs, junction_position_all = self.get_exo_pol_analysis_data()
                if time_range_all is None:  # Error occurred in get_exo_pol_analysis_data
                    self.status_var.set("Error in exo/pol data analysis")
                    return
            
            # Clear previous plots
            self.ot_fig.clear()
            
            # Create plot
            ax = self.ot_fig.add_subplot(111)
            
            # Plot junction movement using the updated parameters
            ax.scatter(time_range_all/1000, distance_range_all - bead_size, color='black', linestyle='dashed', s=2, label='End-to-End Distance')
            ax.scatter(time_range_all/1000, junction_position_all, color='green', s=2, label='DNA Polymerase Trace')
            
            # Fill between the lines
            ax.fill_between(np.array(time_range_all/1000), distance_range_all - bead_size, junction_position_all, color='gray', alpha=0.2)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Distance (μm)')
            ax.set_title('ssDNA/dsDNA Junction Position' + (' (Reverse Direction)' if reverse else ''))
            ax.legend()
            
            # Invert y-axis and move x-axis to top
            ax.invert_yaxis()
            ax.xaxis.set_ticks_position('top')
            
            self.ot_fig.tight_layout()
            self.ot_canvas.draw()
            
            # Update status
            self.status_var.set("Junction tracking completed successfully")
            
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred while tracking junction position: {str(e)}")
            self.status_var.set(f"Error in junction tracking: {str(e)}")
    
    def plot_dna_polymerase_trace(self):
        """Plot and save DNA polymerase trace using exo and pol data"""
        try:
            # Update status
            self.status_var.set("Generating DNA polymerase trace...")
            self.master.update_idletasks()
            
            # Reuse the track_junction method with default parameters (forward direction)
            self.track_junction(reverse=False)
            
            # Save the plot
            base_dir = os.path.dirname(self.ot_filename_var.get())
            base_name = os.path.splitext(os.path.basename(self.ot_filename_var.get()))[0]
            cycle = self.cycle_var.get()
            results_dir = os.path.join(base_dir, 'results')
            
            # Create results directory if it doesn't exist
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
                
            output_filename = os.path.join(results_dir, f"{base_name}-cycle#{cycle}-DNApTraces.png")
            self.ot_fig.savefig(output_filename, format='png', dpi=300, bbox_inches='tight')
            
            # Also save as EPS for publication-quality
            output_filename_eps = os.path.join(results_dir, f"{base_name}-cycle#{cycle}-DNApTraces.eps")
            self.ot_fig.savefig(output_filename_eps, format='eps', dpi=300, bbox_inches='tight')
            
            # Show success message
            messagebox.showinfo("Success", f"DNA Polymerase Trace saved to {output_filename}")
            self.status_var.set(f"DNA Polymerase Trace saved to {os.path.basename(output_filename)}")
            
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred while plotting DNA polymerase trace: {str(e)}")
            self.status_var.set(f"Error saving DNA polymerase trace: {str(e)}")
    
    def linear_segment_fitting(self):
        """Perform piecewise linear segment fitting on the trajectory data"""
        try:
            # Update status
            self.status_var.set("Performing linear segment fitting...")
            self.master.update_idletasks()
            
            # Check if FastPWLFit is available
            if FastPWLFit is None:
                messagebox.showerror("Error", "FastPWLFit module not found. Please install it to use segment fitting.")
                self.status_var.set("Error: FastPWLFit module not found")
                return
            
            if self.time_data is None:
                messagebox.showerror("Error", "Please load a TDMS file first")
                self.status_var.set("Error: No data loaded")
                return
            
            # Use exo/pol data instead of ROI data
            if (self.time_from_exo_var.get() and self.time_to_exo_var.get() and 
                self.time_from_pol_var.get() and self.time_to_pol_var.get()):
                time_range, _, _, _, basepairs, _ = self.get_exo_pol_analysis_data()
                if time_range is None:  # Error occurred in get_exo_pol_analysis_data
                    self.status_var.set("Error in exo/pol data analysis")
                    return
            else:
                messagebox.showerror("Error", "Please set Exo and Pol time ranges first")
                self.status_var.set("Error: Missing time ranges")
                return
            
            # Convert time to seconds and prepare data
            time_seconds = time_range / 1000
            
            # Setup results directory
            base_dir = os.path.dirname(self.ot_filename_var.get())
            base_name = os.path.splitext(os.path.basename(self.ot_filename_var.get()))[0]
            cycle = self.cycle_var.get()
            results_dir = os.path.join(base_dir, 'results')
            
            # Create results directory if it doesn't exist
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            
            # Initialize and fit the model
            segment_number = int(self.segment_number_var.get())
            pwlf = FastPWLFit(time_seconds, basepairs)
            pwlf.fit_model(segment_number)
            
            # Clear previous plot and create new one for the segments
            self.ot_fig.clear()
            ax = self.ot_fig.add_subplot(111)
            
            # Plot the original data and the fitted segments
            ax.scatter(time_seconds, basepairs, s=5, color='black', label='Data')
            x_fit = time_seconds
            y_fit = pwlf.predict(x_fit)
            ax.plot(x_fit, y_fit, color='yellow', label='Piecewise Linear Fit')
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('Basepairs (bp)', fontsize=12)
            ax.legend()
            
            self.ot_fig.tight_layout()
            self.ot_canvas.draw()
            
            # Save the fit results
            output_filename = os.path.join(results_dir, f"{base_name}-cycle#{cycle}_segments.csv")
            pwlf.save_results_csv(output_filename)
            
            # Save the fitted plot
            plot_filename = os.path.join(results_dir, f"{base_name}-cycle#{cycle}_segments.png")
            self.ot_fig.savefig(plot_filename, format='png', dpi=300, bbox_inches='tight')
            
            # Show success message
            messagebox.showinfo("Success", f"Linear segment fitting results saved to {results_dir}")
            self.status_var.set(f"Linear segment fitting completed and saved")
            
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred during linear segment fitting: {str(e)}")
            self.status_var.set(f"Error in linear segment fitting: {str(e)}")
    
    def save_ot_data(self):
        """Save processed OT data and plots"""
        try:
            # Update status
            self.status_var.set("Saving OT data and plots...")
            self.master.update_idletasks()
            
            if self.time_data is None:
                messagebox.showerror("Error", "Please load a TDMS file first")
                self.status_var.set("Error: No data loaded")
                return
            
            # Check if we have exo/pol data
            if (self.time_from_exo_var.get() and self.time_to_exo_var.get() and 
                self.time_from_pol_var.get() and self.time_to_pol_var.get()):
                # Use exo/pol data
                time_range, force_range, distance_range, ssDNA_percentage, basepairs, junction_position = self.get_exo_pol_analysis_data()
                if time_range is None:  # Error occurred in get_exo_pol_analysis_data
                    self.status_var.set("Error in exo/pol data analysis")
                    return
                
                # Apply Savitzky-Golay filter to basepairs
                basepairs_filtered = savgol_filter(basepairs, 31, 3)
                
                # Extract base filename without extension and create results directory
                base_dir = os.path.dirname(self.ot_filename_var.get())
                base_name = os.path.splitext(os.path.basename(self.ot_filename_var.get()))[0]
                cycle = self.cycle_var.get()
                results_dir = os.path.join(base_dir, 'results')
                
                # Create results directory if it doesn't exist
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                
                # Define font settings for plots
                import matplotlib as mpl
                font = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 16}
                mpl.rc('font', **font)
                
                # 1. Save Excel data
                output_filename = os.path.join(results_dir, f"{base_name}-cycle#{cycle}_processedData.xlsx")
                data = {
                    'time': time_range,
                    'force': force_range,
                    'EED': distance_range - float(self.bead_size_var.get()),
                    'ssDNA_percentage': ssDNA_percentage,
                    'basepairs': basepairs,
                    'basepairs_filtered': basepairs_filtered,
                    'ssDNA/dsDNA junction': junction_position
                }
                df = pd.DataFrame(data)
                df.to_excel(output_filename)
                
                # 2. Save Plot 1: Basepair Change (Filtered)
                plot_filename = os.path.join(results_dir, f"{base_name}-cycle#{cycle}-BasepairChange-filtered.png")
                plt.figure(figsize=(6, 4))
                plt.xlabel('Time (s)', fontdict=font)
                plt.ylabel('Basepairs', fontdict=font)
                plt.plot(time_range/1000, basepairs, color='lightgrey', linewidth=1)
                plt.plot(time_range/1000, basepairs_filtered, color='green', linewidth=1, label='Basepairs')
                plt.tight_layout()
                plt.savefig(plot_filename, format='png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # 3. Save Plot 2: ssDNA Percentage (as Basepairs)
                plot_filename = os.path.join(results_dir, f"{base_name}-cycle#{cycle}-ssDNA_percentage.png")
                plt.figure(figsize=(8, 3))
                plt.ylabel('Basepairs (bp)', fontdict=font)
                plt.xlabel('Time (s)', fontdict=font)
                plt.scatter(time_range/1000, basepairs, color='black', s=0.5, label='End-to-End Distance')
                plt.tight_layout()
                plt.savefig(plot_filename, format='png', dpi=300)
                plt.close()
                
                # 4. Save Plot 3: DNA Polymerase Traces
                plot_filename = os.path.join(results_dir, f"{base_name}-cycle#{cycle}-DNApTraces.png")
                plt.figure(figsize=(6, 4))
                plt.title('Time (s)', fontdict=font)
                plt.ylabel('Distance (µm)', fontdict=font)
                plt.scatter(time_range/1000, distance_range - float(self.bead_size_var.get()), color='black', s=2, label='End-to-End Distance')
                plt.scatter(time_range/1000, junction_position, color='green', s=2, label='DNA Polymerase Trace')
                plt.fill_between(time_range/1000, distance_range - float(self.bead_size_var.get()), junction_position, color='gray', alpha=0.2)
                plt.ylim(0, 3.8)
                ax = plt.gca()
                ax.invert_yaxis()
                ax.xaxis.set_ticks_position('top')
                plt.tight_layout()
                plt.savefig(plot_filename, format='png', dpi=300)
                plt.close()
                
                # 5. Save Plot 4: Basepair Change (Raw)
                plot_filename = os.path.join(results_dir, f"{base_name}-cycle#{cycle}-BasepairChange.png")
                plt.figure(figsize=(8, 3))
                plt.xlabel('Time (s)', fontdict=font)
                plt.ylabel('Basepairs', fontdict=font)
                plt.plot(time_range/1000, basepairs, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=2, label='Basepairs')
                plt.tight_layout()
                plt.savefig(plot_filename, format='png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Set matplotlib back to defaults
                mpl.rcdefaults()
                
                # 6. Save the current plot from the GUI as an EPS file
                output_filename_eps = os.path.join(results_dir, f"{base_name}-cycle#{cycle}-DNApTraces.eps")
                self.track_junction(reverse=False)
                self.ot_fig.savefig(output_filename_eps, format='eps', dpi=300, bbox_inches='tight')
                
                messagebox.showinfo("Success", f"All data and plots saved to {results_dir}")
                self.status_var.set(f"Data and plots saved to {os.path.basename(results_dir)}")
            else:
                messagebox.showerror("Error", "Please set Exo and Pol time ranges first")
                self.status_var.set("Error: Missing time ranges")
            
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred while saving data: {str(e)}")
            self.status_var.set(f"Error saving data: {str(e)}")

    def setup_interaction_analysis_tab(self, tab):
        """Set up the DNAp-SSB interaction analysis tab to combine OT and kymograph data"""
        # Main frame for interaction analysis
        main_frame = ttk.Frame(tab)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Create top section for data selection
        data_frame = ttk.LabelFrame(main_frame, text="Data Selection")
        data_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # Create a frame for file selection
        file_frame = ttk.Frame(data_frame)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # OT data selection
        ttk.Label(file_frame, text="OT Data:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.ot_data_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.ot_data_var, width=40).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_ot_data).grid(row=0, column=2, padx=5)
        
        # Kymograph data selection
        ttk.Label(file_frame, text="Kymograph Data:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.kymo_data_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.kymo_data_var, width=40).grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_kymo_data).grid(row=1, column=2, padx=5)
        
        # Synchronization parameters
        sync_frame = ttk.LabelFrame(data_frame, text="Synchronization Parameters")
        sync_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Time offset between datasets
        ttk.Label(sync_frame, text="Time Offset (s):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.time_offset_var = tk.StringVar(value="0.0")
        ttk.Entry(sync_frame, textvariable=self.time_offset_var, width=10).grid(row=0, column=1, sticky="w", padx=5)
        
        # Scale factor for distance
        ttk.Label(sync_frame, text="Distance Scale Factor:").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        self.distance_scale_var = tk.StringVar(value="1.0")
        ttk.Entry(sync_frame, textvariable=self.distance_scale_var, width=10).grid(row=0, column=3, sticky="w", padx=5)
        
        # Analysis controls
        control_frame = ttk.Frame(data_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Analysis buttons
        ttk.Button(control_frame, text="Load and Synchronize Data", 
                  command=self.load_and_sync_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Analyze Interaction", 
                  command=self.analyze_dnap_ssb_interaction).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Export Combined Results", 
                  command=self.export_interaction_results).pack(side=tk.LEFT, padx=5)
        
        # Create bottom section with plot area
        plot_frame = ttk.LabelFrame(main_frame, text="Interaction Analysis")
        plot_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        
        # Create a figure for interaction plots
        self.interaction_fig = plt.Figure(figsize=(10, 6))
        self.interaction_canvas = FigureCanvasTkAgg(self.interaction_fig, master=plot_frame)
        self.interaction_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Add toolbar
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        toolbar = NavigationToolbar2Tk(self.interaction_canvas, toolbar_frame)
        toolbar.update()
    
    def browse_ot_data(self):
        """Browse for OT data file"""
        filename = filedialog.askopenfilename(
            filetypes=[("Excel files", "*.xlsx;*.xls"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.ot_data_var.set(filename)
    
    def browse_kymo_data(self):
        """Browse for kymograph analysis data file"""
        filename = filedialog.askopenfilename(
            filetypes=[("Excel files", "*.xlsx;*.xls"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.kymo_data_var.set(filename)
    
    def load_and_sync_data(self):
        """Load and synchronize OT and kymograph data"""
        try:
            # Update status
            self.status_var.set("Loading and synchronizing data...")
            self.master.update_idletasks()
            
            # Check if files are selected
            if not self.ot_data_var.get():
                messagebox.showerror("Error", "Please select an OT data file")
                self.status_var.set("Error: No OT data file selected")
                return
            
            if not self.kymo_data_var.get():
                messagebox.showerror("Error", "Please select a kymograph data file")
                self.status_var.set("Error: No kymograph data file selected")
                return
            
            # Load OT data
            ot_file = self.ot_data_var.get()
            if ot_file.endswith(('.xlsx', '.xls')):
                self.ot_combined_data = pd.read_excel(ot_file)
            else:
                self.ot_combined_data = pd.read_csv(ot_file)
            
            # Load kymograph data
            kymo_file = self.kymo_data_var.get()
            if kymo_file.endswith(('.xlsx', '.xls')):
                self.kymo_data = pd.read_excel(kymo_file)
            else:
                self.kymo_data = pd.read_csv(kymo_file)
            
            # Apply time offset to synchronize datasets
            time_offset = float(self.time_offset_var.get())
            
            # Ensure we have the right columns
            required_ot_cols = ['time', 'basepairs', 'ssDNA/dsDNA junction']
            required_kymo_cols = ['time', 'ssb_position']
            
            # Check and map column names if needed
            for dataset, req_cols, name in [
                (self.ot_combined_data, required_ot_cols, 'OT data'),
                (self.kymo_data, required_kymo_cols, 'Kymograph data')
            ]:
                missing_cols = [col for col in req_cols if col not in dataset.columns]
                if missing_cols:
                    # Try to identify alternative column names
                    if 'ssb_position' in missing_cols and 'position' in dataset.columns:
                        dataset.rename(columns={'position': 'ssb_position'}, inplace=True)
                        missing_cols.remove('ssb_position')
                    
                    if 'ssDNA/dsDNA junction' in missing_cols and 'junction_forward' in dataset.columns:
                        dataset.rename(columns={'junction_forward': 'ssDNA/dsDNA junction'}, inplace=True)
                        missing_cols.remove('ssDNA/dsDNA junction')
                    
                    if 'time' in missing_cols and 'Time' in dataset.columns:
                        dataset.rename(columns={'Time': 'time'}, inplace=True)
                        missing_cols.remove('time')
                
                # Check again after mapping
                missing_cols = [col for col in req_cols if col not in dataset.columns]
                if missing_cols:
                    messagebox.showerror("Error", f"Missing required columns in {name}: {', '.join(missing_cols)}")
                    self.status_var.set(f"Error: Missing columns in {name}")
                    return
            
            # Apply time synchronization
            # Create synchronized versions of the dataframes
            self.kymo_data['time_sync'] = self.kymo_data['time'] + time_offset
            
            # Convert distance units if needed
            distance_scale = float(self.distance_scale_var.get())
            if 'ssb_position' in self.kymo_data.columns:
                self.kymo_data['ssb_position_scaled'] = self.kymo_data['ssb_position'] * distance_scale
            
            # Plot synchronized data
            self.plot_synchronized_data()
            
            # Update status
            self.status_var.set("Data loaded and synchronized successfully")
            
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred while loading and synchronizing data: {str(e)}")
            self.status_var.set(f"Error in data synchronization: {str(e)}")
    
    def plot_synchronized_data(self):
        """Plot synchronized OT and kymograph data"""
        try:
            # Clear previous plots
            self.interaction_fig.clear()
            
            # Create subplots
            gs = self.interaction_fig.add_gridspec(2, 1, height_ratios=[1, 1])
            ax1 = self.interaction_fig.add_subplot(gs[0])
            ax2 = self.interaction_fig.add_subplot(gs[1], sharex=ax1)
            
            # Plot DNAp trace from OT data
            ax1.plot(self.ot_combined_data['time'], self.ot_combined_data['ssDNA/dsDNA junction'], 
                   'g-', label='DNAp (OT data)')
            ax1.set_ylabel('DNAp Position (μm)')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.6)
            
            # Plot SSB trace from kymograph data
            ax2.plot(self.kymo_data['time_sync'], self.kymo_data['ssb_position_scaled'], 
                   'r-', label='SSB (Kymograph)')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('SSB Position (μm)')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.6)
            
            # Set title
            self.interaction_fig.suptitle('Synchronized DNAp and SSB Trajectories')
            self.interaction_fig.tight_layout()
            
            # Draw the figure
            self.interaction_canvas.draw()
            
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred while plotting synchronized data: {str(e)}")
    
    def analyze_dnap_ssb_interaction(self):
        """Analyze interaction between DNAp and SSB from synchronized data"""
        try:
            # Update status
            self.status_var.set("Analyzing DNAp-SSB interaction...")
            self.master.update_idletasks()
            
            # Check if data is loaded
            if not hasattr(self, 'ot_combined_data') or not hasattr(self, 'kymo_data'):
                messagebox.showerror("Error", "Please load and synchronize data first")
                self.status_var.set("Error: Data not loaded")
                return
            
            # Interpolate data to common time points
            # Create a common time base for both datasets
            combined_times = np.sort(np.unique(np.concatenate([
                self.ot_combined_data['time'].values,
                self.kymo_data['time_sync'].values
            ])))
            
            # Interpolate both datasets to the common time base
            from scipy.interpolate import interp1d
            
            # Interpolate DNAp position
            dnap_interp = interp1d(
                self.ot_combined_data['time'].values,
                self.ot_combined_data['ssDNA/dsDNA junction'].values,
                bounds_error=False,
                fill_value="extrapolate"
            )
            
            # Interpolate SSB position
            ssb_interp = interp1d(
                self.kymo_data['time_sync'].values,
                self.kymo_data['ssb_position_scaled'].values,
                bounds_error=False,
                fill_value="extrapolate"
            )
            
            # Create interpolated values
            dnap_positions = dnap_interp(combined_times)
            ssb_positions = ssb_interp(combined_times)
            
            # Calculate distance between DNAp and SSB
            distances = np.abs(dnap_positions - ssb_positions)
            
            # Create a new figure for analysis
            self.interaction_fig.clear()
            gs = self.interaction_fig.add_gridspec(2, 1, height_ratios=[1, 1])
            
            # Upper plot: Show both trajectories on the same plot
            ax1 = self.interaction_fig.add_subplot(gs[0])
            ax1.plot(combined_times, dnap_positions, 'g-', label='DNAp Trajectory')
            ax1.plot(combined_times, ssb_positions, 'r-', label='SSB Trajectory')
            ax1.set_ylabel('Position (μm)')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.6)
            
            # Lower plot: Distance between DNAp and SSB
            ax2 = self.interaction_fig.add_subplot(gs[1], sharex=ax1)
            ax2.plot(combined_times, distances, 'b-', label='DNAp-SSB Distance')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Distance (μm)')
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.6)
            
            # Set title and adjust layout
            self.interaction_fig.suptitle('DNAp-SSB Interaction Analysis')
            self.interaction_fig.tight_layout()
            
            # Draw the figure
            self.interaction_canvas.draw()
            
            # Store the analysis results for export
            self.interaction_results = pd.DataFrame({
                'time': combined_times,
                'dnap_position': dnap_positions,
                'ssb_position': ssb_positions,
                'distance': distances
            })
            
            # Update status
            self.status_var.set("DNAp-SSB interaction analysis completed")
            
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred during interaction analysis: {str(e)}")
            self.status_var.set(f"Error in interaction analysis: {str(e)}")
    
    def export_interaction_results(self):
        """Export DNAp-SSB interaction analysis results"""
        try:
            # Update status
            self.status_var.set("Exporting interaction results...")
            self.master.update_idletasks()
            
            # Check if analysis has been performed
            if not hasattr(self, 'interaction_results'):
                messagebox.showerror("Error", "Please perform interaction analysis first")
                self.status_var.set("Error: No interaction results to export")
                return
            
            # Prompt for save location
            save_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")]
            )
            
            if not save_path:
                self.status_var.set("Export cancelled")
                return
            
            # Save results
            if save_path.endswith('.xlsx'):
                self.interaction_results.to_excel(save_path, index=False)
            else:
                self.interaction_results.to_csv(save_path, index=False)
            
            # Also save the current figure
            fig_path = os.path.splitext(save_path)[0] + "_plot.png"
            self.interaction_fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            
            # Update status
            messagebox.showinfo("Success", f"Interaction results exported to {save_path}")
            self.status_var.set(f"Interaction results exported successfully")
            
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred while exporting results: {str(e)}")
            self.status_var.set(f"Error exporting results: {str(e)}")