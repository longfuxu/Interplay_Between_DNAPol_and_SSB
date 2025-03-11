"""
Main window class for the Correlation Image Force Analyzer.
"""
import tkinter as tk
from tkinter import ttk
import sys
import os

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

# Import UI component creation methods
from gui.ui_components import (
    create_roi_frame, create_file_frame, create_display_frame,
    create_analysis_frame, create_status_frame, create_image_display,
    select_tdms_file
)

class CorrelationImageForceAnalyzer:
    """
    Main class for the Correlation Image Force Analyzer application.
    
    This class integrates all the methods from the methods package and
    provides a graphical user interface for analyzing correlation images
    and force data.
    """
    
    # Add the methods from the methods package
    update_contrast_from_slider = update_contrast_from_slider
    display_single_channel = display_single_channel
    display_image_left = display_image_left
    save_high_res = save_high_res
    
    select_kymo_file = select_kymo_file
    read_kymo_file = read_kymo_file
    select_trace_file = select_trace_file
    read_trace_file = read_trace_file
    read_tdms_file = read_tdms_file
    export_data = export_data
    export_distance_data = export_distance_data
    
    plot_time_vs_eed_and_fork_front_popout = plot_time_vs_eed_and_fork_front_popout
    plot_time_vs_fork_front_reverse_popout = plot_time_vs_fork_front_reverse_popout
    force_image_analyzer = force_image_analyzer
    force_image_analyzer_reverse = force_image_analyzer_reverse
    display_intensity_with_unwinding = display_intensity_with_unwinding
    detect_ssb_trajectories = detect_ssb_trajectories
    plot_ssb_trajectories = plot_ssb_trajectories
    plot_dnap_ssb = plot_dnap_ssb
    
    first_derivative = first_derivative
    segment_data = segment_data
    process_segments = process_segments
    segment_distance = segment_distance
    export_segments = export_segments
    calculate_dnap_ssb_distance = calculate_dnap_ssb_distance
    
    update_channel = update_channel
    on_entry_focus = on_entry_focus
    on_entry_focus_out = on_entry_focus_out
    on_roi_entry = on_roi_entry
    onselect_rect = onselect_rect
    on_image_click = on_image_click
    
    # Add the UI component creation methods
    create_roi_frame = create_roi_frame
    create_file_frame = create_file_frame
    create_display_frame = create_display_frame
    create_analysis_frame = create_analysis_frame
    create_status_frame = create_status_frame
    create_image_display = create_image_display
    select_tdms_file = select_tdms_file
    
    def __init__(self, master):
        """
        Initialize the application.
        
        Parameters:
        -----------
        master : tk.Tk
            The root window
        """
        self.master = master
        master.title("Correlation Image Force Analyzer")
        
        # Configure master grid layout
        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=0)  # Left panel (doesn't resize)
        master.grid_columnconfigure(1, weight=1)  # Right panel (resizes)
        
        # Initialize variables
        self.img = None
        self.time_per_line = None
        self.px_size = None
        self.trace = None
        self.intensity_data = None
        self.selected_channel = 'Blue'
        self.rect_selector = None
        self.active_entry = None  # To track focused entry
        self.contrast_max = 98  # Initial contrast value
        self.last_overlap_type = None  # 'overlap' or 'overlap_reverse'
        self.updating_display = False  # Prevent recursive updates
        
        # SSB trajectory variables
        self.traces = None
        self.roi_coords = {'start_x': None, 'end_x': None, 'start_y': None, 'end_y': None}
        self.distance_data = None
        self.segmented_data = None
        self.time_idx_smooth_s = None
        self.coordinate_idx_smooth_um = None
        self.trace_time_s_filter = None
        self.position_um_filter = None
        self.smoothed_traces = None
        self.dnap_trace = None

        # Configure styles for ttk
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#ccc")
        style.configure("TLabel", padding=6, font=('Helvetica', 10))
        style.configure("TEntry", padding=6)
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabelframe", background="#f0f0f0", font=('Helvetica', 10, 'bold'))
        style.configure("TLabelframe.Label", font=('Helvetica', 10, 'bold'))
        
        # Create left panel (commands)
        self.left_panel = ttk.Frame(master, padding=10)
        self.left_panel.grid(row=0, column=0, sticky="nsew")
        
        # Create right panel (image display)
        self.right_panel = ttk.Frame(master, padding=10)
        self.right_panel.grid(row=0, column=1, sticky="nsew")
        self.right_panel.grid_rowconfigure(0, weight=1)
        self.right_panel.grid_columnconfigure(0, weight=1)
        
        # Create app title
        title_frame = ttk.Frame(self.left_panel)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(title_frame, text="Correlation Image Force Analyzer", 
                 font=('Helvetica', 14, 'bold')).pack(side=tk.TOP)
        ttk.Separator(self.left_panel).pack(fill=tk.X, pady=5)
        
        # Create UI components in the left panel
        self.create_file_frame(self.left_panel)
        self.create_roi_frame(self.left_panel)
        self.create_display_frame(self.left_panel)
        self.create_analysis_frame(self.left_panel)
        
        # Create image display in the right panel
        self.create_image_display(self.right_panel)
        
        # Create status bar at the bottom of the main window
        self.status_frame = ttk.Frame(master)
        self.status_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.status_label = ttk.Label(self.status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Set the status
        self.status_label.configure(text="Ready") 