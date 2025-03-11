"""
UI component creation methods for the Correlation Image Force Analyzer.
"""
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

def create_roi_frame(self, parent_frame):
    """Create the ROI frame with input fields"""
    roi_frame = ttk.LabelFrame(parent_frame, text="Region of Interest (ROI)")
    roi_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
    
    # Create a grid layout for the ROI inputs
    roi_grid = ttk.Frame(roi_frame)
    roi_grid.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
    
    # X-axis ROI
    ttk.Label(roi_grid, text="X Left:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
    self.roi_x_left_entry = ttk.Entry(roi_grid, width=10)
    self.roi_x_left_entry.grid(row=0, column=1, padx=5, pady=2)
    self.roi_x_left_entry.bind("<FocusIn>", self.on_entry_focus)
    self.roi_x_left_entry.bind("<FocusOut>", self.on_entry_focus_out)
    self.roi_x_left_entry.bind("<Return>", self.on_roi_entry)
    
    ttk.Label(roi_grid, text="X Right:").grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)
    self.roi_x_right_entry = ttk.Entry(roi_grid, width=10)
    self.roi_x_right_entry.grid(row=0, column=3, padx=5, pady=2)
    self.roi_x_right_entry.bind("<FocusIn>", self.on_entry_focus)
    self.roi_x_right_entry.bind("<FocusOut>", self.on_entry_focus_out)
    self.roi_x_right_entry.bind("<Return>", self.on_roi_entry)
    
    # Y-axis ROI
    ttk.Label(roi_grid, text="Y Top:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
    self.roi_y_top_entry = ttk.Entry(roi_grid, width=10)
    self.roi_y_top_entry.grid(row=1, column=1, padx=5, pady=2)
    self.roi_y_top_entry.bind("<FocusIn>", self.on_entry_focus)
    self.roi_y_top_entry.bind("<FocusOut>", self.on_entry_focus_out)
    self.roi_y_top_entry.bind("<Return>", self.on_roi_entry)
    
    ttk.Label(roi_grid, text="Y Bottom:").grid(row=1, column=2, padx=5, pady=2, sticky=tk.W)
    self.roi_y_bottom_entry = ttk.Entry(roi_grid, width=10)
    self.roi_y_bottom_entry.grid(row=1, column=3, padx=5, pady=2)
    self.roi_y_bottom_entry.bind("<FocusIn>", self.on_entry_focus)
    self.roi_y_bottom_entry.bind("<FocusOut>", self.on_entry_focus_out)
    self.roi_y_bottom_entry.bind("<Return>", self.on_roi_entry)
    
    # Apply ROI button
    apply_roi_button = ttk.Button(roi_grid, text="Apply ROI", command=self.display_image_left)
    apply_roi_button.grid(row=0, column=4, rowspan=2, padx=10, pady=2)

def create_file_frame(self, parent_frame):
    """Create the file selection frame"""
    file_frame = ttk.LabelFrame(parent_frame, text="File Selection")
    file_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
    
    # Create grid layout for files
    file_grid = ttk.Frame(file_frame)
    file_grid.pack(fill=tk.X, expand=True, padx=5, pady=5)
    
    # Kymograph file selection
    ttk.Label(file_grid, text="Kymograph File:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
    self.kymo_file_label = ttk.Label(file_grid, text="None", foreground="gray")
    self.kymo_file_label.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
    kymo_button = ttk.Button(file_grid, text="Select Kymograph", command=self.select_kymo_file)
    kymo_button.grid(row=0, column=2, sticky=tk.E, padx=5, pady=2)
    
    # Trace file selection
    ttk.Label(file_grid, text="DNAp Trace File:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
    self.trace_file_label = ttk.Label(file_grid, text="None", foreground="gray")
    self.trace_file_label.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
    trace_button = ttk.Button(file_grid, text="Select Trace", command=self.select_trace_file)
    trace_button.grid(row=1, column=2, sticky=tk.E, padx=5, pady=2)

def create_display_frame(self, parent_frame):
    """Create the display options frame"""
    display_frame = ttk.LabelFrame(parent_frame, text="Display Options")
    display_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
    
    # Channel and control layout
    control_frame = ttk.Frame(display_frame)
    control_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
    
    # Channel selection
    ttk.Label(control_frame, text="Channel:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
    self.channel_var = tk.StringVar(value=self.selected_channel)
    self.channel_combobox = ttk.Combobox(control_frame, textvariable=self.channel_var, 
                                       values=["Red", "Green", "Blue"], state="disabled", width=15)
    self.channel_combobox.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
    
    # Apply channel button
    apply_channel_button = ttk.Button(control_frame, text="Apply Channel", 
                                    command=lambda: self.display_single_channel(self.channel_var.get()), 
                                    width=15)
    apply_channel_button.grid(row=0, column=2, padx=5, pady=2)
    
    # Contrast adjustment
    ttk.Label(control_frame, text="Contrast:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
    self.contrast_label = ttk.Label(control_frame, text=f"{int(self.contrast_max)}%")
    self.contrast_label.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
    
    self.contrast_slider = ttk.Scale(control_frame, from_=50, to=100, orient=tk.HORIZONTAL, 
                                   value=self.contrast_max, command=self.update_contrast_from_slider)
    self.contrast_slider.grid(row=2, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=2)
    
    # Display buttons
    button_frame = ttk.Frame(display_frame)
    button_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
    
    display_button = ttk.Button(button_frame, text="Display Kymograph", 
                              command=self.display_image_left, width=20)
    display_button.pack(side=tk.LEFT, padx=5, pady=2)

def create_analysis_frame(self, parent_frame):
    """Create the analysis options frame"""
    analysis_frame = ttk.LabelFrame(parent_frame, text="Analysis Options")
    analysis_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
    
    # Basic analysis buttons
    analysis_grid = ttk.Frame(analysis_frame)
    analysis_grid.pack(fill=tk.X, expand=True, padx=5, pady=5)
    
    plot_eed_button = ttk.Button(analysis_grid, text="Plot Junction Position", 
                                command=self.plot_time_vs_eed_and_fork_front_popout, width=20)
    plot_eed_button.grid(row=0, column=0, padx=5, pady=3)
    
    plot_reverse_button = ttk.Button(analysis_grid, text="Plot Junction (Reverse)", 
                                    command=self.plot_time_vs_fork_front_reverse_popout, width=20)
    plot_reverse_button.grid(row=0, column=1, padx=5, pady=3)
    
    # Force-image analysis
    force_image_button = ttk.Button(analysis_grid, text="Force-Image Analysis", 
                                   command=self.force_image_analyzer, width=20)
    force_image_button.grid(row=1, column=0, padx=5, pady=3)
    
    force_image_reverse_button = ttk.Button(analysis_grid, text="Force-Image (Reverse)", 
                                          command=self.force_image_analyzer_reverse, width=20)
    force_image_reverse_button.grid(row=1, column=1, padx=5, pady=3)
    
    intensity_button = ttk.Button(analysis_grid, text="Intensity Analysis", 
                                 command=self.display_intensity_with_unwinding, width=20)
    intensity_button.grid(row=2, column=0, padx=5, pady=3)
    
    # SSB analysis buttons
    ssb_frame = ttk.LabelFrame(analysis_frame, text="SSB Analysis")
    ssb_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
    
    ssb_grid = ttk.Frame(ssb_frame)
    ssb_grid.pack(fill=tk.X, expand=True, padx=5, pady=5)
    
    detect_ssb_button = ttk.Button(ssb_grid, text="Detect SSB Trajectories", 
                                  command=self.detect_ssb_trajectories, width=20)
    detect_ssb_button.grid(row=0, column=0, padx=5, pady=3)
    
    plot_ssb_button = ttk.Button(ssb_grid, text="Plot SSB Trajectories", 
                               command=self.plot_ssb_trajectories, width=20)
    plot_ssb_button.grid(row=0, column=1, padx=5, pady=3)
    
    dnap_ssb_button = ttk.Button(ssb_grid, text="Plot DNAp-SSB", 
                               command=self.plot_dnap_ssb, width=20)
    dnap_ssb_button.grid(row=1, column=0, padx=5, pady=3)
    
    # Export data
    export_button = ttk.Button(ssb_grid, text="Export Data", 
                              command=self.export_data, width=20)
    export_button.grid(row=1, column=1, padx=5, pady=3)

def create_status_frame(self, parent_frame):
    """Create the status frame"""
    status_frame = ttk.Frame(parent_frame)
    status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
    
    self.status_label = ttk.Label(status_frame, text="Ready", font=('Helvetica', 10))
    self.status_label.pack(side=tk.LEFT, padx=5)

def create_image_display(self, parent_frame):
    """Create the image display frame"""
    try:
        # Create a container frame for the image display
        display_container = ttk.Frame(parent_frame, padding=10)
        display_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Create a figure for the image
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Kymograph - No Data Loaded")
        self.ax.set_xlabel("Time (px)")
        self.ax.set_ylabel("Position (px)")
        self.ax.text(0.5, 0.5, "Please load a kymograph file", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=self.ax.transAxes, fontsize=14)
        
        # Create a canvas for the figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=display_container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar
        self.toolbar_frame = ttk.Frame(display_container)
        self.toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        
        # Connect the mouse click event
        self.canvas.mpl_connect('button_press_event', self.on_image_click)
        
        # Initialize a placeholder for colorbar
        self.colorbar = None
        
    except Exception as e:
        print(f"Error creating image display: {e}")

def select_tdms_file(self):
    """Open a file dialog to select a TDMS file"""
    filename = tk.filedialog.askopenfilename(
        title="Select TDMS File",
        filetypes=[("TDMS files", "*.tdms"), ("All files", "*.*")]
    )
    if filename:
        self.read_tdms_file(filename) 