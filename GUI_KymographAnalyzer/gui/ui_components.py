"""
UI component creation methods for the Kymograph Analyzer.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Try to import ttkthemes for modern styling if available
try:
    from ttkthemes import ThemedTk, ThemedStyle
    HAS_TTKTHEMES = True
except ImportError:
    HAS_TTKTHEMES = False

def set_modern_style(root):
    """Apply a modern style to the UI if ttkthemes is available"""
    if HAS_TTKTHEMES:
        if not isinstance(root, ThemedTk):
            style = ThemedStyle(root)
            style.set_theme("arc")  # Modern flat theme
        else:
            root.set_theme("arc")
    
    # Configure colors for a modern look regardless of ttkthemes
    bg_color = "#f5f6f7"
    accent_color = "#4a6fd4"
    
    root.configure(background=bg_color)
    style = ttk.Style()
    style.configure(".", font=("Helvetica", 10))
    style.configure("TLabelframe", background=bg_color)
    style.configure("TLabelframe.Label", background=bg_color, foreground="#333333", font=("Helvetica", 11, "bold"))
    style.configure("TButton", background=accent_color, foreground="#ffffff")
    style.map("TButton", background=[("active", "#5a7fe4")])
    
    return style

def create_tab_control(parent):
    """Create a tabbed interface"""
    notebook = ttk.Notebook(parent)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    tabs = {
        "data_loading": ttk.Frame(notebook),
        "visualization": ttk.Frame(notebook),
        "analysis": ttk.Frame(notebook),
        "results": ttk.Frame(notebook)
    }
    
    notebook.add(tabs["data_loading"], text="Data Loading")
    notebook.add(tabs["visualization"], text="Visualization")
    notebook.add(tabs["analysis"], text="Analysis")
    notebook.add(tabs["results"], text="Results")
    
    # Configure tab grids
    for tab in tabs.values():
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)
    
    return notebook, tabs

def create_matplotlib_figure(parent, title="", xlabel="", ylabel="", use_grid=False, figsize=(6, 4), toolbar=True):
    """Create a matplotlib figure embedded in a tkinter frame"""
    frame = ttk.Frame(parent)
    if not use_grid:
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    fig = Figure(figsize=figsize, dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    if use_grid:
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
    else:
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    if toolbar:
        toolbar_frame = ttk.Frame(frame)
        if use_grid:
            toolbar_frame.grid(row=1, column=0, sticky="ew")
        else:
            toolbar_frame.pack(fill=tk.X)
            
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
    
    return fig, ax, canvas, frame

def create_channel_controls(parent, selected_channel, update_callback):
    """Create channel selection controls with visual color indicators"""
    frame = ttk.LabelFrame(parent, text="Channel Selection")
    frame.pack(fill=tk.X, padx=5, pady=5)
    
    inner_frame = ttk.Frame(frame)
    inner_frame.pack(fill=tk.X, padx=5, pady=5)
    
    channel_var = tk.StringVar(value=selected_channel)
    
    # Channel colors
    colors = {"Red": "#ff6b6b", "Green": "#51cf66", "Blue": "#339af0"}
    
    for channel in ["Red", "Green", "Blue"]:
        channel_frame = ttk.Frame(inner_frame)
        channel_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Create color indicator
        color_indicator = tk.Canvas(channel_frame, width=15, height=15, 
                                   background=colors[channel], highlightthickness=1)
        color_indicator.pack(side=tk.LEFT, padx=(0, 5))
        
        # Create radiobutton
        rb = ttk.Radiobutton(channel_frame, text=channel, value=channel, 
                           variable=channel_var,
                           command=lambda c=channel: update_callback(c))
        rb.pack(side=tk.LEFT)
    
    # Create contrast slider
    contrast_frame = ttk.Frame(frame)
    contrast_frame.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Label(contrast_frame, text="Contrast:").pack(side=tk.LEFT, padx=5)
    contrast_var = tk.DoubleVar(value=98)
    contrast_slider = ttk.Scale(contrast_frame, from_=0, to=100, 
                              variable=contrast_var, orient=tk.HORIZONTAL)
    contrast_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    # Create a label to show the current value
    value_label = ttk.Label(contrast_frame, text="98%")
    value_label.pack(side=tk.LEFT, padx=5)
    
    # Update the label when the slider changes
    def update_contrast_label(event=None):
        value_label.config(text=f"{int(contrast_var.get())}%")
    
    contrast_slider.bind("<Motion>", update_contrast_label)
    
    return channel_var, contrast_var

def create_roi_controls(parent, roi_callback, add_minimap=True):
    """Create enhanced ROI selection controls with minimap preview"""
    frame = ttk.LabelFrame(parent, text="ROI Selection")
    frame.pack(fill=tk.X, padx=5, pady=5)
    
    control_frame = ttk.Frame(frame)
    control_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Create a 2x2 grid for the ROI entries
    entries = {}
    for label, row, col in [("X Left:", 0, 0), ("X Right:", 0, 2),
                           ("Y Top:", 1, 0), ("Y Bottom:", 1, 2)]:
        ttk.Label(control_frame, text=label).grid(row=row, column=col, padx=5, pady=2, sticky="e")
        entry = ttk.Entry(control_frame, width=10)
        entry.grid(row=row, column=col+1, padx=5, pady=2, sticky="w")
        key = label.replace(":", "").lower()
        entries[key] = entry
        
        # Add tooltip
        create_tooltip(entry, f"Enter the {key} coordinate for ROI selection")
    
    button_frame = ttk.Frame(frame)
    button_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Apply button
    apply_button = ttk.Button(button_frame, text="Apply ROI", 
                            command=lambda: roi_callback(
                                x_left=float(entries["x left"].get()),
                                x_right=float(entries["x right"].get()),
                                y_top=float(entries["y top"].get()),
                                y_bottom=float(entries["y bottom"].get())
                            ))
    apply_button.pack(side=tk.LEFT, padx=5)
    
    # Reset button
    reset_button = ttk.Button(button_frame, text="Reset", 
                             command=lambda: reset_roi_entries(entries))
    reset_button.pack(side=tk.LEFT, padx=5)
    
    # Add minimap if requested
    if add_minimap:
        minimap_frame = ttk.Frame(frame)
        minimap_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(minimap_frame, text="ROI Preview:").pack(anchor=tk.W, padx=5, pady=2)
        
        canvas = tk.Canvas(minimap_frame, width=200, height=100, bg="lightgray", highlightthickness=1)
        canvas.pack(padx=5, pady=5)
        
        # This will be updated when the image is loaded
        entries["minimap_canvas"] = canvas
    
    return entries

def reset_roi_entries(entries):
    """Reset ROI entries to empty"""
    for key in ["x left", "x right", "y top", "y bottom"]:
        if key in entries:
            entries[key].delete(0, tk.END)

def create_parameter_controls(parent, params, sections=None):
    """Create parameter input controls with optional sections"""
    frame = ttk.LabelFrame(parent, text="Analysis Parameters")
    frame.pack(fill=tk.X, padx=5, pady=5)
    
    entries = {}
    
    if sections:
        # Create a notebook for parameter sections
        param_notebook = ttk.Notebook(frame)
        param_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        for section_name, section_params in sections.items():
            section_frame = ttk.Frame(param_notebook)
            param_notebook.add(section_frame, text=section_name)
            
            for param, default in section_params.items():
                param_frame = ttk.Frame(section_frame)
                param_frame.pack(fill=tk.X, padx=5, pady=2)
                
                ttk.Label(param_frame, text=f"{param}:").pack(side=tk.LEFT, padx=5)
                entry = ttk.Entry(param_frame, width=10)
                entry.insert(0, str(default))
                entry.pack(side=tk.LEFT, padx=5)
                entries[param] = entry
    else:
        # Simple flat list of parameters
        for param, default in params.items():
            param_frame = ttk.Frame(frame)
            param_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(param_frame, text=f"{param}:").pack(side=tk.LEFT, padx=5)
            entry = ttk.Entry(param_frame, width=10)
            entry.insert(0, str(default))
            entry.pack(side=tk.LEFT, padx=5)
            entries[param] = entry
            
            # Add tooltip
            create_tooltip(entry, f"Parameter for {param}")
    
    return entries

def create_file_selector(parent, label, callback, file_types=None):
    """Create an enhanced file selection row with label, entry, and browse button"""
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Label(frame, text=label).pack(side=tk.LEFT, padx=5)
    entry = ttk.Entry(frame)
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    if file_types is None:
        file_types = [("All files", "*.*")]
    
    def browse():
        filename = filedialog.askopenfilename(filetypes=file_types)
        if filename:
            entry.delete(0, tk.END)
            entry.insert(0, filename)
            callback(filename)
    
    browse_button = ttk.Button(frame, text="Browse", command=browse)
    browse_button.pack(side=tk.LEFT, padx=5)
    
    # Add clear button
    def clear_entry():
        entry.delete(0, tk.END)
    
    clear_button = ttk.Button(frame, text="Clear", command=clear_entry)
    clear_button.pack(side=tk.LEFT, padx=5)
    
    return entry

def create_plot_frame(parent, title, add_toolbar=True, figsize=(6, 4)):
    """Create an enhanced frame for plotting with title"""
    frame = ttk.LabelFrame(parent, text=title)
    frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Create a frame inside the labelframe to hold the figure
    plot_frame = ttk.Frame(frame)
    plot_frame.pack(fill=tk.BOTH, expand=True)
    
    # Create figure with specified size
    fig = Figure(figsize=figsize, dpi=100)
    ax = fig.add_subplot(111)
    
    # Create canvas and pack it into the frame
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Add navigation toolbar if requested
    if add_toolbar:
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill=tk.X, side=tk.BOTTOM)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
    
    return frame, fig, ax, canvas

def create_tooltip(widget, text):
    """Create a tooltip for a widget"""
    def enter(event):
        x, y, _, _ = widget.bbox("insert")
        x += widget.winfo_rootx() + 25
        y += widget.winfo_rooty() + 25
        
        # Create a toplevel window
        tooltip = tk.Toplevel(widget)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(f"+{x}+{y}")
        
        # Add a label to the tooltip
        label = ttk.Label(tooltip, text=text, background="#ffffe0", relief="solid", borderwidth=1, wraplength=180)
        label.pack(padx=2, pady=2)
        
        # Store the tooltip in the widget
        widget.tooltip = tooltip
    
    def leave(event):
        if hasattr(widget, "tooltip"):
            widget.tooltip.destroy()
    
    widget.bind("<Enter>", enter)
    widget.bind("<Leave>", leave)

def create_button_with_icon(parent, text, command, icon=None, compound=tk.LEFT):
    """Create a button with an optional icon"""
    button = ttk.Button(parent, text=text, command=command, compound=compound)
    
    if icon:
        try:
            # Try to load the icon
            img = tk.PhotoImage(file=icon)
            button.configure(image=img)
            button.image = img  # Keep a reference to prevent garbage collection
        except tk.TclError:
            print(f"Could not load icon: {icon}")
    
    return button

def create_status_bar(parent):
    """Create a status bar with progress indicator"""
    frame = ttk.Frame(parent, relief=tk.SUNKEN, border=1)
    
    # Status text
    status_var = tk.StringVar(value="Ready")
    status_label = ttk.Label(frame, textvariable=status_var, anchor=tk.W)
    status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=2)
    
    # Progress bar (initially hidden)
    progress_var = tk.DoubleVar(value=0)
    progress_bar = ttk.Progressbar(frame, orient=tk.HORIZONTAL, 
                                 length=100, mode='determinate', 
                                 variable=progress_var)
    progress_bar.pack(side=tk.RIGHT, padx=5, pady=2)
    progress_bar.pack_forget()  # Hide initially
    
    return frame, status_var, progress_var, progress_bar

def create_export_controls(parent, export_callback):
    """Create controls for exporting data"""
    frame = ttk.LabelFrame(parent, text="Export Options")
    frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Export format options
    format_frame = ttk.Frame(frame)
    format_frame.pack(fill=tk.X, padx=5, pady=2)
    
    ttk.Label(format_frame, text="Format:").pack(side=tk.LEFT, padx=5)
    format_var = tk.StringVar(value="CSV")
    
    for fmt in ["CSV", "Excel", "PNG", "PDF"]:
        ttk.Radiobutton(format_frame, text=fmt, value=fmt, 
                      variable=format_var).pack(side=tk.LEFT, padx=5)
    
    # Export button
    export_button = ttk.Button(frame, text="Export Data", 
                             command=lambda: export_callback(format_var.get()))
    export_button.pack(anchor=tk.E, padx=5, pady=5)
    
    return format_var

def create_help_button(parent, help_text):
    """Create a help button that shows a message dialog with help text"""
    def show_help():
        messagebox.showinfo("Help", help_text)
    
    help_button = ttk.Button(parent, text="?", width=2, command=show_help)
    
    return help_button

