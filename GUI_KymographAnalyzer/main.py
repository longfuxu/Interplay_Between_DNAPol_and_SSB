#!/usr/bin/env python3
"""
Correlation Image Force Analyzer

This script runs the Correlation Image Force Analyzer application, which
provides a graphical user interface for analyzing correlation images and
force data.

The application allows for loading kymograph TDMS files and DNAp trace files,
and analyzing the correlation between them, with a particular focus on
detecting and analyzing SSB trajectories.

Usage:
    python main.py
"""
import tkinter as tk
import sys
import os
import traceback
import matplotlib
# Use TkAgg backend for matplotlib
matplotlib.use('TkAgg')

# Add the current directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main window class
from gui.main_window import CorrelationImageForceAnalyzer

def show_error(exception_type, exception_value, exception_traceback):
    """Display uncaught exceptions in a message box"""
    # Format the traceback
    error_msg = "".join(traceback.format_exception(exception_type, exception_value, exception_traceback))
    
    # Create a simple error dialog
    error_window = tk.Toplevel()
    error_window.title("Uncaught Exception")
    error_window.geometry("600x400")
    
    # Add a label with the error message
    tk.Label(error_window, text="An unexpected error occurred:").pack(padx=10, pady=10)
    
    # Add a text widget with the traceback
    text_widget = tk.Text(error_window, wrap=tk.WORD, height=15, width=80)
    text_widget.insert(tk.END, error_msg)
    text_widget.config(state=tk.DISABLED)
    text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    
    # Add a dismiss button
    tk.Button(error_window, text="Dismiss", command=error_window.destroy).pack(pady=10)
    
    # Print to console as well
    print(f"Uncaught exception: {error_msg}")

def main():
    """Run the application"""
    try:
        # Create the root window
        root = tk.Tk()
        
        # Set window title and icon
        root.title("Correlation Image Force Analyzer")
        
        # Configure root grid
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        
        # Set the default exception handler
        sys.excepthook = show_error
        
        # Create the application
        app = CorrelationImageForceAnalyzer(root)
        
        # Configure window size and position
        root.geometry("1800x1000")
        root.update_idletasks()  # Update "idle" tasks to get accurate window dimensions
        
        # Center the window on the screen
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f"{width}x{height}+{x}+{y}")
        
        # Run the main loop
        root.mainloop()
    
    except Exception as e:
        # If an exception occurs during setup, show it
        if 'root' in locals() and root:
            show_error(type(e), e, e.__traceback__)
        else:
            # If root doesn't exist yet, just print to console
            traceback.print_exc()

if __name__ == "__main__":
    main() 