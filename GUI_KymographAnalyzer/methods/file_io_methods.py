"""
File I/O methods for the Correlation Image Force Analyzer.
"""
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
from nptdms import TdmsFile
import os
import cv2
import lumicks.pylake as lk

def select_kymo_file(self):
    """Open a file dialog to select a kymograph file"""
    filename = filedialog.askopenfilename(
        title="Select Kymograph File",
        filetypes=[("TDMS files", "*.tdms"), ("HDF5 files", "*.h5"), ("All files", "*.*")]
    )
    if filename:
        self.read_kymo_file(filename)
        # Update the status
        self.status_label.configure(text=f"Loaded kymograph: {os.path.basename(filename)}")

def read_kymo_file(self, filename):
    """Read the kymograph file (supports both TDMS and H5 formats)"""
    try:
        if filename.lower().endswith('.tdms'):
            # Read TDMS file (similar to the notebook)
            metadata = TdmsFile.read_metadata(filename)
            width = metadata.properties['Pixels per line']
            self.px_size = float(metadata.properties['Scan Command.Scan Command.scanning_axes.0.pix_size_nm']) / 1000  # Convert to µm
            px_dwell_time = float(metadata.properties['Scan Command.PI Fast Scan Command.pixel_dwell_time_ms'])
            
            tdms_file = TdmsFile(filename)
            kymo_time = tdms_file['Data']['Time (ms)'][:]
            kymo_time = np.array([int(i) for i in kymo_time])
            kymo_position = tdms_file['Data']['Actual position X (um)'][:]
            kymo_position = np.array([int(i) for i in kymo_position])
            height = len(kymo_time) / width
            self.time_per_line = kymo_time[-1] / height  # ms
            
            chn_r = tdms_file['Data']['Pixel ch 1'][:]
            chn_r = np.array([int(i) for i in chn_r])
            chn_g = tdms_file['Data']['Pixel ch 2'][:]
            chn_g = np.array([int(i) for i in chn_g])
            chn_b = tdms_file['Data']['Pixel ch 3'][:]
            chn_b = np.array([int(i) for i in chn_b])
            
            chn_rgb = np.vstack((chn_r, chn_g, chn_b)).T
            self.img = chn_rgb.reshape((int(height), int(width), 3))
            self.img = self.img.transpose((1, 0, 2))
            self.img = self.img.astype(np.uint16)
            
        elif filename.lower().endswith('.h5'):
            # Read H5 file using pylake
            kymo = lk.kymo.Kymo(filename)
            
            # Get the image data
            self.img = kymo.get_image('rgb')
            
            # Get metadata
            self.time_per_line = kymo.line_time_seconds * 1000  # Convert to ms
            self.px_size = kymo.pixelsize_um
        else:
            messagebox.showerror("Error", "Unsupported file format. Please use TDMS or H5 files.")
            return
        
        # Update the UI
        self.kymo_file_label.configure(text=f"{os.path.basename(filename)}")
        
        # Enable the channel selection
        self.channel_combobox.configure(state="readonly")
        
        # Display the image
        self.display_image_left()
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read kymograph file: {e}")

def select_trace_file(self):
    """Open a file dialog to select a trace file"""
    try:
        # Use a simpler file dialog configuration to avoid crashes
        filename = filedialog.askopenfilename(
            title="Select DNAp Trace File",
            initialdir=os.path.expanduser("~"),  # Start in home directory
            filetypes=[
                ("Excel files", "*.xlsx"),
                ("Excel files", "*.xls"),
                ("CSV files", "*.csv"),
                ("All files", "*")
            ]
        )
        
        if filename and os.path.exists(filename):
            self.read_trace_file(filename)
            # Update the status
            self.status_label.configure(text=f"Loaded trace file: {os.path.basename(filename)}")
    except Exception as e:
        messagebox.showerror("Error", f"Error in file selection: {str(e)}")
        # Print to console for debugging
        print(f"Error in file selection: {str(e)}")

def read_trace_file(self, filename):
    """Read the trace file (supports both Excel and CSV formats)"""
    try:
        # Read the trace file based on extension
        if filename.lower().endswith(('.xlsx', '.xls')):
            self.trace = pd.read_excel(filename)
        else:
            self.trace = pd.read_csv(filename)
        
        # Rename column headers if they contain 'fork' to 'junction'
        rename_dict = {}
        for col in self.trace.columns:
            if 'fork' in col.lower():
                rename_dict[col] = col.lower().replace('fork', 'junction')
        
        if rename_dict:
            self.trace = self.trace.rename(columns=rename_dict)
        
        # Check for required columns and rename if necessary
        if 'time' not in self.trace.columns and 'Time' in self.trace.columns:
            self.trace = self.trace.rename(columns={'Time': 'time'})
        
        if 'junction_position_all' not in self.trace.columns:
            # Try to find a column that might contain junction position
            possible_columns = ['junction_position', 'junction position', 'Position', 'position']
            for col in possible_columns:
                if col in self.trace.columns:
                    self.trace = self.trace.rename(columns={col: 'junction_position_all'})
                    break
            
            # If still not found, use the second column (assuming first is time)
            if 'junction_position_all' not in self.trace.columns and len(self.trace.columns) > 1:
                position_col = self.trace.columns[1]
                self.trace = self.trace.rename(columns={position_col: 'junction_position_all'})
        
        # Update the UI
        self.trace_file_label.configure(text=f"{os.path.basename(filename)}")
        
        # Display the image if both files loaded
        if self.img is not None and self.trace is not None:
            self.display_image_left()
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read trace file: {e}")

def read_tdms_file(self, filename):
    """Read a TDMS file containing force-extension data"""
    try:
        # Read the TDMS file (same as read_kymo_file for TDMS)
        if filename.lower().endswith('.tdms'):
            metadata = TdmsFile.read_metadata(filename)
            width = metadata.properties['Pixels per line']
            self.px_size = float(metadata.properties['Scan Command.Scan Command.scanning_axes.0.pix_size_nm']) / 1000  # Convert to µm
            px_dwell_time = float(metadata.properties['Scan Command.PI Fast Scan Command.pixel_dwell_time_ms'])
            
            tdms_file = TdmsFile(filename)
            kymo_time = tdms_file['Data']['Time (ms)'][:]
            kymo_time = np.array([int(i) for i in kymo_time])
            kymo_position = tdms_file['Data']['Actual position X (um)'][:]
            kymo_position = np.array([int(i) for i in kymo_position])
            height = len(kymo_time) / width
            self.time_per_line = kymo_time[-1] / height  # ms
            
            chn_r = tdms_file['Data']['Pixel ch 1'][:]
            chn_r = np.array([int(i) for i in chn_r])
            chn_g = tdms_file['Data']['Pixel ch 2'][:]
            chn_g = np.array([int(i) for i in chn_g])
            chn_b = tdms_file['Data']['Pixel ch 3'][:]
            chn_b = np.array([int(i) for i in chn_b])
            
            chn_rgb = np.vstack((chn_r, chn_g, chn_b)).T
            self.img = chn_rgb.reshape((int(height), int(width), 3))
            self.img = self.img.transpose((1, 0, 2))
            self.img = self.img.astype(np.uint16)
            
            # Update the UI
            self.kymo_file_label.configure(text=f"{os.path.basename(filename)}")
            
            # Enable the channel selection
            self.channel_combobox.configure(state="readonly")
            
            # Display the image
            self.display_image_left()
        else:
            messagebox.showerror("Error", "Unsupported file format. Please use TDMS files.")
            
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read TDMS file: {e}")

def save_image(self, fig, format='png'):
    """Save the current figure as an image file"""
    file_path = filedialog.asksaveasfilename(
        defaultextension=f".{format}",
        filetypes=[(f"{format.upper()} files", f"*.{format}"), ("All files", "*.*")]
    )
    if file_path:
        try:
            fig.savefig(file_path, format=format, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Image saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {e}")

def export_data(self):
    """Export the analyzed data to a CSV file"""
    if self.trace is None or self.img is None:
        messagebox.showinfo("Info", "Please load both kymograph and trace files first.")
        return
    
    # Ask for the output file
    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    if not file_path:
        return
    
    try:
        # Create a DataFrame with the data
        data = pd.DataFrame({
            'Time': self.trace['time'] if 'time' in self.trace.columns else self.trace['Time'],
            'Junction_Position': self.trace['junction_position_all'] if 'junction_position_all' in self.trace.columns else self.trace['Position'],
            # Add more columns as needed
        })
        
        # Save to CSV
        data.to_csv(file_path, index=False)
        
        messagebox.showinfo("Success", f"Data exported to {file_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to export data: {e}")

def export_distance_data(self, all_times, all_distances):
    """Export the distance data to a CSV file"""
    if all_times is None or all_distances is None:
        messagebox.showinfo("Info", "No distance data to export.")
        return
    
    # Ask for the output file
    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    if not file_path:
        return
    
    try:
        # Create a DataFrame with the data
        data = pd.DataFrame({
            'Time': all_times,
            'Distance': all_distances
        })
        
        # Save to CSV
        data.to_csv(file_path, index=False)
        
        messagebox.showinfo("Success", f"Distance data exported to {file_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to export distance data: {e}")

def save_all_data(self, save_dir=None):
    """Save all the data in separate CSV files"""
    if self.trace is None or self.smoothed_traces is None:
        messagebox.showinfo("Info", "Please detect SSB trajectories first.")
        return
    
    if save_dir is None:
        # Ask for the output directory
        save_dir = filedialog.askdirectory(
            title="Select Directory to Save Data"
        )
    
    if not save_dir:
        return
    
    try:
        # Check if the directory exists, if not, create it
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save the DNAp trace
        df_DNAp = pd.DataFrame({
            'time_s': self.trace_time_s_filter,
            'position_um': self.position_um_filter
        })
        df_DNAp.to_csv(os.path.join(save_dir, 'DNAp_trace.csv'), index=False)
        
        # Save each SSB trace and the corresponding distance
        for i, trace in enumerate(self.smoothed_traces):
            # Create a DataFrame and save it as a CSV file
            df_trace = pd.DataFrame({
                'time_s': trace['Time'],
                'position_um': trace['Position']
            })
            df_trace.to_csv(os.path.join(save_dir, f'trace_{i}.csv'), index=False)
            
        # Update status
        self.status_label.configure(text=f"Data saved to {save_dir}")
            
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save data: {e}") 