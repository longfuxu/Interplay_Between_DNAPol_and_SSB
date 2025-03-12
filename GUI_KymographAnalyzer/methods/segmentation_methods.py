"""
Segmentation methods for the Kymograph Analyzer.
"""
import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def first_derivative(self, x, y, window_size=3):
    """
    Calculate the first derivative while handling edge effects
    
    Parameters:
    -----------
    x : array-like
        x values
    y : array-like
        y values
    window_size : int
        Window size for derivative calculation
        
    Returns:
    --------
    array-like
        First derivative values
    """
    # Ensure x and y are numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    first_derivative_values = np.zeros(len(y))
    
    for i in range(len(y)):
        if i < window_size:
            # Forward difference for points near the start
            dy = y[i + window_size] - y[0]
            dx = x[i + window_size] - x[0]
        elif i > len(y) - window_size - 1:
            # Backward difference for points near the end
            dy = y[-1] - y[i - window_size]
            dx = x[-1] - x[i - window_size]
        else:
            # Central difference for points in the middle
            dy = y[i + window_size] - y[i - window_size]
            dx = x[i + window_size] - x[i - window_size]
            
        first_derivative_values[i] = dy / dx if dx != 0 else 0
        
    return first_derivative_values

def segment_data(self, derivative, threshold=0.01, min_segment_length=10):
    """
    Segment data based on the sign of the derivative.
    
    Parameters:
    -----------
    derivative : array-like
        The derivative of the data to segment
    threshold : float
        Threshold for considering a derivative to be zero
    min_segment_length : int
        Minimum number of points required for a segment
        
    Returns:
    --------
    segments : list of tuples
        List of (start_index, end_index) tuples for each segment
    """
    segments = []
    start_index = 0
    
    # Determine the sign of the first point, considering the threshold
    current_sign = 0 if abs(derivative[0]) < threshold else np.sign(derivative[0])
    
    for i in range(1, len(derivative)):
        # Determine the sign of the current point, considering the threshold
        sign = 0 if abs(derivative[i]) < threshold else np.sign(derivative[i])
        
        # If the sign changes, we have a new segment
        if sign != current_sign:
            # Check for very short constant segments between two segments of the same sign
            if current_sign == 0 and i - start_index < min_segment_length and len(segments) > 0:
                # Merge with previous segment
                start, _ = segments.pop()
                segments.append((start, i - 1))
            else:
                # Add the segment
                segments.append((start_index, i - 1))
                
            start_index = i
            current_sign = sign
    
    # Handle the last segment
    if len(derivative) - start_index < min_segment_length and len(segments) > 0 and current_sign == 0:
        # Last segment is very short and constant, merge with previous
        start, _ = segments.pop()
        segments.append((start, len(derivative) - 1))
    else:
        # Add the last segment
        segments.append((start_index, len(derivative) - 1))
        
    return segments

def process_segments(self, time, position, segments):
    """
    Process the segments to extract segment statistics.
    
    Parameters:
    -----------
    time : array-like
        Time data
    position : array-like
        Position data
    segments : list of tuples
        List of (start_index, end_index) tuples for each segment
        
    Returns:
    --------
    Tuple containing:
        segmented_time_start : list
            Start time of each segment
        segmented_time_end : list
            End time of each segment
        segmented_position_start : list
            Start position of each segment
        segmented_position_end : list
            End position of each segment
        segmented_rates : list
            Rate of change of position for each segment
        segmented_diffs : list
            Difference between start of current segment and end of previous segment
    """
    segmented_time_start = []
    segmented_time_end = []
    segmented_position_start = []
    segmented_position_end = []
    segmented_rates = []
    segmented_diffs = []
    
    for start, end in segments:
        segmented_time_start.append(time[start])
        segmented_time_end.append(time[end])
        segmented_position_start.append(position[start])
        segmented_position_end.append(position[end])
        
        # Calculate rate (handle case where time doesn't change)
        rate = 0 if time[end] == time[start] else (position[end] - position[start]) / (time[end] - time[start])
        segmented_rates.append(rate)
    
    # Calculate discontinuities between segments
    for i in range(1, len(segmented_position_end)):
        segmented_diffs.append(segmented_position_start[i] - segmented_position_end[i - 1])
    
    # Add NaN for the last difference to maintain length
    segmented_diffs.append(np.nan)
    
    return (
        segmented_time_start, 
        segmented_time_end, 
        segmented_position_start, 
        segmented_position_end, 
        segmented_rates, 
        segmented_diffs
    )

def segment_distance(self):
    """Segment DNAp-SSB distance data and analyze transition points"""
    # Check if we have trace data
    if getattr(self, 'traces', None) is None or len(self.traces) == 0:
        messagebox.showinfo("Info", "No SSB trajectories detected. Please detect SSB trajectories first.")
        return
    
    # Check if we have DNAp trace
    if getattr(self, 'trace', None) is None:
        messagebox.showinfo("Info", "Please load a trace file for DNAp trajectory first.")
        return
    
    try:
        # Calculate distance if not already done
        distances = self.calculate_dnap_ssb_distance(plot_result=False)
            
        if not distances or len(distances) == 0:
            messagebox.showinfo("Info", "No valid distance measurements could be calculated.")
            return
        
        # Choose which SSB trace to analyze
        if len(distances) > 1:
            # Create a dialog to select which trace to segment
            trace_dialog = tk.Toplevel(self.master)
            trace_dialog.title("Select SSB Trace for Segmentation")
            trace_dialog.geometry("300x200")
            trace_dialog.transient(self.master)
            trace_dialog.grab_set()
            
            ttk.Label(trace_dialog, text="Select SSB trace to segment:").pack(pady=(10, 5))
            
            trace_var = tk.IntVar(value=0)
            for i in range(len(distances)):
                ttk.Radiobutton(
                    trace_dialog, 
                    text=f"SSB Trace {i+1}", 
                    variable=trace_var, 
                    value=i
                ).pack(anchor=tk.W, padx=20)
            
            # Function to segment the selected trace
            def process_trace():
                self.segment_selected_trace(distances[trace_var.get()])
                trace_dialog.destroy()
            
            # Buttons
            button_frame = ttk.Frame(trace_dialog)
            button_frame.pack(pady=10, fill=tk.X)
            
            ttk.Button(button_frame, text="OK", command=process_trace).pack(side=tk.LEFT, padx=10)
            ttk.Button(button_frame, text="Cancel", command=trace_dialog.destroy).pack(side=tk.RIGHT, padx=10)
        else:
            # Just one trace, segment it directly
            self.segment_selected_trace(distances[0])
            
    except Exception as e:
        messagebox.showerror("Error", f"Failed to segment distance data: {e}")
        if hasattr(self, 'status_var'):
            self.status_var.set(f"Error: {e}")

def segment_selected_trace(self, distance_data):
    """
    Segment the selected distance trace
    
    Parameters:
    -----------
    distance_data : dict
        Dictionary containing distance data for one SSB trace
    """
    try:
        # Extract time and distance data
        time = distance_data['ssb_time']
        distance = distance_data['distance']
        
        # Update status
        if hasattr(self, 'status_var'):
            self.status_var.set("Processing segments...")
            
        # Smooth the distance data
        window_size = min(51, len(time) - (len(time) % 2) - 1)  # Ensure odd length
        if window_size >= 5:
            smoothed_distance = savgol_filter(distance, window_size, 3)
        else:
            smoothed_distance = distance
        
        # Calculate the derivative
        derivative = self.first_derivative(time, smoothed_distance, window_size=3)
        
        # Segment the data
        segments = self.segment_data(derivative, threshold=0.01, min_segment_length=5)
        
        # Process the segments
        (
            segmented_time_start, 
            segmented_time_end, 
            segmented_position_start, 
            segmented_position_end, 
            segmented_rates, 
            segmented_diffs
        ) = self.process_segments(time, smoothed_distance, segments)
        
        # Store the segmented data
        self.segmented_data = pd.DataFrame({
            'segment_start_time': segmented_time_start,
            'segment_end_time': segmented_time_end,
            'segment_start_position': segmented_position_start,
            'segment_end_position': segmented_position_end,
            'segment_rate': segmented_rates,
            'segment_discontinuity': segmented_diffs
        })
        
        # Create a new Toplevel window for the segmentation plot
        plot_window = tk.Toplevel(self.master)
        plot_window.title("Segmented DNAp-SSB Distance")
        plot_window.geometry("900x700")
        
        # Create a Figure with subplots
        fig = Figure(figsize=(9, 7))
        ax = fig.add_subplot(111)
        
        # Plot the raw data
        ax.scatter(time, distance, marker='o', s=10, alpha=0.3, label='Raw')
        
        # Plot the smoothed data
        ax.plot(time, smoothed_distance, 'b-', alpha=0.5, linewidth=1, label='Smoothed')
        
        # Plot the segments
        segment_colors = ['r', 'g', 'm', 'c', 'y']  # Colors for segments
        
        for i, (t_start, t_end, p_start, p_end, rate) in enumerate(zip(
            segmented_time_start, segmented_time_end, 
            segmented_position_start, segmented_position_end, segmented_rates
        )):
            # Plot segment with cycling colors
            color = segment_colors[i % len(segment_colors)]
            ax.plot([t_start, t_end], [p_start, p_end], f'{color}-', linewidth=2)
            
            # Add rate annotation
            ax.text(
                (t_start + t_end) / 2, 
                (p_start + p_end) / 2, 
                f'{rate:.2f}', 
                horizontalalignment='center', 
                verticalalignment='bottom',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7)
            )
        
        # Add segment start/end points
        ax.plot(segmented_time_start, segmented_position_start, 'ro', markersize=4)
        ax.plot(segmented_time_end, segmented_position_end, 'go', markersize=4)
        
        # Set labels and title
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance (μm)')
        ax.set_title('Segmented DNAp-SSB Distance')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add text explaining the colored points
        ax.text(
            0.02, 0.02, 
            "Red: Segment Start\nGreen: Segment End\nValues: Rate (μm/s)", 
            transform=ax.transAxes, 
            fontsize=9, 
            bbox=dict(facecolor='white', alpha=0.7)
        )
        
        # Add a horizontal line at y=0 to show when DNAp passes SSB
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Add text annotation explaining the sign of distance
        ax.text(0.02, 0.92, "Negative: DNAp ahead of SSB\nPositive: SSB ahead of DNAp", 
               transform=ax.transAxes, fontsize=9, 
               bbox=dict(facecolor='white', alpha=0.7))
        
        # Adjust layout
        fig.tight_layout()
        
        # Embed the plot into the Toplevel window
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Add Navigation Toolbar
        toolbar = NavigationToolbar2Tk(canvas, plot_window)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        # Add Export button
        button_frame = ttk.Frame(plot_window)
        button_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        ttk.Button(
            button_frame, 
            text="Export Segmented Data", 
            command=lambda: self.export_segments(self.segmented_data)
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="Save Plot", 
            command=lambda: self.save_high_res(fig)
        ).pack(side=tk.LEFT, padx=5)
        
        # Update status
        if hasattr(self, 'status_var'):
            self.status_var.set(f"Segmented data into {len(segments)} segments")
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to segment trace: {e}")
        if hasattr(self, 'status_var'):
            self.status_var.set(f"Error segmenting trace: {e}")

def export_segments(self, segment_data=None):
    """Export the segmented data to a CSV file"""
    try:
        # Use stored segmented data if none provided
        if segment_data is None:
            if not hasattr(self, 'segmented_data') or self.segmented_data is None:
                messagebox.showinfo("Info", "No segmented data available. Please segment the data first.")
                return
            segment_data = self.segmented_data
            
        # Ask for file name
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Segmented Data"
        )
        
        if file_path:
            # Save data to CSV
            segment_data.to_csv(file_path, index=False)
            messagebox.showinfo("Success", f"Segmented data saved to {os.path.basename(file_path)}")
            
            # Update status
            if hasattr(self, 'status_var'):
                self.status_var.set(f"Saved segmented data to {os.path.basename(file_path)}")
                
    except Exception as e:
        messagebox.showerror("Error", f"Failed to export segmented data: {e}")
        if hasattr(self, 'status_var'):
            self.status_var.set(f"Error exporting data: {e}")

def calculate_dnap_ssb_distance(self, plot_result=True):
    """
    Calculate the distance between DNAp and SSB trajectories.
    
    Parameters:
    -----------
    plot_result : bool
        Whether to plot the distance results
        
    Returns:
    --------
    list of dict
        List of dictionaries containing distance data for each SSB trace
    """
    if self.trace is None:
        messagebox.showinfo("Info", "Please load a trace file first.")
        return None
    
    if not hasattr(self, 'traces') or self.traces is None or len(self.traces) == 0:
        messagebox.showinfo("Info", "No SSB trajectories detected.")
        return None
    
    try:
        # Update status
        if hasattr(self, 'status_var'):
            self.status_var.set("Calculating DNAp-SSB distances...")
        
        # Get DNAp trace data
        dnap_time = self.trace['time'] if 'time' in self.trace.columns else self.trace['Time']
        dnap_pos = self.trace['junction_position_all'] if 'junction_position_all' in self.trace.columns else self.trace['Position']
        
        # Apply Savitzky-Golay filter to DNAp data
        window_length = min(21, len(dnap_time) - (len(dnap_time) % 2) - 1)  # Ensure odd length
        if window_length < 3:
            window_length = 3  # Minimum window length for Savitzky-Golay
            
        dnap_time_smooth = dnap_time
        dnap_pos_smooth = savgol_filter(dnap_pos, window_length, 3) if len(dnap_time) > window_length else dnap_pos
        
        # Create interpolation function for smooth DNAp position
        interp_DNAp = interp1d(
            dnap_time_smooth, dnap_pos_smooth, 
            bounds_error=False, fill_value="extrapolate"
        )
        
        distances = []
        
        # For each SSB trace, calculate the distance to DNAp
        for i, trace in enumerate(self.smoothed_traces):
            ssb_time = trace['Time'].values
            ssb_pos = trace['Position'].values
            
            # Get DNAp position at SSB timepoints
            dnap_pos_at_ssb_time = interp_DNAp(ssb_time)
            
            # Calculate distance (negative means SSB is ahead of DNAp)
            position_diff = -(dnap_pos_at_ssb_time - ssb_pos)
            
            # Add to distances list
            distances.append({
                'trace_index': i,
                'ssb_time': ssb_time,
                'ssb_pos': ssb_pos, 
                'dnap_pos': dnap_pos_at_ssb_time,
                'distance': position_diff
            })
        
        if not distances:
            messagebox.showinfo("Info", "No valid distance measurements could be calculated.")
            return None
            
        # Store the distance data
        self.distance_data = distances
            
        if plot_result:
            # Create a new window for the distance plot
            self.plot_distance_data(distances)
            
        # Update status
        if hasattr(self, 'status_var'):
            self.status_var.set(f"Calculated distances for {len(distances)} SSB trajectories")
            
        return distances
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to calculate DNAp-SSB distance: {e}")
        if hasattr(self, 'status_var'):
            self.status_var.set(f"Error calculating distance: {e}")
        return None

def plot_distance_data(self, distances):
    """
    Plot the distance data between DNAp and SSB
    
    Parameters:
    -----------
    distances : list of dict
        List of dictionaries containing distance data for each SSB trace
    """
    try:
        # Create a new Toplevel window for the plot
        plot_window = tk.Toplevel(self.master)
        plot_window.title("DNAp-SSB Distance Analysis")
        plot_window.geometry("800x600")
        
        # Create a Figure
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        # Plot the distance for each trace
        for i, distance_data in enumerate(distances):
            time = distance_data['ssb_time']
            distance = distance_data['distance']
            
            # Plot both raw data and smoothed curve
            ax.scatter(time, distance, marker='o', s=10, alpha=0.5, label=f"Raw {i+1}")
            
            # Smooth the distance data if there are enough points
            if len(time) >= 5:
                # Use Savitzky-Golay filter with appropriate window size
                window_size = min(71, len(time) - (len(time) % 2) - 1)  # Ensure odd length
                if window_size < 5:
                    window_size = 5  # Minimum window size
                    
                smoothed_time = time
                smoothed_distance = savgol_filter(distance, window_size, 3)
                ax.plot(smoothed_time, smoothed_distance, '-', linewidth=1.5, label=f"Smooth {i+1}")
            else:
                # Just connect the dots for very short traces
                ax.plot(time, distance, '-', linewidth=1.5, label=f"Trace {i+1}")
        
        # Set labels and title
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance Between DNAp and SSB (μm)')
        ax.set_title('DNAp-SSB Distance Analysis')
        ax.grid(True, alpha=0.3)
        
        # Add a legend
        if len(distances) <= 5:  # Only show legend for a reasonable number of traces
            ax.legend(loc='best')
        
        # Add a horizontal line at y=0 to show when DNAp passes SSB
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Add text annotation explaining the sign of distance
        ax.text(0.02, 0.02, "Negative: DNAp ahead of SSB\nPositive: SSB ahead of DNAp", 
               transform=ax.transAxes, fontsize=8, 
               bbox=dict(facecolor='white', alpha=0.7))
        
        # Adjust layout
        fig.tight_layout()
        
        # Embed the plot into the Toplevel window
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Add Navigation Toolbar
        toolbar = NavigationToolbar2Tk(canvas, plot_window)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        # Add buttons
        button_frame = ttk.Frame(plot_window)
        button_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        # Add Export button
        export_button = ttk.Button(
            button_frame, 
            text="Export Distance Data", 
            command=lambda: self.export_distance_data()
        )
        export_button.pack(side=tk.LEFT, padx=5)
        
        # Add Segment button
        segment_button = ttk.Button(
            button_frame, 
            text="Segment Distance", 
            command=self.segment_distance
        )
        segment_button.pack(side=tk.LEFT, padx=5)
        
        # Add Save Plot button
        save_button = ttk.Button(
            button_frame, 
            text="Save Plot", 
            command=lambda: self.save_high_res(fig)
        )
        save_button.pack(side=tk.LEFT, padx=5)
    
    except Exception as e:
        messagebox.showerror("Error", f"Failed to plot distance data: {e}")
        if hasattr(self, 'status_var'):
            self.status_var.set(f"Error plotting distance: {e}")

def export_distance_data(self):
    """Export the distance data to a CSV file"""
    try:
        if not hasattr(self, 'distance_data') or self.distance_data is None:
            messagebox.showinfo("Info", "No distance data available. Please calculate distances first.")
            return
            
        # Ask for file name
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Distance Data"
        )
        
        if not file_path:
            return
        
        # Create a DataFrame for each trace and export
        for i, distance_data in enumerate(self.distance_data):
            trace_file = file_path.replace('.csv', f'_trace{i+1}.csv')
            
            df = pd.DataFrame({
                'Time': distance_data['ssb_time'],
                'SSB_Position': distance_data['ssb_pos'],
                'DNAp_Position': distance_data['dnap_pos'],
                'Distance': distance_data['distance']
            })
            
            df.to_csv(trace_file, index=False)
        
        messagebox.showinfo("Success", f"Distance data saved to {os.path.dirname(file_path)}")
        
        # Update status
        if hasattr(self, 'status_var'):
            self.status_var.set(f"Saved distance data to {os.path.basename(file_path)}")
            
    except Exception as e:
        messagebox.showerror("Error", f"Failed to export distance data: {e}")
        if hasattr(self, 'status_var'):
            self.status_var.set(f"Error exporting distance data: {e}") 