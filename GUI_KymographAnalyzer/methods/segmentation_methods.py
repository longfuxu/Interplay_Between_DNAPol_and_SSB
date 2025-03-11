"""
Segmentation methods for the Correlation Image Force Analyzer.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import tkinter as tk
from tkinter import messagebox, filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

def first_derivative(self, x, y, window_size):
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
    
    # Calculate the derivative
    derivative = np.zeros_like(y)
    
    # Use central difference for interior points
    for i in range(window_size, len(y) - window_size):
        # Use points within the window for better stability
        x_window = x[i-window_size:i+window_size+1]
        y_window = y[i-window_size:i+window_size+1]
        
        # Linear fit to get the slope
        coeffs = np.polyfit(x_window, y_window, 1)
        derivative[i] = coeffs[0]
    
    # Forward difference for the first few points
    for i in range(window_size):
        x_window = x[:2*window_size+1]
        y_window = y[:2*window_size+1]
        coeffs = np.polyfit(x_window, y_window, 1)
        derivative[i] = coeffs[0]
    
    # Backward difference for the last few points
    for i in range(len(y) - window_size, len(y)):
        x_window = x[-2*window_size-1:]
        y_window = y[-2*window_size-1:]
        coeffs = np.polyfit(x_window, y_window, 1)
        derivative[i] = coeffs[0]
    
    return derivative

def segment_data(self, derivative):
    """
    Segment the data based on the derivative
    
    Parameters:
    -----------
    derivative : array-like
        First derivative values
        
    Returns:
    --------
    list
        List of segment indices (start, end)
    """
    # Define the threshold for segmentation
    threshold = 0.05  # Adjust as needed
    
    # Find where the derivative crosses the threshold
    crossings = []
    for i in range(1, len(derivative)):
        if (derivative[i-1] < threshold and derivative[i] >= threshold) or \
           (derivative[i-1] >= threshold and derivative[i] < threshold):
            crossings.append(i)
    
    # Ensure we have an even number of crossings
    if len(crossings) % 2 != 0:
        crossings.append(len(derivative) - 1)
    
    # Group crossings into segments
    segments = []
    for i in range(0, len(crossings), 2):
        if i + 1 < len(crossings):
            segments.append((crossings[i], crossings[i+1]))
    
    # Add the last segment if needed
    if len(segments) == 0 or segments[-1][1] < len(derivative) - 1:
        segments.append((segments[-1][1] if segments else 0, len(derivative) - 1))
    
    return segments

def process_segments(self, time, position, segments):
    """
    Process the segments to extract features
    
    Parameters:
    -----------
    time : array-like
        Time values
    position : array-like
        Position values
    segments : list
        List of segment indices (start, end)
        
    Returns:
    --------
    dict
        Dictionary of segment features
    """
    segment_features = {
        'start_time': [],
        'end_time': [],
        'duration': [],
        'start_position': [],
        'end_position': [],
        'displacement': [],
        'velocity': []
    }
    
    for start, end in segments:
        # Extract segment data
        segment_time = time[start:end+1]
        segment_position = position[start:end+1]
        
        # Calculate features
        start_time = segment_time[0]
        end_time = segment_time[-1]
        duration = end_time - start_time
        start_position = segment_position[0]
        end_position = segment_position[-1]
        displacement = end_position - start_position
        
        # Calculate velocity (avoid division by zero)
        velocity = displacement / duration if duration > 0 else 0
        
        # Store features
        segment_features['start_time'].append(start_time)
        segment_features['end_time'].append(end_time)
        segment_features['duration'].append(duration)
        segment_features['start_position'].append(start_position)
        segment_features['end_position'].append(end_position)
        segment_features['displacement'].append(displacement)
        segment_features['velocity'].append(velocity)
    
    return segment_features

def segment_distance(self):
    """Segment the distance data and analyze it"""
    if self.distance_data is None:
        messagebox.showinfo("Info", "Please calculate DNAp-SSB distance first.")
        return
    
    try:
        # Extract time and distance data
        time = self.distance_data['Time'].values
        distance = self.distance_data['Distance'].values
        
        # Smooth the distance data
        window_size = 5  # Adjust as needed
        distance_smooth = savgol_filter(distance, window_size*2+1, 2)
        
        # Calculate the derivative
        derivative = self.first_derivative(time, distance_smooth, window_size)
        
        # Segment the data
        segments = self.segment_data(derivative)
        
        # Process the segments
        segment_features = self.process_segments(time, distance_smooth, segments)
        
        # Store the segmented data
        self.segmented_data = pd.DataFrame(segment_features)
        
        # Create a new Toplevel window for the plot
        plot_window = tk.Toplevel(self.master)
        plot_window.title("Segmented Distance Analysis")
        plot_window.geometry("1000x800")
        
        # Create a Figure with subplots
        fig = Figure(figsize=(10, 8))
        
        # Plot the distance and segments
        ax1 = fig.add_subplot(211)
        ax1.plot(time, distance, 'b-', alpha=0.5, label='Raw Distance')
        ax1.plot(time, distance_smooth, 'r-', label='Smoothed Distance')
        
        # Highlight segments
        for i, (start, end) in enumerate(segments):
            ax1.axvspan(time[start], time[end], alpha=0.2, color=f'C{i%10}')
            
            # Add segment number
            mid_time = (time[start] + time[end]) / 2
            mid_distance = np.mean(distance_smooth[start:end+1])
            ax1.text(mid_time, mid_distance, str(i+1), 
                     horizontalalignment='center', verticalalignment='center',
                     bbox=dict(facecolor='white', alpha=0.7))
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Distance (µm)')
        ax1.set_title('Distance vs Time with Segments')
        ax1.legend()
        ax1.grid(True)
        
        # Plot the derivative
        ax2 = fig.add_subplot(212, sharex=ax1)
        ax2.plot(time, derivative, 'g-', label='Derivative')
        ax2.axhline(y=0.05, color='r', linestyle='--', label='Threshold')
        
        # Highlight segments
        for i, (start, end) in enumerate(segments):
            ax2.axvspan(time[start], time[end], alpha=0.2, color=f'C{i%10}')
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Derivative (µm/s)')
        ax2.set_title('Derivative vs Time with Segments')
        ax2.legend()
        ax2.grid(True)
        
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
        
        # Add Export Button
        export_button = tk.Button(plot_window, text="Export Segments", command=self.export_segments)
        export_button.pack(side=tk.TOP, pady=5)
        
        # Add Save High-Res Figure Button
        save_button = tk.Button(plot_window, text="Save High-Res Figure", 
                               command=lambda: self.save_high_res(fig))
        save_button.pack(side=tk.TOP, pady=5)
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to segment distance data: {e}")

def export_segments(self):
    """Export the segmented data to a CSV file"""
    if self.segmented_data is None:
        messagebox.showinfo("Info", "No segmented data to export.")
        return
    
    # Ask for the output file
    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    if not file_path:
        return
    
    try:
        # Save to CSV
        self.segmented_data.to_csv(file_path, index=False)
        
        messagebox.showinfo("Success", f"Segmented data exported to {file_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to export segmented data: {e}")

def calculate_dnap_ssb_distance(self):
    """Calculate the distance between DNAp and SSB"""
    if self.dnap_trace is None or self.smoothed_traces is None:
        messagebox.showinfo("Info", "Please detect SSB trajectories first.")
        return
    
    try:
        # Create a new Toplevel window for the plot
        plot_window = tk.Toplevel(self.master)
        plot_window.title("DNAp-SSB Distance Analysis")
        plot_window.geometry("1000x800")
        
        # Create a Figure
        fig = Figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        # Plot DNAp trajectory
        ax.plot(self.dnap_trace['Time'], self.dnap_trace['Position'], 'r-', label='DNAp')
        
        # Plot SSB trajectories
        for i, trace in enumerate(self.smoothed_traces):
            ax.plot(trace['Time'], trace['Position'], '-', label=f'SSB {i+1}')
        
        # Calculate distances
        all_times = []
        all_distances = []
        
        # For each SSB trajectory
        for i, ssb_trace in enumerate(self.smoothed_traces):
            # Interpolate DNAp position to match SSB time points
            dnap_interp = np.interp(ssb_trace['Time'], self.dnap_trace['Time'], self.dnap_trace['Position'])
            
            # Calculate distance
            distance = np.abs(ssb_trace['Position'] - dnap_interp)
            
            # Store data
            all_times.extend(ssb_trace['Time'])
            all_distances.extend(distance)
            
            # Plot distance
            ax2 = ax.twinx()
            ax2.plot(ssb_trace['Time'], distance, '--', alpha=0.7, label=f'Distance {i+1}')
            ax2.set_ylabel('Distance (µm)')
            
        # Sort by time
        idx = np.argsort(all_times)
        all_times = np.array(all_times)[idx]
        all_distances = np.array(all_distances)[idx]
        
        # Store the distance data
        self.distance_data = pd.DataFrame({
            'Time': all_times,
            'Distance': all_distances
        })
        
        # Set labels and title
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (µm)')
        ax.set_title('DNAp and SSB Trajectories with Distance')
        ax.legend(loc='upper left')
        if hasattr(ax2, 'legend'):
            ax2.legend(loc='upper right')
        
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
        button_frame = tk.Frame(plot_window)
        button_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        # Add Export Button
        export_button = tk.Button(button_frame, text="Export Distance Data", 
                                 command=lambda: self.export_distance_data(all_times, all_distances))
        export_button.pack(side=tk.LEFT, padx=5)
        
        # Add Segment Button
        segment_button = tk.Button(button_frame, text="Segment Distance", command=self.segment_distance)
        segment_button.pack(side=tk.LEFT, padx=5)
        
        # Add Save High-Res Figure Button
        save_button = tk.Button(button_frame, text="Save High-Res Figure", 
                               command=lambda: self.save_high_res(fig))
        save_button.pack(side=tk.LEFT, padx=5)
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to calculate DNAp-SSB distance: {e}") 