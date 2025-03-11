"""
Analysis methods for the Correlation Image Force Analyzer.
"""
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.signal import savgol_filter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import cv2
import lumicks.pylake as lk

def plot_time_vs_eed_and_fork_front_popout(self):
    """Plot time vs end-to-end distance and junction front position in a pop-out window"""
    if self.trace is None:
        messagebox.showinfo("Info", "Please load a trace file first.")
        return
    
    # Create a new Toplevel window for the plot
    plot_window = tk.Toplevel(self.master)
    plot_window.title("Time vs End-to-End Distance and Junction Front Position")
    plot_window.geometry("800x600")
    
    # Create a new Figure instance
    fig = Figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    
    # Plot time vs end-to-end distance
    x_data = self.trace['time'] if 'time' in self.trace.columns else self.trace['Time']
    y_data = self.trace['junction_position_all'] if 'junction_position_all' in self.trace.columns else self.trace['Position']
    
    ax1.plot(x_data, y_data, 'b-', label='Junction Position')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (µm)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    # Create a second y-axis for velocity
    ax2 = ax1.twinx()
    
    # Calculate velocity
    velocity = np.gradient(y_data, x_data)
    # Smooth the velocity
    velocity_smooth = savgol_filter(velocity, 21, 3)
    
    # Plot time vs velocity
    ax2.plot(x_data, velocity_smooth, 'r-', label='Velocity')
    ax2.set_ylabel('Velocity (µm/s)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
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
    
    # Add Save High-Res Figure Button
    save_button = ttk.Button(plot_window, text="Save High-Res Figure", command=lambda: self.save_high_res(fig))
    save_button.pack(side=tk.TOP, pady=5)

def plot_time_vs_fork_front_reverse_popout(self):
    """Plot time vs junction front position in reverse direction in a pop-out window"""
    if self.trace is None:
        messagebox.showinfo("Info", "Please load a trace file first.")
        return
    
    # Create a new Toplevel window for the plot
    plot_window = tk.Toplevel(self.master)
    plot_window.title("Time vs Junction Front Position (Reverse)")
    plot_window.geometry("800x600")
    
    # Create a new Figure instance
    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    # Calculate junction front position in reverse direction
    x_data = self.trace['time'] if 'time' in self.trace.columns else self.trace['Time']
    y_data = self.trace['junction_position_all'] if 'junction_position_all' in self.trace.columns else self.trace['Position']
    pos_max = y_data.max()
    pos_reverse = pos_max - y_data
    
    # Plot time vs junction front position in reverse direction
    ax.plot(x_data, pos_reverse, 'g-', label='Junction Front Position (Reverse)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (µm)')
    ax.set_title('Time vs Junction Front Position (Reverse)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
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
    
    # Add Save High-Res Figure Button
    save_button = ttk.Button(plot_window, text="Save High-Res Figure", command=lambda: self.save_high_res(fig))
    save_button.pack(side=tk.TOP, pady=5)

def force_image_analyzer(self):
    """Analyze junction position and image data together"""
    if self.trace is None or self.img is None:
        messagebox.showinfo("Info", "Please load both kymograph and trace files first.")
        return
    
    try:
        # Extract the selected channel
        channel_idx = {'Red': 0, 'Green': 1, 'Blue': 2}[self.selected_channel]
        img_channel = self.img[:, :, channel_idx]
        
        # Apply ROI if specified
        roi_applied = False
        try:
            x_left = int(self.roi_x_left_entry.get())
            x_right = int(self.roi_x_right_entry.get())
            y_top = int(self.roi_y_top_entry.get())
            y_bottom = int(self.roi_y_bottom_entry.get())
            
            if all([x_left, x_right, y_top, y_bottom]):
                roi_img = img_channel[y_top:y_bottom, x_left:x_right]
                if roi_img.size > 0:  # Check if ROI is valid
                    img_channel = roi_img
                    roi_applied = True
        except ValueError:
            pass  # Use the whole image if ROI inputs are invalid
        
        # Create a new Toplevel window for the plot
        plot_window = tk.Toplevel(self.master)
        plot_window.title("Junction-Image Analysis")
        plot_window.geometry("1000x800")
        
        # Create a Figure with subplots
        fig = Figure(figsize=(10, 8))
        
        # Plot the kymograph
        ax1 = fig.add_subplot(211)
        
        # Adjust contrast for better visibility
        p2, p98 = np.percentile(img_channel, (2, self.contrast_max))
        img_rescale = np.clip(img_channel, p2, p98)
        
        im = ax1.imshow(img_rescale, cmap='gray', aspect='auto')
        ax1.set_title(f"Kymograph - {self.selected_channel} Channel")
        
        if roi_applied:
            ax1.set_xlabel("Time (pixels) - ROI")
            ax1.set_ylabel("Position (pixels) - ROI")
        else:
            ax1.set_xlabel("Time (pixels)")
            ax1.set_ylabel("Position (pixels)")
        
        # Add colorbar
        fig.colorbar(im, ax=ax1, label='Intensity')
        
        # Plot the trace
        ax2 = fig.add_subplot(212)
        
        # Get the data from the trace file
        x_data = self.trace['time'] if 'time' in self.trace.columns else self.trace['Time']
        y_data = self.trace['junction_position_all'] if 'junction_position_all' in self.trace.columns else self.trace['Position']
        
        ax2.plot(x_data, y_data, 'b-')
        ax2.set_title("Junction Position Trace")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Position (µm)")
        ax2.grid(True, alpha=0.3)
        
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
        
        # Add Save High-Res Figure Button
        save_button = ttk.Button(plot_window, text="Save High-Res Figure", command=lambda: self.save_high_res(fig))
        save_button.pack(side=tk.TOP, pady=5)
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to analyze force and image data: {e}")

def force_image_analyzer_reverse(self):
    """Analyze junction position and image data together in reverse direction"""
    if self.trace is None or self.img is None:
        messagebox.showinfo("Info", "Please load both kymograph and trace files first.")
        return
    
    try:
        # Extract the selected channel
        channel_idx = {'Red': 0, 'Green': 1, 'Blue': 2}[self.selected_channel]
        img_channel = self.img[:, :, channel_idx]
        
        # Apply ROI if specified
        roi_applied = False
        try:
            x_left = int(self.roi_x_left_entry.get())
            x_right = int(self.roi_x_right_entry.get())
            y_top = int(self.roi_y_top_entry.get())
            y_bottom = int(self.roi_y_bottom_entry.get())
            
            if all([x_left, x_right, y_top, y_bottom]):
                roi_img = img_channel[y_top:y_bottom, x_left:x_right]
                if roi_img.size > 0:  # Check if ROI is valid
                    img_channel = roi_img
                    roi_applied = True
        except ValueError:
            pass  # Use the whole image if ROI inputs are invalid
        
        # Reverse the image (flip horizontally)
        img_channel_reverse = np.fliplr(img_channel)
        
        # Create a new Toplevel window for the plot
        plot_window = tk.Toplevel(self.master)
        plot_window.title("Junction-Image Analysis (Reverse)")
        plot_window.geometry("1000x800")
        
        # Create a Figure with subplots
        fig = Figure(figsize=(10, 8))
        
        # Plot the reversed kymograph
        ax1 = fig.add_subplot(211)
        
        # Adjust contrast for better visibility
        p2, p98 = np.percentile(img_channel_reverse, (2, self.contrast_max))
        img_rescale = np.clip(img_channel_reverse, p2, p98)
        
        im = ax1.imshow(img_rescale, cmap='gray', aspect='auto')
        ax1.set_title(f"Kymograph (Reverse) - {self.selected_channel} Channel")
        
        if roi_applied:
            ax1.set_xlabel("Time (pixels) - ROI")
            ax1.set_ylabel("Position (pixels) - ROI")
        else:
            ax1.set_xlabel("Time (pixels)")
            ax1.set_ylabel("Position (pixels)")
        
        # Add colorbar
        fig.colorbar(im, ax=ax1, label='Intensity')
        
        # Plot the trace in reverse direction
        ax2 = fig.add_subplot(212)
        
        # Get the data from the trace file and reverse it
        x_data = self.trace['time'] if 'time' in self.trace.columns else self.trace['Time']
        y_data = self.trace['junction_position_all'] if 'junction_position_all' in self.trace.columns else self.trace['Position']
        position_reverse = y_data.max() - y_data
        
        ax2.plot(x_data, position_reverse, 'b-')
        ax2.set_title("Junction Position Trace (Reverse)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Position (µm)")
        ax2.grid(True, alpha=0.3)
        
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
        
        # Add Save High-Res Figure Button
        save_button = ttk.Button(plot_window, text="Save High-Res Figure", command=lambda: self.save_high_res(fig))
        save_button.pack(side=tk.TOP, pady=5)
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to analyze force and image data in reverse: {e}")

def display_intensity_with_unwinding(self):
    """Display intensity profile with unwinding data"""
    if self.trace is None or self.img is None:
        messagebox.showinfo("Info", "Please load both kymograph and trace files first.")
        return
    
    try:
        # Extract the selected channel
        channel_idx = {'Red': 0, 'Green': 1, 'Blue': 2}[self.selected_channel]
        img_channel = self.img[:, :, channel_idx]
        
        # Get the data from the trace file
        x_data = self.trace['time'] if 'time' in self.trace.columns else self.trace['Time']
        y_data = self.trace['junction_position_all'] if 'junction_position_all' in self.trace.columns else self.trace['Position']
        
        # Calculate the image intensity along the trace
        intensity_profile = []
        
        # Use time_per_line to convert from time to pixel coordinates
        # and px_size to convert from position to pixel coordinates
        time_px = (x_data * 1000 / self.time_per_line).astype(int)  # Convert time to pixel index
        pos_px = (y_data / self.px_size).astype(int)  # Convert position to pixel index
        
        # Clip indices to image dimensions
        time_px = np.clip(time_px, 0, img_channel.shape[1] - 1)
        pos_px = np.clip(pos_px, 0, img_channel.shape[0] - 1)
        
        # Extract intensity values
        for t, p in zip(time_px, pos_px):
            if 0 <= t < img_channel.shape[1] and 0 <= p < img_channel.shape[0]:
                intensity_profile.append(img_channel[p, t])
            else:
                intensity_profile.append(0)
        
        # Create a new Toplevel window for the plot
        plot_window = tk.Toplevel(self.master)
        plot_window.title("Intensity Profile Along Junction Path")
        plot_window.geometry("1000x800")
        
        # Create a Figure with subplots
        fig = Figure(figsize=(10, 8))
        
        # Plot the kymograph with the trace overlaid
        ax1 = fig.add_subplot(311)
        im = ax1.imshow(img_channel, cmap='gray', aspect='auto')
        ax1.plot(time_px, pos_px, 'r-', linewidth=1)
        ax1.set_title(f"Kymograph with Junction Path - {self.selected_channel} Channel")
        ax1.set_xlabel("Time (pixels)")
        ax1.set_ylabel("Position (pixels)")
        fig.colorbar(im, ax=ax1, label='Intensity')
        
        # Plot the position vs time
        ax2 = fig.add_subplot(312)
        ax2.plot(x_data, y_data, 'b-')
        ax2.set_title("Junction Position vs Time")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Position (µm)")
        ax2.grid(True, alpha=0.3)
        
        # Plot the intensity profile along the trace
        ax3 = fig.add_subplot(313)
        ax3.plot(x_data, intensity_profile, 'g-')
        ax3.set_title("Intensity Profile Along Junction Path")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Intensity")
        ax3.grid(True, alpha=0.3)
        
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
        
        # Add Save High-Res Figure Button
        save_button = ttk.Button(plot_window, text="Save High-Res Figure", command=lambda: self.save_high_res(fig))
        save_button.pack(side=tk.TOP, pady=5)
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to display intensity with unwinding: {e}")

def detect_ssb_trajectories(self):
    """Detect SSB trajectories from the kymograph"""
    if self.img is None:
        messagebox.showinfo("Info", "Please load a kymograph file first.")
        return
    
    try:
        # Check if ROI is specified
        roi_specified = False
        try:
            x_left = int(self.roi_x_left_entry.get()) if self.roi_x_left_entry.get() else None
            x_right = int(self.roi_x_right_entry.get()) if self.roi_x_right_entry.get() else None
            y_top = int(self.roi_y_top_entry.get()) if self.roi_y_top_entry.get() else None
            y_bottom = int(self.roi_y_bottom_entry.get()) if self.roi_y_bottom_entry.get() else None
            
            if all(v is not None for v in [x_left, x_right, y_top, y_bottom]):
                roi_specified = True
                self.roi_coords = {'start_x': x_left, 'end_x': x_right, 
                                  'start_y': y_top, 'end_y': y_bottom}
        except (ValueError, TypeError):
            pass
        
        if not roi_specified:
            messagebox.showinfo("Info", "Please specify a ROI for SSB trajectory detection.")
            return
        
        # Extract the blue channel (for SSB detection)
        b_channel = self.img[:, :, 2]
        
        # Extract the ROI for trajectory detection
        if roi_specified:
            b_ROI = b_channel[y_top:y_bottom, x_left:x_right]
        else:
            b_ROI = b_channel
        
        # Check if ROI is valid
        if b_ROI.size == 0 or b_ROI.shape[0] <= 1 or b_ROI.shape[1] <= 1:
            messagebox.showerror("Error", "Invalid ROI size. Please select a larger region.")
            return
        
        # Use pylake to detect trajectories
        try:
            # Parameters for trajectory detection
            line_width = 5
            pixel_threshold = 25
            window = 6
            
            # Create parameter entry dialog
            param_dialog = tk.Toplevel(self.master)
            param_dialog.title("Trajectory Detection Parameters")
            param_dialog.geometry("400x200")
            param_dialog.transient(self.master)
            param_dialog.grab_set()
            
            # Create parameter entry fields
            ttk.Label(param_dialog, text="Line Width:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
            line_width_var = tk.IntVar(value=line_width)
            ttk.Entry(param_dialog, textvariable=line_width_var, width=10).grid(row=0, column=1, padx=5, pady=5)
            
            ttk.Label(param_dialog, text="Pixel Threshold:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
            pixel_threshold_var = tk.IntVar(value=pixel_threshold)
            ttk.Entry(param_dialog, textvariable=pixel_threshold_var, width=10).grid(row=1, column=1, padx=5, pady=5)
            
            ttk.Label(param_dialog, text="Window Size:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
            window_var = tk.IntVar(value=window)
            ttk.Entry(param_dialog, textvariable=window_var, width=10).grid(row=2, column=1, padx=5, pady=5)
            
            # Function to process parameters and detect trajectories
            def process_parameters():
                nonlocal line_width, pixel_threshold, window
                try:
                    line_width = line_width_var.get()
                    pixel_threshold = pixel_threshold_var.get()
                    window = window_var.get()
                    param_dialog.destroy()
                    
                    # Status update
                    self.status_label.configure(text="Detecting SSB trajectories...")
                    self.master.update_idletasks()  # Update the UI
                    
                    # Detect trajectories
                    traces = lk.track_greedy(b_ROI, line_width=line_width, pixel_threshold=pixel_threshold, window=window)
                    
                    # Check if any trajectories were detected
                    if not traces:
                        messagebox.showinfo("Info", "No trajectories detected. Try adjusting parameters.")
                        self.status_label.configure(text="No trajectories detected.")
                        return
                    
                    # Filter very short trajectories
                    min_length = 3  # Minimum number of points
                    traces = [trace for trace in traces if len(trace.time_idx) >= min_length]
                    
                    # Apply additional filtering
                    try:
                        traces = lk.filter_lines(traces, min_length)
                    except Exception as e:
                        print(f"Warning in filter_lines: {e}")
                        # If filter_lines fails, continue with unfiltered traces
                    
                    # Check if any trajectories remain after filtering
                    if not traces:
                        messagebox.showinfo("Info", "No trajectories detected after filtering. Try adjusting parameters.")
                        self.status_label.configure(text="No trajectories detected after filtering.")
                        return
                    
                    # Refine lines with centroid
                    try:
                        traces = lk.refine_lines_centroid(traces, line_width=line_width)
                    except Exception as e:
                        print(f"Warning in refine_lines_centroid: {e}")
                        # If refine_lines_centroid fails, continue with unrefined traces
                    
                    # Convert traces to DataFrames
                    self.traces = []
                    for trace in traces:
                        # Safely extract data from trace
                        if not hasattr(trace, 'time_idx') or not hasattr(trace, 'coordinate_idx'):
                            print(f"Warning: Invalid trace object: {trace}")
                            continue
                            
                        time_idx = np.array(trace.time_idx)
                        coordinate_idx = np.array(trace.coordinate_idx)
                        
                        # Skip empty traces
                        if len(time_idx) == 0 or len(coordinate_idx) == 0:
                            continue
                        
                        # Convert pixel coordinates to physical units
                        time_s = (time_idx + x_left) * self.time_per_line / 1000
                        position_um = (coordinate_idx + y_top) * self.px_size
                        
                        # Create DataFrame
                        df = pd.DataFrame({
                            'Time': time_s,
                            'Position': position_um,
                            'time_idx': time_idx,
                            'coordinate_idx': coordinate_idx
                        })
                        self.traces.append(df)
                    
                    # Make sure we have at least one valid trace
                    if not self.traces:
                        messagebox.showinfo("Info", "No valid trajectories could be extracted. Try adjusting parameters.")
                        self.status_label.configure(text="No valid trajectories extracted.")
                        return
                    
                    # Smooth the trajectories
                    self.smoothed_traces = []
                    for trace in self.traces:
                        # Check if trace has enough points for smoothing
                        if len(trace) < 3:
                            # For very short traces, just use the original data
                            df_smooth = trace.copy()
                            self.smoothed_traces.append(df_smooth)
                            continue
                            
                        # Apply Savitzky-Golay filter for longer traces
                        try:
                            window_size = min(31, len(trace) - 1)
                            # Ensure window size is odd
                            if window_size % 2 == 0:
                                window_size -= 1
                            # Ensure window size is at least 3
                            if window_size >= 3:
                                position_smooth = savgol_filter(trace['Position'], window_size, 2)
                            else:
                                position_smooth = trace['Position'].values
                        except Exception as e:
                            print(f"Warning in smoothing: {e}")
                            position_smooth = trace['Position'].values
                        
                        # Create a new DataFrame with smoothed position
                        df_smooth = pd.DataFrame({
                            'Time': trace['Time'],
                            'Position': position_smooth,
                            'time_idx': trace['time_idx'],
                            'coordinate_idx': trace['coordinate_idx']
                        })
                        
                        self.smoothed_traces.append(df_smooth)
                    
                    # Set the DNAp trace from the trace file
                    if self.trace is not None:
                        try:
                            x_data = self.trace['time'] if 'time' in self.trace.columns else self.trace['Time']
                            y_data = self.trace['junction_position_all'] if 'junction_position_all' in self.trace.columns else self.trace['Position']
                            
                            # Convert to same format as SSB traces
                            self.trace_time_s_filter = x_data
                            self.position_um_filter = y_data
                            
                            # Create a DataFrame for the DNAp trace
                            self.dnap_trace = pd.DataFrame({
                                'Time': x_data,
                                'Position': y_data
                            })
                        except Exception as e:
                            print(f"Warning setting DNAp trace: {e}")
                            # Continue even if DNAp trace can't be set
                    
                    messagebox.showinfo("Success", f"Detected {len(self.traces)} SSB trajectories.")
                    self.status_label.configure(text=f"Detected {len(self.traces)} SSB trajectories.")
                    
                    # Plot the trajectories
                    self.plot_ssb_trajectories()
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to detect trajectories: {str(e)}")
                    self.status_label.configure(text="Failed to detect trajectories.")
                    print(f"Trajectory detection error: {str(e)}")
            
            # Add OK and Cancel buttons
            button_frame = ttk.Frame(param_dialog)
            button_frame.grid(row=3, column=0, columnspan=2, pady=10)
            
            ttk.Button(button_frame, text="OK", command=process_parameters).pack(side=tk.LEFT, padx=10)
            ttk.Button(button_frame, text="Cancel", command=param_dialog.destroy).pack(side=tk.LEFT, padx=10)
            
            # Wait for the dialog to close
            self.master.wait_window(param_dialog)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to detect SSB trajectories: {str(e)}")
            self.status_label.configure(text="Failed to detect trajectories.")
            print(f"SSB trajectory detection error: {str(e)}")
    
    except Exception as e:
        messagebox.showerror("Error", f"Failed to detect SSB trajectories: {str(e)}")
        self.status_label.configure(text="Failed to detect trajectories.")
        print(f"SSB trajectory detection outer error: {str(e)}")

def plot_ssb_trajectories(self):
    """Plot the detected SSB trajectories"""
    if self.traces is None or len(self.traces) == 0:
        messagebox.showinfo("Info", "No SSB trajectories detected.")
        return
    
    try:
        # Create a new Toplevel window for the plot
        plot_window = tk.Toplevel(self.master)
        plot_window.title("SSB Trajectories")
        plot_window.geometry("1000x800")
        
        # Create a Figure with subplots
        fig = Figure(figsize=(10, 8))
        
        # Plot the kymograph
        ax1 = fig.add_subplot(211)
        
        # Extract the blue channel (for SSB visualization)
        b_channel = self.img[:, :, 2]
        
        # Apply ROI if specified
        roi_applied = False
        if all(v is not None for v in self.roi_coords.values()):
            x_left = self.roi_coords['start_x']
            x_right = self.roi_coords['end_x']
            y_top = self.roi_coords['start_y']
            y_bottom = self.roi_coords['end_y']
            
            if all([x_left, x_right, y_top, y_bottom]):
                roi_img = b_channel[y_top:y_bottom, x_left:x_right]
                if roi_img.size > 0:  # Check if ROI is valid
                    b_channel = roi_img
                    roi_applied = True
        
        # Apply contrast stretching
        vmax = 75  # Suitable for blue channel (SSB)
        b_normalized = (b_channel / vmax) * 255
        b_normalized[b_normalized > 255] = 255
        
        im = ax1.imshow(b_channel, cmap='gray', aspect='auto', vmax=vmax)
        ax1.set_title("Kymograph - Blue Channel (SSB)")
        
        if roi_applied:
            ax1.set_xlabel("Time (pixels) - ROI")
            ax1.set_ylabel("Position (pixels) - ROI")
        else:
            ax1.set_xlabel("Time (pixels)")
            ax1.set_ylabel("Position (pixels)")
        
        # Add colorbar
        fig.colorbar(im, ax=ax1, label='Intensity')
        
        # Plot trajectories on the kymograph
        for i, trace in enumerate(self.traces):
            # Convert physical units back to pixel coordinates for plotting on the kymograph
            if roi_applied:
                time_idx = trace['time_idx']
                coord_idx = trace['coordinate_idx']
            else:
                time_idx = (trace['Time'] * 1000 / self.time_per_line).astype(int)
                coord_idx = (trace['Position'] / self.px_size).astype(int)
            
            ax1.plot(time_idx, coord_idx, 'o-', markersize=2, linewidth=1, label=f'Trace {i+1}')
        
        # Plot the trajectories in physical units
        ax2 = fig.add_subplot(212)
        
        for i, (trace, smooth_trace) in enumerate(zip(self.traces, self.smoothed_traces)):
            ax2.plot(trace['Time'], trace['Position'], 'o', alpha=0.3, markersize=2, label=f'Raw {i+1}')
            ax2.plot(smooth_trace['Time'], smooth_trace['Position'], '-', linewidth=2, label=f'Smoothed {i+1}')
        
        ax2.set_title("SSB Trajectories")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Position (µm)")
        ax2.grid(True, alpha=0.3)
        
        # Add legend with small font
        ax2.legend(loc='best', fontsize='small')
        
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
        
        # Add DNAp-SSB Button
        dnap_ssb_button = ttk.Button(button_frame, text="Plot DNAp-SSB", command=self.plot_dnap_ssb)
        dnap_ssb_button.pack(side=tk.LEFT, padx=5)
        
        # Add Save High-Res Figure Button
        save_button = ttk.Button(button_frame, text="Save High-Res Figure", command=lambda: self.save_high_res(fig))
        save_button.pack(side=tk.LEFT, padx=5)
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to plot SSB trajectories: {e}")

def plot_dnap_ssb(self):
    """Plot DNAp and SSB trajectories together"""
    if self.trace is None:
        messagebox.showinfo("Info", "Please load a trace file first.")
        return
    
    if self.traces is None or len(self.traces) == 0:
        messagebox.showinfo("Info", "No SSB trajectories detected. Please detect SSB trajectories first.")
        return
    
    try:
        # Create a new Toplevel window for the plot
        plot_window = tk.Toplevel(self.master)
        plot_window.title("DNAp and SSB Trajectories")
        plot_window.geometry("1000x800")
        
        # Create a Figure
        fig = Figure(figsize=(10, 8))
        
        # Create subplots
        ax1 = fig.add_subplot(211)  # For trajectories
        ax2 = fig.add_subplot(212)  # For distance
        
        # Get DNAp data
        dnap_time = self.trace['time'] if 'time' in self.trace.columns else self.trace['Time']
        dnap_pos = self.trace['junction_position_all'] if 'junction_position_all' in self.trace.columns else self.trace['Position']
        
        # Plot DNAp trajectory
        ax1.plot(dnap_time, dnap_pos, 'g-', linewidth=2, label='DNAp')
        
        # Plot SSB trajectories
        for i, trace in enumerate(self.smoothed_traces):
            ax1.plot(trace['Time'], trace['Position'], '-', linewidth=1.5, label=f'SSB {i+1}')
        
        # Calculate distance between DNAp and SSB
        all_times = []
        all_distances = []
        
        # For each SSB trajectory, calculate distance to DNAp
        for i, ssb_trace in enumerate(self.smoothed_traces):
            # Create interpolation function for DNAp
            if len(dnap_time) > 1:  # Ensure we have enough points for interpolation
                dnap_interp = interpolate.interp1d(
                    dnap_time, dnap_pos, 
                    bounds_error=False, fill_value="extrapolate"
                )
                
                # Interpolate DNAp position at SSB timepoints
                ssb_time = ssb_trace['Time']
                dnap_at_ssb = dnap_interp(ssb_time)
                
                # Calculate distance
                distance = np.abs(ssb_trace['Position'] - dnap_at_ssb)
                
                # Store data for plotting
                all_times.extend(ssb_time)
                all_distances.extend(distance)
                
                # Plot distance for this trace
                ax2.plot(ssb_time, distance, '-', linewidth=1.5, label=f'Distance {i+1}')
        
        # Set labels and titles
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position (µm)')
        ax1.set_title('DNAp and SSB Trajectories')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Distance (µm)')
        ax2.set_title('Distance Between DNAp and SSB')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        
        # Store the distance data
        if all_times and all_distances:
            # Sort by time
            idx = np.argsort(all_times)
            all_times = np.array(all_times)[idx]
            all_distances = np.array(all_distances)[idx]
            
            self.distance_data = pd.DataFrame({
                'Time': all_times,
                'Distance': all_distances
            })
        
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
        
        # Add Calculate Distance Button
        distance_button = ttk.Button(button_frame, text="Segment Distance", 
                                    command=self.segment_distance)
        distance_button.pack(side=tk.LEFT, padx=5)
        
        # Add Export Distance Data Button
        export_button = ttk.Button(button_frame, text="Export Distance Data", 
                                  command=lambda: self.export_distance_data(all_times, all_distances))
        export_button.pack(side=tk.LEFT, padx=5)
        
        # Add Save High-Res Figure Button
        save_button = ttk.Button(button_frame, text="Save High-Res Figure", 
                                command=lambda: self.save_high_res(fig))
        save_button.pack(side=tk.LEFT, padx=5)
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to plot DNAp and SSB trajectories: {e}") 