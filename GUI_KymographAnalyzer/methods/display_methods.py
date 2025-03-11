"""
Display methods for the Correlation Image Force Analyzer.
"""
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import cv2
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector, Cursor

def update_contrast_from_slider(self, val):
    """Update the contrast based on the slider's value"""
    try:
        if not self.updating_display:
            self.contrast_max = float(val)
            # Update the contrast label
            self.contrast_label.configure(text=f"{int(val)}%")
            # Safely update the display
            self.master.after(100, self.display_image_left)  # Add small delay to avoid rapid updates
    except Exception as e:
        print(f"Error updating contrast: {e}")

def display_single_channel(self, channel):
    """Display a single color channel in the main window"""
    if self.img is None:
        messagebox.showinfo("Info", "Please load a kymograph file first.")
        return

    try:
        # Update the selected channel
        self.selected_channel = channel
        self.channel_var.set(channel)
        
        # Display the image
        self.display_image_left()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to display {channel} channel: {e}")
        print(f"Error displaying channel: {e}")

def display_image_left(self):
    """Display the kymograph in the main window"""
    if self.updating_display:
        return

    self.updating_display = True
    try:
        if self.img is None:
            self.updating_display = False
            return

        # Clear the existing plot
        if hasattr(self, 'ax') and self.ax is not None:
            self.ax.clear()
        else:
            # Create a figure and axes if they don't exist
            if not hasattr(self, 'fig') or self.fig is None:
                self.fig = Figure(figsize=(10, 8), dpi=100)
                
            if not hasattr(self, 'canvas') or self.canvas is None:
                frame = getattr(self, 'right_panel', self.master)
                self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
                self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                # Add navigation toolbar
                if not hasattr(self, 'toolbar') or self.toolbar is None:
                    self.toolbar_frame = ttk.Frame(frame)
                    self.toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
                    self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
                    self.toolbar.update()
                
            # Create subplot if it doesn't exist
            self.ax = self.fig.add_subplot(111)

        # Extract the selected channel
        channel_idx = {'Red': 0, 'Green': 1, 'Blue': 2}[self.selected_channel]
        img_channel = self.img[:, :, channel_idx]

        # Apply ROI if specified
        roi_applied = False
        try:
            x_left = int(self.roi_x_left_entry.get()) if self.roi_x_left_entry.get() else None
            x_right = int(self.roi_x_right_entry.get()) if self.roi_x_right_entry.get() else None
            y_top = int(self.roi_y_top_entry.get()) if self.roi_y_top_entry.get() else None
            y_bottom = int(self.roi_y_bottom_entry.get()) if self.roi_y_bottom_entry.get() else None
            
            if all(v is not None for v in [x_left, x_right, y_top, y_bottom]):
                roi_img = img_channel[y_top:y_bottom, x_left:x_right]
                if roi_img.size > 0:  # Check if ROI is valid
                    img_channel = roi_img
                    roi_applied = True
                    # Update ROI coordinates
                    self.roi_coords = {'start_x': x_left, 'end_x': x_right, 
                                      'start_y': y_top, 'end_y': y_bottom}
        except (ValueError, TypeError):
            pass  # Use the whole image if ROI inputs are invalid

        # Apply contrast stretching
        p2, p98 = np.percentile(img_channel, (2, self.contrast_max))
        img_rescale = np.clip(img_channel, p2, p98)

        # Plot the image
        im = self.ax.imshow(img_rescale, cmap='gray', aspect='auto')
        
        # Set plot title and labels
        self.ax.set_title(f"Kymograph - {self.selected_channel} Channel")
        self.ax.set_xlabel("Time (pixels)")
        self.ax.set_ylabel("Position (pixels)")
        
        # Add colorbar (remove old one if it exists)
        if hasattr(self, 'colorbar') and self.colorbar is not None:
            try:
                self.colorbar.remove()
            except:
                pass  # If removal fails, just continue
                
        self.colorbar = self.fig.colorbar(im, ax=self.ax)
        self.colorbar.set_label('Intensity')
        
        # Set ROI limits if specified
        if roi_applied:
            self.ax.set_xlim(0, x_right - x_left)
            self.ax.set_ylim(y_bottom - y_top, 0)
            self.ax.set_title(f"Kymograph ROI - {self.selected_channel} Channel")
            self.status_label.configure(text=f"ROI: X={x_left}:{x_right}, Y={y_top}:{y_bottom}")
        
        # Add Rectangle Selector for ROI
        self.rect_selector = RectangleSelector(
            self.ax, self.onselect_rect, useblit=True,
            button=[1],  # Left mouse button
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )

        # Adjust layout
        self.fig.tight_layout()
        
        # Redraw the canvas
        self.canvas.draw()
        
        # Update status if not already set
        if not roi_applied:
            self.status_label.configure(text=f"Displaying {self.selected_channel} channel")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to display image: {e}")
        print(f"Error displaying image: {e}")  # Print for debugging
    finally:
        self.updating_display = False

def save_high_res(self, fig=None):
    """Save the figure in high resolution"""
    try:
        if fig is None:
            if hasattr(self, 'fig') and self.fig is not None:
                fig = self.fig
            else:
                messagebox.showerror("Error", "No figure available to save")
                return
            
        file_path = tk.filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*")]
        )
        
        if file_path:
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Figure saved to {file_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save figure: {e}")
        print(f"Error saving figure: {e}")  # Print for debugging 