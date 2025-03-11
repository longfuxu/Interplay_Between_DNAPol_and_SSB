"""
Utility methods for the Correlation Image Force Analyzer.
"""
import tkinter as tk
from tkinter import messagebox

def update_channel(self):
    """Update the selected channel from the combobox"""
    new_channel = self.channel_var.get()
    if new_channel != self.selected_channel:
        self.selected_channel = new_channel
        self.display_image_left()

def on_entry_focus(self, event):
    """Handle entry focus event"""
    self.active_entry = event.widget
    event.widget.configure(bg='lightyellow')  # Highlight active entry

def on_entry_focus_out(self, event):
    """Handle entry focus out event"""
    if self.active_entry == event.widget:
        self.active_entry = None
    event.widget.configure(bg='white')  # Reset background

def on_roi_entry(self, event=None):
    """Handle ROI entry return event"""
    try:
        # Update ROI coordinates
        x_left = int(self.roi_x_left_entry.get().strip())
        x_right = int(self.roi_x_right_entry.get().strip())
        y_top = int(self.roi_y_top_entry.get().strip())
        y_bottom = int(self.roi_y_bottom_entry.get().strip())
        
        # Store ROI coordinates
        self.roi_coords = {
            'start_x': x_left,
            'end_x': x_right,
            'start_y': y_top,
            'end_y': y_bottom
        }
        
        # Update the display
        self.display_image_left()
    except ValueError:
        messagebox.showerror("Error", "Please enter valid integers for ROI coordinates.")

def onselect_rect(self, eclick, erelease):
    """Handle rectangle selection event for ROI"""
    if eclick.ydata is None or erelease.ydata is None:
        return
    
    # Get ROI coordinates
    x1, y1 = int(min(eclick.xdata, erelease.xdata)), int(min(eclick.ydata, erelease.ydata))
    x2, y2 = int(max(eclick.xdata, erelease.xdata)), int(max(eclick.ydata, erelease.ydata))
    
    # Apply ROI offsets if we're working with a sub-region
    if all(v is not None for v in self.roi_coords.values()):
        x1 += self.roi_coords['start_x']
        x2 += self.roi_coords['start_x']
        y1 += self.roi_coords['start_y']
        y2 += self.roi_coords['start_y']
    
    # Update entry fields
    self.roi_x_left_entry.delete(0, tk.END)
    self.roi_x_left_entry.insert(0, str(x1))
    
    self.roi_x_right_entry.delete(0, tk.END)
    self.roi_x_right_entry.insert(0, str(x2))
    
    self.roi_y_top_entry.delete(0, tk.END)
    self.roi_y_top_entry.insert(0, str(y1))
    
    self.roi_y_bottom_entry.delete(0, tk.END)
    self.roi_y_bottom_entry.insert(0, str(y2))
    
    # Update status
    self.status_label.configure(text=f"ROI selected: X={x1}:{x2}, Y={y1}:{y2}")

def on_image_click(self, event):
    """Handle mouse click on the image"""
    if event.button == 3:  # Right click
        # Reset ROI entry fields
        self.roi_x_left_entry.delete(0, tk.END)
        self.roi_x_right_entry.delete(0, tk.END)
        self.roi_y_top_entry.delete(0, tk.END)
        self.roi_y_bottom_entry.delete(0, tk.END)
        
        # Reset ROI coordinates
        self.roi_coords = {'start_x': None, 'end_x': None, 'start_y': None, 'end_y': None}
        
        # Update the display
        self.display_image_left()
        
        # Update status
        self.status_label.configure(text="ROI reset. Right-click to reset, left-click and drag to select ROI.") 