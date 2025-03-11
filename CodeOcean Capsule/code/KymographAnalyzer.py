import os
import numpy as np
import lumicks.pylake as lk  
import cv2
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import collections as mc
from matplotlib import colors as mcolors
import pandas as pd
from nptdms import TdmsFile
from scipy.signal import savgol_filter
import tifffile as tif
from scipy import interpolate
from scipy.interpolate import interp1d

# Suppress Matplotlib warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Set file paths relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
results_folder = os.path.join(parent_dir, 'results')
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

kymo_filename = os.path.join(parent_dir, 'data', 'image data example.tdms')
kymo_cycle = '#1'
base_name = os.path.splitext(os.path.basename(kymo_filename))[0]
trace_file = os.path.join(results_folder, 'OT data example-cycle#01processedData.xlsx')

# Embed fonts for portability
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Read TDMS file and extract metadata and image data
metadata = TdmsFile.read_metadata(kymo_filename)
width = metadata.properties['Pixels per line']
px_size = float(metadata.properties['Scan Command.Scan Command.scanning_axes.0.pix_size_nm'])
px_dwell_time = float(metadata.properties['Scan Command.PI Fast Scan Command.pixel_dwell_time_ms'])
inter_frame_time = float(metadata.properties['Scan Command.PI Fast Scan Command.inter_frame_wait_time_ms'])

tdms_file = TdmsFile(kymo_filename)
kymo_time = tdms_file['Data']['Time (ms)'][:]
kymo_time = np.array([int(i) for i in kymo_time])
kymo_position = tdms_file['Data']['Actual position X (um)'][:]
kymo_position = np.array([int(i) for i in kymo_position])
height = len(kymo_time) / width
time_per_line = kymo_time[-1] / height

chn_r = tdms_file['Data']['Pixel ch 1'][:]
chn_r = np.array([int(i) for i in chn_r])
chn_g = tdms_file['Data']['Pixel ch 2'][:]
chn_g = np.array([int(i) for i in chn_g])
chn_b = tdms_file['Data']['Pixel ch 3'][:]
chn_b = np.array([int(i) for i in chn_b])

chn_rgb = np.vstack((chn_r, chn_g, chn_b)).T
img = chn_rgb.reshape((int(height), int(width), 3))
img = img.transpose((1, 0, 2))
img = img.astype(np.uint16)

# Plot and save the full image
plt.figure(figsize=(6, 4))
plt.imshow(img.astype('uint16'), vmax=255)  # Normalize to avoid clipping warning
plt.xlabel('Time/px')
plt.ylabel('Position/px')
plt.tight_layout()
plot_filename = os.path.join(results_folder, f"{base_name}-cycle{kymo_cycle}-full_image.png")
if not os.path.exists(plot_filename):
    plt.savefig(plot_filename, format='png', dpi=300, bbox_inches='tight')
plt.close()
tif.imwrite(os.path.join(results_folder, f"{base_name}.tiff"), img)

# Split image into channels
b, g, r = cv2.split(img)  # b is chn_r (red/SSB), g is chn_g (green/DNAp), r is chn_b (blue)

# Plot green channel (DNA Polymerase)
vmax = 255  # Adjust based on your data range
fig, ax = plt.subplots(figsize=(10, 4))
ax.imshow(g.astype('uint16'), cmap='gray', vmax=vmax, aspect="auto")
plt.xlabel("Time/px")
plt.ylabel("Position/px")
kymo_xlim_left = 40
kymo_xlim_right = 1234
kymo_ylim_top = 3
kymo_ylim_bottom = 60
ax.set_xlim(kymo_xlim_left, kymo_xlim_right)
ax.set_ylim(kymo_ylim_bottom, kymo_ylim_top)
plot_filename = os.path.join(results_folder, f"{base_name}-cycle{kymo_cycle}-splited_channel_green.png")
if not os.path.exists(plot_filename):
    plt.savefig(plot_filename, format='png', dpi=300, bbox_inches='tight')
plt.close()

# Plot red channel (SSB)
vmax_red = 75
r_normalized = (b / vmax_red) * 255  # Normalize red channel (SSB)
r_normalized[r_normalized > 255] = 255
red_image = np.zeros_like(img)
red_image[:, :, 2] = r_normalized.astype(np.uint16)  # Set red channel in BGR
fig, ax = plt.subplots(figsize=(10, 4))
ax.imshow(cv2.cvtColor(red_image, cv2.COLOR_BGR2RGB), aspect="auto", vmin=0, vmax=255)
ax.set_xlim(kymo_xlim_left, kymo_xlim_right)
ax.set_ylim(kymo_ylim_bottom, kymo_ylim_top)
plot_filename = os.path.join(results_folder, f"{base_name}-cycle{kymo_cycle}-splited_channel_red.png")
if not os.path.exists(plot_filename):
    plt.savefig(plot_filename, format='png', dpi=300, bbox_inches='tight')
plt.close()

# Read and plot DNA Polymerase trace from Excel
trace = pd.read_excel(trace_file)
# Removed print(trace.head()) to suppress output

plt.figure(figsize=(8, 6))
plt.xlabel("Time/px")
plt.ylabel("Position/px")
plt.title("DNA Polymerase Trace")
x_offset_searching = -23
y_offset_searching = 7
x_cali = 1000 / time_per_line
y_cali = 1000 / px_size
trace_time = trace['time'] / 1000 * x_cali + x_offset_searching
trace_time = trace_time.dropna()
position = pd.to_numeric(trace['junction_position_all'], errors='coerce')
position = position * y_cali + y_offset_searching
position = position.dropna()
plt.plot(trace_time, position, 'blue', linewidth=1, label='first trial correlation')
plot_filename = os.path.join(results_folder, f"{base_name}-cycle{kymo_cycle}-overlapped_DNAp_image_with_color.png")
if not os.path.exists(plot_filename):
    plt.savefig(plot_filename, format='png', dpi=300, bbox_inches='tight')
plt.close()

# Overlay DNAp trace on SSB image
fig, ax = plt.subplots(figsize=(10, 4))
ax.imshow(cv2.cvtColor(red_image, cv2.COLOR_BGR2RGB), aspect="auto", vmin=0, vmax=255)
ax.set_xlim(kymo_xlim_left, kymo_xlim_right)
ax.set_ylim(kymo_ylim_bottom, kymo_ylim_top)
plt.xlabel("Time/px")
plt.ylabel("Position/px")
plt.title("SSB Trace")
ax.plot(trace_time, position, 'yellow', linewidth=1.5, label='first trial correlation')
plot_filename = os.path.join(results_folder, f"{base_name}-cycle{kymo_cycle}-overlapped_SSB_image_with_color.png")
if not os.path.exists(plot_filename):
    plt.savefig(plot_filename, format='png', dpi=300, bbox_inches='tight')
plt.close()

# Detect SSB trajectories in ROI
roi_start_y, roi_end_y = 31, 40
roi_start_x, roi_end_x = 766, 1058
b_ROI = b[roi_start_y:roi_end_y, roi_start_x:roi_end_x]
traces = lk.track_greedy(b_ROI, line_width=5, pixel_threshold=25, window=6)
traces = lk.filter_lines(traces, 3)
traces = lk.refine_lines_centroid(traces, line_width=5)
print(f"Number of traces detected: {len(traces)}")

# Plot detected traces
plt.figure()
plt.imshow(b, aspect="auto", vmax=50, cmap='gray')
for trace in traces:
    plt.plot(np.array(trace.time_idx) + roi_start_x, np.array(trace.coordinate_idx) + roi_start_y)
plot_filename = os.path.join(results_folder, f"{base_name}-cycle{kymo_cycle}-detected_traces.png")
if not os.path.exists(plot_filename):
    plt.savefig(plot_filename, format='png', dpi=300, bbox_inches='tight')
plt.close()

# Smooth traces and plot
window_size = 31
poly_order = 3
plt.figure()
plt.imshow(b, aspect="auto", vmax=50, cmap='gray')
for trace in traces:
    plt.plot(np.array(trace.time_idx) + roi_start_x, np.array(trace.coordinate_idx) + roi_start_y)
for trace in traces:
    time_idx_smooth = savgol_filter(np.array(trace.time_idx), window_size, poly_order)
    coordinate_idx_smooth = savgol_filter(np.array(trace.coordinate_idx), window_size, poly_order)
    plt.plot(time_idx_smooth + roi_start_x, coordinate_idx_smooth + roi_start_y)
plot_filename = os.path.join(results_folder, f"{base_name}-cycle{kymo_cycle}-smoothed_traces.png")
if not os.path.exists(plot_filename):
    plt.savefig(plot_filename, format='png', dpi=300, bbox_inches='tight')
plt.close()

# Plot DNAp and SSB trajectories together
fig, ax = plt.subplots(figsize=(6, 4))
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12})
ax.imshow(cv2.cvtColor(red_image, cv2.COLOR_BGR2RGB), aspect="auto", vmin=0, vmax=255)
ax.set_xlim(kymo_xlim_left, kymo_xlim_right)
ax.set_ylim(kymo_ylim_bottom, kymo_ylim_top)
plt.xlabel("Time/px")
plt.ylabel("Position/px")
plt.title("SSB Trace")
ax.plot(trace_time, position, 'green', linewidth=1.5)
for trace in traces:
    time_idx_smooth = savgol_filter(np.array(trace.time_idx), window_size, poly_order)
    coordinate_idx_smooth = savgol_filter(np.array(trace.coordinate_idx), window_size, poly_order)
    ax.plot(np.array(trace.time_idx) + roi_start_x, np.array(trace.coordinate_idx) + roi_start_y, 'lightgray', linewidth=0.2)
    ax.plot(time_idx_smooth + roi_start_x, coordinate_idx_smooth + roi_start_y, "yellow", linewidth=1.5)
plot_filename = os.path.join(results_folder, f"{base_name}-cycle{kymo_cycle}-overlapped_DNAp+SSB_detected_Traje.png")
if not os.path.exists(plot_filename):
    plt.savefig(plot_filename, format='png', dpi=300, bbox_inches='tight')
plt.close()

# Plot in physical units
fig, ax = plt.subplots(figsize=(6, 4))
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12})
trace_time_s = (np.array(trace_time) - x_offset_searching) / x_cali
position_um = (np.array(position) - y_offset_searching) / y_cali
trace_time_s_filter = savgol_filter(trace_time_s, window_length=21, polyorder=3)
position_um_filter = savgol_filter(position_um, window_length=21, polyorder=3)
for trace in traces:
    time_idx_smooth = savgol_filter(np.array(trace.time_idx), window_size, poly_order)
    coordinate_idx_smooth = savgol_filter(np.array(trace.coordinate_idx), window_size, poly_order)
    time_idx_smooth_s = (time_idx_smooth + roi_start_x - x_offset_searching) / x_cali
    coordinate_idx_smooth_um = (coordinate_idx_smooth + roi_start_y - y_offset_searching) / y_cali
    ax.plot((np.array(trace.time_idx) + roi_start_x - x_offset_searching) / x_cali, (np.array(trace.coordinate_idx) + roi_start_y - y_offset_searching) / y_cali, 'gray', linewidth=0.2)
    ax.plot(time_idx_smooth_s, coordinate_idx_smooth_um, "red", linewidth=2, label="SSB trace" if trace is traces[0] else "")
ax.plot(trace_time_s, position_um, 'gray', linewidth=0.3)
ax.plot(trace_time_s_filter, position_um_filter, 'green', linewidth=2, label="DNA polymerase trace")
ax.set_xlim(77.5, 131.5)
ax.set_ylim(1.4, 2.6)
ax.invert_yaxis()
plt.xlabel("Time (s)")
plt.ylabel("Distance (um)")
plt.legend()
plt.tight_layout()
plot_filename = os.path.join(results_folder, f"{base_name}-cycle{kymo_cycle}-detected_DNAp+SSB.png")
if not os.path.exists(plot_filename):
    plt.savefig(plot_filename, format='png', dpi=300, bbox_inches='tight')
plt.close()

# Calculate and plot distance
interp_DNAp = interp1d(trace_time_s_filter, position_um_filter)
position_diff = -(interp_DNAp(time_idx_smooth_s) - coordinate_idx_smooth_um)
fig, ax = plt.subplots(figsize=(6, 4))
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12})
ax.scatter(time_idx_smooth_s, position_diff, c='teal', marker='o', edgecolors='gray', linewidths=0.5, s=10)
ax.plot(savgol_filter(time_idx_smooth_s, 71, 3), savgol_filter(position_diff, 71, 3), 'olive', linewidth=1.5)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Distance Between \n DNA Polymerase and SSB (um)')
plt.tight_layout()
plot_filename = os.path.join(results_folder, f"{base_name}-cycle{kymo_cycle}-distance_between_DNAp+SSB.png")
if not os.path.exists(plot_filename):
    plt.savefig(plot_filename, format='png', dpi=300, bbox_inches='tight')
plt.close()

# Plot individual SSB traces
n_traces = len(traces)
fig, axs = plt.subplots(n_traces, figsize=(10, 4 * n_traces))
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12})
for i, trace in enumerate(traces):
    time_idx_smooth = savgol_filter(np.array(trace.time_idx), window_size, poly_order)
    coordinate_idx_smooth = savgol_filter(np.array(trace.coordinate_idx), window_size, poly_order)
    time_idx_smooth_s = (time_idx_smooth + roi_start_x - x_offset_searching) / x_cali
    coordinate_idx_smooth_um = (coordinate_idx_smooth + roi_start_y - y_offset_searching) / y_cali
    ax = axs if n_traces == 1 else axs[i]
    ax.plot((np.array(trace.time_idx) + roi_start_x - x_offset_searching) / x_cali, (np.array(trace.coordinate_idx) + roi_start_y - y_offset_searching) / y_cali, 'gray', linewidth=0.2)
    ax.plot(time_idx_smooth_s, coordinate_idx_smooth_um, "red", linewidth=2, label="SSB trace")
    ax.plot(trace_time_s, position_um, 'gray', linewidth=0.3)
    ax.plot(trace_time_s_filter, position_um_filter, 'green', linewidth=2, label="DNA polymerase trace")
    ax.invert_yaxis()
plt.xlabel("Time (s)")
plt.ylabel("Distance (um)")
plt.legend()
plt.tight_layout()
plot_filename = os.path.join(results_folder, f"{base_name}-cycle{kymo_cycle}-all_traces_DNAp+SSB.png")
if not os.path.exists(plot_filename):
    plt.savefig(plot_filename, format='png', dpi=300, bbox_inches='tight')
plt.close()

# Plot distances for individual traces
fig, axs = plt.subplots(n_traces, figsize=(10, 4 * n_traces))
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12})
for i, trace in enumerate(traces):
    time_idx_smooth_s = (time_idx_smooth + roi_start_x - x_offset_searching) / x_cali
    coordinate_idx_smooth_um = (coordinate_idx_smooth + roi_start_y - y_offset_searching) / y_cali
    position_diff = -(interp_DNAp(time_idx_smooth_s) - coordinate_idx_smooth_um)
    ax = axs if n_traces == 1 else axs[i]
    ax.scatter(time_idx_smooth_s, position_diff, c='teal', marker='o', edgecolors='gray', linewidths=0.5, s=10)
    ax.plot(savgol_filter(time_idx_smooth_s, 71, 3), savgol_filter(position_diff, 71, 3), 'olive', linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('Distance Between \n DNA Polymerase and SSB (um)')
plt.tight_layout()
plot_filename = os.path.join(results_folder, f"{base_name}-cycle{kymo_cycle}-all_distances_DNAp+SSB.png")
if not os.path.exists(plot_filename):
    plt.savefig(plot_filename, format='png', dpi=300, bbox_inches='tight')
plt.close()

# Save data to CSV
save_dir = os.path.join(results_folder, f"{base_name}-cycle{kymo_cycle}-analyzed_data")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
df_DNAp = pd.DataFrame({'time_s': trace_time_s, 'position_um': position_um, 'time_s_filter': trace_time_s_filter, 'position_um_filter': position_um_filter})
df_DNAp.to_csv(os.path.join(save_dir, 'DNAp_trace.csv'), index=False)
for i, trace in enumerate(traces):
    time_idx_smooth_s = (time_idx_smooth + roi_start_x - x_offset_searching) / x_cali
    coordinate_idx_smooth_um = (coordinate_idx_smooth + roi_start_y - y_offset_searching) / y_cali
    position_diff = -(interp_DNAp(time_idx_smooth_s) - coordinate_idx_smooth_um)
    df_trace = pd.DataFrame({
        'time_s': time_idx_smooth_s,
        'position_um': coordinate_idx_smooth_um,
        'position_diff_um': position_diff
    })
    df_trace.to_csv(os.path.join(save_dir, f'trace_{i}.csv'), index=False)

# Segment analysis functions
def first_derivative(x, y, window_size):
    first_derivative_values = np.zeros(len(y))
    for i in range(len(y)):
        if i < window_size:
            dy = y[i + window_size] - y[0]
            dx = x[i + window_size] - x[0]
        elif i > len(y) - window_size - 1:
            dy = y[-1] - y[i - window_size]
            dx = x[-1] - x[i - window_size]
        else:
            dy = y[i + window_size] - y[i - window_size]
            dx = x[i + window_size] - x[i - window_size]
        first_derivative_values[i] = dy / dx
    return first_derivative_values

def segment_data(derivative):
    segments = []
    start_index = 0
    current_sign = 0 if abs(derivative[0]) < 0.01 else np.sign(derivative[0])
    for i in range(1, len(derivative)):
        sign = 0 if abs(derivative[i]) < 0.01 else np.sign(derivative[i])
        if sign != current_sign:
            if current_sign == 0 and i - start_index < 10 and len(segments) > 0:
                start, _ = segments.pop()
                segments.append((start, i - 1))
            else:
                segments.append((start_index, i - 1))
            start_index = i
            current_sign = sign
    if len(derivative) - start_index < 10 and len(segments) > 0 and current_sign == 0:
        start, _ = segments.pop()
        segments.append((start, len(derivative) - 1))
    else:
        segments.append((start_index, len(derivative) - 1))
    return segments

def process_segments(time, position, segments):
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
        segmented_rates.append(0 if time[end] == time[start] else (position[end] - position[start]) / (time[end] - time[start]))
    for i in range(1, len(segmented_position_end)):
        segmented_diffs.append(segmented_position_start[i] - segmented_position_end[i - 1])
    segmented_diffs.append(np.nan)
    return segmented_time_start, segmented_time_end, segmented_position_start, segmented_position_end, segmented_rates, segmented_diffs

# Perform segment analysis
filtered_position_diff = savgol_filter(position_diff, 51, 3)
derivative = first_derivative(time_idx_smooth_s, filtered_position_diff, window_size=3)
segments = segment_data(derivative)
segmented_time_start, segmented_time_end, segmented_position_start, segmented_position_end, segmented_rates, segmented_diffs = process_segments(time_idx_smooth_s, position_diff, segments)

# Plot segmented data
fig, ax = plt.subplots(figsize=(6, 4))
for start_time, end_time, start_position, end_position in zip(segmented_time_start, segmented_time_end, segmented_position_start, segmented_position_end):
    ax.plot([start_time, end_time], [start_position, end_position], marker='o')
ax.scatter(time_idx_smooth_s, position_diff, c='teal', marker='o', edgecolors='gray', linewidths=0.5, s=10)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position (um)')
ax.set_title('Segmented Time vs Position')
ax.grid(True)
plt.tight_layout()
plot_filename = os.path.join(results_folder, f"{base_name}-cycle{kymo_cycle}-Segmented_Time_Position.png")
if not os.path.exists(plot_filename):
    plt.savefig(plot_filename, format='png', dpi=300, bbox_inches='tight')
plt.close()

# Save segmented data
segmented_data_filename = os.path.join(results_folder, f"{base_name}-cycle{kymo_cycle}-segmented_data.csv")
df = pd.DataFrame({
    'segment_start_time': segmented_time_start,
    'segment_end_time': segmented_time_end,
    'segment_start_position': segmented_position_start,
    'segment_end_position': segmented_position_end,
    'segment_rate': segmented_rates
})
df.to_csv(segmented_data_filename, index=False)

print("Processing complete. All results saved to /results folder.")