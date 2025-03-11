import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nptdms import TdmsFile
from scipy.signal import savgol_filter

# Define numerical coth function for FJC calculations
def coth(x):
    return 1 / np.tanh(x)

# Define tWLC model function (returns end-to-end distance in µm)
def tWLC(F, Lc=2.85056, Lp=56, C=440, g0=-637, g1=17, S=1500):
    return Lc * (1 - 0.5 * (4.1 / (F * Lp))**0.5 + C * F / (-(g0 + g1 * F)**2 + S * C))

# Define FJC model function for a single force value (returns end-to-end distance in µm)
def FJC_single(F, Lss=4.69504, b=1.5, Sss=800):
    return Lss * (coth(F * b / 4.1) - 4.1 / (F * b)) * (1 + F / Sss)

# Set file paths relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
results_folder = os.path.join(parent_dir, 'results')
filename = os.path.join(parent_dir, 'data', 'OT data example.tdms')
base_name = os.path.splitext(os.path.basename(filename))[0]  # 'OT data example'

# Hardcoded inputs
cycle = '01'
time_from_exo = 6850.0      # Starting time of exo phase in ms
time_to_exo = 75690.0       # Ending time of exo phase in ms
time_from_pol = 78450.0     # Starting time of pol phase in ms
time_to_pol = 131078.0      # Ending time of pol phase in ms
bead_size = 1.76            # Bead size in µm
exo_force = 45.0            # Force during exo phase in pN
pol_force = 10.0            # Force during pol phase in pN
SSB_factor = 0.03           # SSB binding factor at 10 pN in µm
total_basepairs = 8393      # Total basepairs in pkyb1 DNA

# Load TDMS file
tdms_file = TdmsFile(filename)

# Extract data as NumPy arrays
time = np.array(tdms_file['FD Data']['Time (ms)'][:])
force = np.array(tdms_file['FD Data']['Force Channel 0 (pN)'][:])
distance = np.array(tdms_file['FD Data']['Distance 1 (um)'][:])

# Select time ranges for exo and pol phases
indtemp_exo = np.where((time >= time_from_exo) & (time <= time_to_exo))
indtemp_pol = np.where((time >= time_from_pol) & (time <= time_to_pol))

time_range_exo = time[indtemp_exo]
force_range_exo = force[indtemp_exo]
distance_range_exo = distance[indtemp_exo]

time_range_pol = time[indtemp_pol]
force_range_pol = force[indtemp_pol]
distance_range_pol = distance[indtemp_pol]

# Combine exo and pol ranges and sort by time
time_range_all = np.append(time_range_exo, time_range_pol)
force_range_all = np.append(force_range_exo, force_range_pol)
distance_range_all = np.append(distance_range_exo, distance_range_pol)

sort_idx = np.argsort(time_range_all)
time_range_all = time_range_all[sort_idx]
force_range_all = force_range_all[sort_idx]
distance_range_all = distance_range_all[sort_idx]

# Calculate reference lengths for dsDNA and ssDNA
dsDNA_exo_ref = tWLC(exo_force)
dsDNA_pol_ref = tWLC(pol_force)
ssDNA_exo_ref = FJC_single(exo_force)
ssDNA_pol_ref = FJC_single(pol_force)

# Calculate ssDNA percentages
ssDNA_exo_percentage = (distance_range_exo - bead_size - dsDNA_exo_ref) / (ssDNA_exo_ref - SSB_factor - dsDNA_exo_ref)
ssDNA_pol_percentage = (distance_range_pol - bead_size - dsDNA_pol_ref) / (ssDNA_pol_ref - SSB_factor - dsDNA_pol_ref)
ssDNA_all_percentage = np.append(ssDNA_exo_percentage, ssDNA_pol_percentage)[sort_idx]

# Calculate basepairs
basepairs = (1 - ssDNA_all_percentage) * total_basepairs

# Calculate junction positions
junction_position_exo = (distance_range_exo - bead_size) - (ssDNA_exo_percentage * ssDNA_exo_ref) * (distance_range_exo - bead_size) / ((ssDNA_exo_percentage * ssDNA_exo_ref) + (1 - ssDNA_exo_percentage) * dsDNA_exo_ref)
junction_position_pol = (distance_range_pol - bead_size) - (ssDNA_pol_percentage * ssDNA_pol_ref) * (distance_range_pol - bead_size) / ((ssDNA_pol_percentage * ssDNA_pol_ref) + (1 - ssDNA_pol_percentage) * dsDNA_pol_ref)
junction_position_all = np.append(junction_position_exo, junction_position_pol)[sort_idx]

# Apply Savitzky-Golay filter to basepairs
bp_filter = savgol_filter(basepairs, 31, 3)

# Define font settings for plots
font = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 16}

# Create /results folder if it doesn't exist
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Function to save plot if file doesn't exist
def save_plot_if_not_exists(filename):
    if os.path.exists(filename):
        print(f"File {filename} already exists. Skipping.")
        return False
    return True

# Plot 1: Basepair Change (Filtered)
plot_filename = os.path.join(results_folder, f"{base_name}-cycle#{cycle}-BasepairChange-filterd.png")
if save_plot_if_not_exists(plot_filename):
    plt.figure(figsize=(6, 4))
    plt.xlabel('Time (s)', fontdict=font)
    plt.ylabel('Basepairs', fontdict=font)
    plt.plot(time_range_all / 1000, basepairs, color='lightgrey', linewidth=1)
    plt.plot(time_range_all / 1000, bp_filter, color='green', linewidth=1, label='Basepairs')
    plt.tight_layout()
    plt.savefig(plot_filename, format='png', dpi=300, bbox_inches='tight')
    plt.close()

# Plot 2: ssDNA Percentage (as Basepairs)
plot_filename = os.path.join(results_folder, f"{base_name}-cycle#{cycle}-ssDNA_percentage.png")
if save_plot_if_not_exists(plot_filename):
    plt.figure(figsize=(8, 3))
    plt.ylabel('Basepairs (bp)', fontdict=font)
    plt.xlabel('Time (s)', fontdict=font)
    plt.scatter(time_range_all / 1000, basepairs, color='black', s=0.5, label='End-to-End Distance')
    plt.tight_layout()
    plt.savefig(plot_filename, format='png', dpi=300)
    plt.close()

# Plot 3: DNA Polymerase Traces
plot_filename = os.path.join(results_folder, f"{base_name}-cycle#{cycle}-DNApTraces.png")
if save_plot_if_not_exists(plot_filename):
    plt.figure(figsize=(6, 4))
    plt.title('Time (s)', fontdict=font)
    plt.ylabel('Distance (µm)', fontdict=font)
    plt.scatter(time_range_all / 1000, distance_range_all - bead_size, color='black', s=2, label='End-to-End Distance')
    plt.scatter(time_range_all / 1000, junction_position_all, color='green', s=2, label='DNA Polymerase Trace')
    plt.fill_between(time_range_all / 1000, distance_range_all - bead_size, junction_position_all, color='gray', alpha=0.2)
    plt.ylim(0, 3.8)
    plt.xlim(0, 159)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.xaxis.set_ticks_position('top')
    plt.tight_layout()
    plt.savefig(plot_filename, format='png', dpi=300)
    plt.close()

# Plot 4: Basepair Change (Raw)
plot_filename = os.path.join(results_folder, f"{base_name}-cycle#{cycle}-BasepairChange.png")
if save_plot_if_not_exists(plot_filename):
    plt.figure(figsize=(8, 3))
    plt.xlabel('Time (s)', fontdict=font)
    plt.ylabel('Basepairs', fontdict=font)
    plt.plot(time_range_all / 1000, basepairs, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=2, label='Basepairs')
    plt.tight_layout()
    plt.savefig(plot_filename, format='png', dpi=300, bbox_inches='tight')
    plt.close()

# Save processed data to Excel in /results folder
excel_filename = os.path.join(results_folder, f"{base_name}-cycle#{cycle}processedData.xlsx")
data = {
    'time': time_range_all,
    'ssDNA_all_percentage': ssDNA_all_percentage,
    'junction_position_all': junction_position_all,
    'basepairs': basepairs
}
df = pd.DataFrame(data)
with pd.ExcelWriter(excel_filename) as writer:
    df.to_excel(writer)

print("Processing complete. Results saved to /results folder.")