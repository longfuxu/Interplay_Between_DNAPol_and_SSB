import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd

def gradual_change_trace(x, y, noise_stddev):
    """
    Generates a gradual changing trace with pausing events.
    x: A list of lists, each sublist contains the start and end x-coordinate of the segment.
    y: A list of lists, each sublist contains the start and end y-coordinate of the segment.
    noise_stddev: Standard deviation of the Gaussian noise added to the y values.
    """
    x_values = []
    y_values = []

    for i, (x_start, x_end) in enumerate(x):
        y_start, y_end = y[i]
        num_points = int((x_end - x_start) * 1000)
        segment_x = np.linspace(x_start, x_end, num_points)
        segment_y = np.linspace(y_start, y_end, num_points)
        noise = np.random.normal(0, noise_stddev, num_points)
        x_values.extend(segment_x)
        y_values.extend(segment_y + noise)

    return np.array(x_values), np.array(y_values)

# using the np.pad function to pad the data before applying the moving window filter. 
# This way, the window size remains the same throughout the filtering process, 
# but we avoid extreme values by padding the data.
def moving_window_filter(y_values, window_size):
    padding_size = window_size // 2
    padded_y_values = np.pad(y_values, (padding_size, padding_size), mode='edge')
    filtered_y_values = np.zeros(len(y_values))

    for i in range(len(y_values)):
        left = i
        right = i + window_size
        filtered_y_values[i] = np.mean(padded_y_values[left:right])

    return filtered_y_values

# Function to calculate the first derivative while handing the edge effect
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

# Fit a single step to the data
def fit_single_step(data):
    n_points = len(data)

    if n_points <= 5:
        return None, None, np.inf
    
    chi2 = np.zeros(n_points)
    for i in range(1, n_points):
        left_mean = np.mean(data[:i])
        right_mean = np.mean(data[i:])
        chi2[i] = np.sum((data[:i] - left_mean) ** 2) + np.sum((data[i:] - right_mean) ** 2)
    best_loc = np.argmin(chi2[1:-1]) + 1
    step_size = np.mean(data[best_loc:]) - np.mean(data[:best_loc])
    return best_loc, step_size, chi2[best_loc]

# Find steps in the data using the described algorithm
def find_steps(data, max_steps=100):
    step_locs = []
    step_sizes = []
    residuals = []
    remaining_data = np.copy(data)
    for _ in range(max_steps):
        best_loc, step_size, chi2 = fit_single_step(remaining_data)
        step_locs.append(best_loc)
        step_sizes.append(step_size)
        residuals.append(chi2)
        remaining_data[best_loc:] -= step_size
    return step_locs, step_sizes, residuals

# Modified find_steps function to find the optimal steps and step sizes based on step size threshold 
# step size threshold is detemined by the noise level of the y_data with estimate_noise_std function.
def find_optimal_steps(data, max_steps=100, step_size_threshold=None):
    step_locs, step_sizes, residuals = find_steps(data, max_steps)
    
    # Sort the step sizes and corresponding step locations and residuals
    sorted_indices = np.argsort(step_sizes)[::-1]  # Sort from largest to smallest
    sorted_step_sizes = np.array(step_sizes)[sorted_indices]
    sorted_step_locs = np.array(step_locs)[sorted_indices]
    sorted_residuals = np.array(residuals)[sorted_indices]

    if step_size_threshold is not None:
        # Find the index where the step size is smaller than the step_size_threshold
        threshold_index = np.argmax(sorted_step_sizes < step_size_threshold)

        # Get the optimal step locations and sizes
        optimal_step_locs = sorted_step_locs[:threshold_index]
        optimal_step_sizes = sorted_step_sizes[:threshold_index]
    else:
        optimal_step_locs = sorted_step_locs
        optimal_step_sizes = sorted_step_sizes

    # Ensure step locations are unique and sorted
    unique_optimal_step_locs, unique_indices = np.unique(optimal_step_locs, return_index=True)
    unique_optimal_step_sizes = optimal_step_sizes[unique_indices]
    
    sorted_unique_indices = np.argsort(unique_optimal_step_locs)
    sorted_unique_step_locs = unique_optimal_step_locs[sorted_unique_indices]
    sorted_unique_step_sizes = unique_optimal_step_sizes[sorted_unique_indices]

    return sorted_unique_step_locs, sorted_unique_step_sizes, sorted_residuals

# Function to recalculate step sizes based on the mean values between adjacent step locations
def recalculate_step_sizes(data, step_locs):
    step_sizes = []
    n_steps = len(step_locs)
    for i in range(n_steps):
        if i == 0:
            left_data = data[:step_locs[i]]
        else:
            left_data = data[step_locs[i-1]:step_locs[i]]
        
        if i == n_steps - 1:
            right_data = data[step_locs[i]:]
        else:
            right_data = data[step_locs[i]:step_locs[i+1]]
        
        step_sizes.append(np.mean(right_data) - np.mean(left_data))
    return step_sizes

def estimate_noise_std(data, scaling_factor=1.4826):
    # Calculate the difference between consecutive data points
    diff_data = np.diff(data)
    
    # Calculate the median absolute deviation (MAD) of the difference data
    mad = np.median(np.abs(diff_data - np.median(diff_data)))
    
    # Estimate the standard deviation using the scaling factor
    estimated_std = mad * scaling_factor
    return estimated_std

# Main function to detect the change-point of the gradual-changing trace
def detect_steps(x_values, y_values, window_size, scaling_factor):
    filtered_y_values = moving_window_filter(y_values, window_size)
    first_derivative_values = first_derivative(x_values, filtered_y_values, window_size)

    filtered_first_derivative = savgol_filter(first_derivative_values, window_length=5, polyorder=3)
    estimated_noise_std = estimate_noise_std(filtered_first_derivative, scaling_factor=scaling_factor)

    optimal_step_locs, _, sorted_residuals = find_optimal_steps(filtered_first_derivative, step_size_threshold=estimated_noise_std)
    recalculated_step_sizes = recalculate_step_sizes(filtered_first_derivative, optimal_step_locs)

    return optimal_step_locs

# function to reconstruct the fitted data while properly handle the edge of the data
def reconstruct_fitted_data(x_values, filtered_y_values, step_locs):
    fitted_y_values = np.zeros_like(filtered_y_values)
    sorted_step_locs = sorted(step_locs)
    n_step_locs = len(sorted_step_locs)

    for i in range(n_step_locs + 1):
        if i == 0:
            x_start, y_start = x_values[0], filtered_y_values[0]
            x_end, y_end = x_values[sorted_step_locs[i]], filtered_y_values[sorted_step_locs[i]]
            segment_length = sorted_step_locs[i]
        elif i == n_step_locs:
            x_start, y_start = x_values[sorted_step_locs[i - 1]], filtered_y_values[sorted_step_locs[i - 1]]
            x_end, y_end = x_values[-1], filtered_y_values[-1]
            segment_length = len(filtered_y_values) - sorted_step_locs[i - 1]
        else:
            x_start, y_start = x_values[sorted_step_locs[i - 1]], filtered_y_values[sorted_step_locs[i - 1]]
            x_end, y_end = x_values[sorted_step_locs[i]], filtered_y_values[sorted_step_locs[i]]
            segment_length = sorted_step_locs[i] - sorted_step_locs[i - 1]
        
        segment_y = np.linspace(y_start, y_end, segment_length)
        fitted_y_values[sorted_step_locs[i - 1] if i > 0 else 0:sorted_step_locs[i] if i < n_step_locs else len(filtered_y_values)] = segment_y
            
    return fitted_y_values

# Plot all the results out
def plot_data(x_values, y_values, filtered_y_values, first_derivative_values, fitted_y_values):
    """
    The first subplot shows the original data, filtered data, and the first derivative with a double y-axis.
    The second subplot shows the original data and the fitted data. 
    The fitted data are linear segments with x and y both from the filtered_y_values.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # First plot: original data, filtered data, and first derivative with double y-axis
    ax1.plot(x_values, y_values, label='Original Data')
    ax1.plot(x_values[:len(filtered_y_values)], filtered_y_values, label='Filtered Data')
    ax1.set_ylabel('Y Values')
    ax1.legend(loc='upper left')
    
    ax1b = ax1.twinx()
    ax1b.plot(x_values[:len(first_derivative_values)], first_derivative_values, color='g', label='First Derivative')
    ax1b.set_ylabel('First Derivative')
    ax1b.legend(loc='upper right')

    # Second plot: original data and fitted data
    ax2.plot(x_values, y_values, label='Original Data')
    ax2.plot(x_values[:len(fitted_y_values)], fitted_y_values, label='Fitted Data')
    ax2.set_ylabel('Y Values')
    ax2.legend()

# Based on simulated data, we determine the optimal window size with a base_window_size and a noise_level_muliplier
def optimal_window_size(estimated_noise_y_value):
    # You can adjust the constants to better suit your specific dataset
    # These constants can be determined experimentally
    base_window_size = 21
    noise_level_multiplier = 171

    window_size = base_window_size + int(noise_level_multiplier * estimated_noise_y_value)
    
    # Ensure the window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    return window_size

# Based on simulated data, we determine the optimal scaling_factor with a base_scaling_factor and a noise_level_muliplier
def optimal_scaling_factor(estimated_noise_y_value):
    # You can adjust the constants to better suit your specific dataset
    # These constants can be determined experimentally
    base_scaling_factor = 0.8655
    noise_level_multiplier = 2.7

    scaling_factor = base_scaling_factor + noise_level_multiplier * estimated_noise_y_value
    scaling_factor = round(scaling_factor,4)

    return scaling_factor

# Below is an example use case
"""
# Example of the simulated data
x = [[0, 0.1], [0.1, 0.3], [0.3, 0.5], [0.5, 0.7], [0.7, 0.9], [0.9, 1]]
y = [[0, 0.5], [0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 1], [1, 1]]
noise_stddev = 0.02
x_values, y_values = gradual_change_trace(x, y, noise_stddev)

# set the window size and scaling factor to get step-like first-derivative trace
# Calculate the optimal window size based on the estimated noise standard deviation
window_size = optimal_window_size(estimate_noise_std(y_values, scaling_factor=1.3))
filtered_y_values = moving_window_filter(y_values, window_size)

# # to detect the changing-point based on the step-location of the first-derivative
scaling_factor = optimal_scaling_factor(estimate_noise_std(y_values, scaling_factor=1.3))
optimal_step_locs = detect_steps(x_values, y_values, window_size, scaling_factor)

# Print the values 
print("The estimated noise standard deviation of y_value is {:.2f}".format(estimate_noise_std(y_values, scaling_factor=1.3)))
print("The optimal window size is {}".format(window_size))
print("The optimal scaling factor is {}".format(scaling_factor))

# Calculate the filtered_y_values,first_derivative_values,fitted_y_values and plot the data
filtered_y_values = moving_window_filter(y_values, window_size)
first_derivative_values = first_derivative(x_values, filtered_y_values, window_size)
fitted_y_values = reconstruct_fitted_data(x_values, filtered_y_values, optimal_step_locs)

plot_data(x_values, y_values, filtered_y_values, first_derivative_values, fitted_y_values)
plt.show()

# Export original data, filtered data, and fitted data to a CSV file
data_export = pd.DataFrame({
    "X": x_values[:len(filtered_y_values)],
    "Original Data": y_values[:len(filtered_y_values)],
    "Filtered Data": filtered_y_values,
    "Fitted Data": fitted_y_values
})
data_export.to_csv("data_export.csv", index=False)

# Calculate the change sizes based on fitted data
fitted_change_sizes = np.diff(fitted_y_values[np.array(optimal_step_locs)])
# calculate the fitted changing_rates while avoiding divided by zero
x_diffs = np.diff(x_values[optimal_step_locs])
non_zero_diffs_mask = x_diffs != 0
fitted_changing_rates = np.zeros_like(x_diffs, dtype=float)
fitted_changing_rates[non_zero_diffs_mask] = fitted_change_sizes[non_zero_diffs_mask] / x_diffs[non_zero_diffs_mask]
# Calculate the burst durations
burst_durations = np.diff(x_values[optimal_step_locs])

# Create a DataFrame with detected changing-points, changing rates, pausing durations, and change sizes
change_points_export = pd.DataFrame({
    "Changing Point": x_values[optimal_step_locs[:-1]],
    "Changing Rate": fitted_changing_rates,
    "Burst Duration(in time)": burst_durations,
    "Processivity": fitted_change_sizes,
})

# Export change points data to a separate CSV file
change_points_export.to_csv("change_points_export.csv", index=False)

# Plot the distributions of Changing Rates, Burst Durations, and Processivity
fig, axes = plt.subplots(2, 3, figsize=(10, 8))
fig.suptitle("Distributions of Changing Rates, Burst Durations, and Processivity")

# Plot histograms for changing rates, burst durations, and processivity
hist_params = {
    'bins': 30,
    'density': True,
    'alpha': 0.6,
    'color': 'b'
}

# When plotting, only use the non-zero differences
axes[0, 0].hist(fitted_changing_rates[non_zero_diffs_mask], **hist_params)
axes[0, 0].set_title("Histogram of Changing Rates")
axes[0, 0].set_xlabel("Changing Rate")
axes[0, 0].set_ylabel("Density")

axes[0, 1].hist(burst_durations, **hist_params)
axes[0, 1].set_title("Histogram of Burst Durations")
axes[0, 1].set_xlabel("Burst Duration")
axes[0, 1].set_ylabel("Density")

axes[0, 2].hist(fitted_change_sizes, **hist_params)
axes[0, 2].set_title("Histogram of Processivity")
axes[0, 2].set_xlabel("Processivity")
axes[0, 2].set_ylabel("Density")

# Plot box plots for changing rates, burst durations, and processivity
box_data = [fitted_changing_rates, burst_durations, fitted_change_sizes]
box_labels = ["Changing Rate", "Burst Duration", "Processivity"]

axes[1, 1].boxplot(box_data, labels=box_labels)
axes[1, 1].set_title("Box Plots of Changing Rates, Burst Durations, and Processivity")
axes[1, 1].set_ylabel("Value")

# Hide unused subplots
axes[1, 0].axis('off')
axes[1, 2].axis('off')

# Show the plots
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
"""