#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun04 15:22:42 2024

@author: sam, minato
"""
import os
import numpy as np
import glob
import csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import circstd, circvar, circmean, ranksums


################################
# Function to load data
def load_data_and_circvars(directory):
    # Load and list Major_Axis_Length, Minor_Axis_Length, and Angles (radians) from all CSV files in the directory.
    # Also calculate and list individual circular variances of angles within each file.
    major_axes, minor_axes, angles = [], [], []
    circ_vars = []

    # Get all CSV files with 'ellipses' in their name
    file_paths = glob.glob(os.path.join(directory, '**', '*ellipses*.csv'), recursive=True)
    for file_path in file_paths:
        df = pd.read_csv(file_path)

        # Collect major, minor axes and angles
        major_axes.extend(df['Major_Axis_Length'].values)
        minor_axes.extend(df['Minor_Axis_Length'].values)
        angle_values = df['Angle'].values
        angles.extend(angle_values)

        # Convert angles to radians for circular variance calculation
        angles_rad = np.radians(angle_values)
        circ_var = circvar(angles_rad, high=np.pi, low=-np.pi)
        circ_vars.append(circ_var)

    return major_axes, minor_axes, np.radians(angles), circ_vars


################################

# Functions to perform Shapiro-Wilk and Levene's tests and store the results
# 1. Shapiro-Wilk Test for normality
def test_normality(data, variable_name):
    stat, p_value = stats.shapiro(data)
    result = f"Shapiro-Wilk test for {variable_name}: W-statistic = {stat}, p-value = {p_value}"
    if p_value > 0.05:
        result += f"\n{variable_name} is likely normally distributed (p > 0.05)\n"
    else:
        result += f"\n{variable_name} is not normally distributed (p <= 0.05)\n"
    return result

# 2. Levene's Test for equal variances
def test_equal_variances(data1, data2, variable_name):
    stat, p_value = stats.levene(data1, data2)
    result = f"Levene's test for {variable_name}: W-statistic = {stat}, p-value = {p_value}"
    if p_value > 0.05:
        result += f"\n{variable_name} likely has equal variances (p > 0.05)\n"
    else:
        result += f"\n{variable_name} likely does not have equal variances (p <= 0.05)\n"
    return result


################################

# Function to create box plot with Plotly
def write_statistical_results(stat_results, output_path):
    # Write the statistical test results to a text file.
    with open(output_path, 'w') as file:
        for result in stat_results:
            file.write(result + '\n')


def write_statistical_results_to_csv(stat_results, output_path):
    # Create a DataFrame from the statistical results
    df = pd.DataFrame(stat_results)
    df.to_csv(output_path, index=False)

################################

# Function to create box plot with Plotly
def plot_boxplot_plotly(data1, data2, labels, title, ylabel, output_path,
                        show_points=True, width=1000, height=1000):
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Box(y=data1, name=labels[0], marker_color='blue', boxmean=True, boxpoints='all' if show_points else None))
    fig.add_trace(
        go.Box(y=data2, name=labels[1], marker_color='purple', boxmean=True, boxpoints='all' if show_points else None))

    # Perform statistical test
    stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
    test_name = "Mann-Whitney U"

    # Add statistical annotation
    annotation_text = f'{test_name} p-value = {p_value:.3e}'
    y_max = max(max(data1), max(data2))
    fig.add_annotation(dict(font=dict(color='#363F56', size=12),
                            x=0.5,
                            y=y_max + 0.1,
                            showarrow=False,
                            text=annotation_text,
                            xanchor='center',
                            yanchor='bottom'))

    # Update layout
    fig.update_layout(title=title,
                      yaxis_title=ylabel,
                      xaxis_title="Cell Type",
                      showlegend=False,
                      plot_bgcolor='white',  # White background panel
                      margin=dict(l=120, r=120, t=80, b=80)
                      )
    # Save
    fig.write_image(output_path, width=width, height=height)




#########################################################
# Set directories and load data
#########################################################

# Get the absolute path of the current working directory
current_dir = os.getcwd()
detection_dir = os.path.join(current_dir, 'detection')
stats_dir = os.path.join(current_dir, 'stats')

type1_dir = os.path.join(detection_dir, 'Type1')
type2_dir = os.path.join(detection_dir, 'Type2')


# Load data for both cell types
major_axes_type1, minor_axes_type1, angles_rad_type1, circ_vars_type1 = load_data_and_circvars(type1_dir)
major_axes_type2, minor_axes_type2, angles_rad_type2, circ_vars_type2 = load_data_and_circvars(type2_dir)


#########################################################
# Preliminary check for normality and equal variance
#########################################################
def run_pretests():
    stat_results = []
    # Test normality for each variable
    stat_results.append(test_normality(major_axes_type1, 'Major Axis Length (Type 1)'))
    stat_results.append(test_normality(major_axes_type2, 'Major Axis Length (Type 2)'))
    stat_results.append(test_normality(minor_axes_type1, 'Minor Axis Length (Type 1)'))
    stat_results.append(test_normality(minor_axes_type2, 'Minor Axis Length (Type 2)'))
    stat_results.append(test_normality(circ_vars_type1, 'Circular Variance (Type 1)'))
    stat_results.append(test_normality(circ_vars_type2, 'Circular Variance (Type 2)'))

    # Test equal variances for each variable
    stat_results.append(test_equal_variances(major_axes_type1, major_axes_type2, 'Major Axis Length'))
    stat_results.append(test_equal_variances(minor_axes_type1, minor_axes_type2, 'Minor Axis Length'))
    stat_results.append(test_equal_variances(circ_vars_type1, circ_vars_type2, 'Circular variance'))

    return stat_results

stat_results = run_pretests()
pretest_filepath = os.path.join(stats_dir, "pretests_norm_var.txt")
write_statistical_results(stat_results, pretest_filepath)



#########################################################
# Perform statistical tests and write results to text file
#########################################################

variables = {
    "Major Axis Length (µm)": (major_axes_type1, major_axes_type2),
    "Minor Axis Length (µm)": (minor_axes_type1, minor_axes_type2),
    "Circular Variance": (circ_vars_type1, circ_vars_type2)
}

stat_results = []
for var_name, (data1, data2) in variables.items():
    # Calculate mean, median, and standard deviation
    stats_results = {
        "Variable": var_name,
        "Type 1 Mean": np.mean(data1),
        "Type 2 Mean": np.mean(data2),
        "Type 1 Median": np.median(data1),
        "Type 2 Median": np.median(data2),
        "Type 1 Std Dev": np.std(data1),
        "Type 2 Std Dev": np.std(data2),
    }
    # Perform Mann-Whitney U test
    u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
    stats_results.update({"U-Statistic": u_stat, "p-value": p_value})

    # Append results to the list
    stat_results.append(stats_results)

# Output to CSV
output_path = os.path.join(stats_dir, "stat_test.csv")
write_statistical_results_to_csv(stat_results, output_path)
print(f"Statistical results have been written to {output_path}")



#########################################################
# Plot box plots
#########################################################

plot_boxplot_plotly(major_axes_type1, major_axes_type2,
                    labels=['Type 1', 'Type 2'], title='Major Axis Length Comparison',
                    ylabel='Major Axis Length (µm)',
                    show_points=False,
                    output_path=os.path.join(stats_dir, 'boxplot_major_axis_length.png'))

plot_boxplot_plotly(minor_axes_type1, minor_axes_type2,
                    labels=['Type 1', 'Type 2'], title='Minor Axis Length Comparison',
                    ylabel='Minor Axis Length (µm)',
                    show_points=False,
                    output_path=os.path.join(stats_dir, 'boxplot_minor_axis_length.png'))

plot_boxplot_plotly(circ_vars_type1, circ_vars_type2,
                    labels=['Type 1', 'Type 2'], title='Circular Variance Comparison',
                    ylabel='Angle Circular Variance',
                    output_path=os.path.join(stats_dir, 'boxplot_circular_variance.png'))

