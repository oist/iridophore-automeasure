#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 8 2024
@author: sam, minato
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import csv


def removeSmallBlobs(binary_image, min_area):
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Remove contours with area smaller than min_area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            cv2.drawContours(binary_image, [contour], -1, 0, cv2.FILLED)
    return binary_image


def calculateLongAxis(binary_image):
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Fit ellipse to each contour
    ellipses = []
    for contour in contours:
        if len(contour) >= 5:  # Minimum points required to fit an ellipse
            ellipse = cv2.fitEllipse(contour)
            ellipses.append(ellipse)
    return ellipses


def writeEllipsesToCSV(ellipses, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Center_X', 'Center_Y', 'Major_Axis_Lengths', 'Minor_Axis_Lengths', 'Angle'])
        for ellipse in ellipses:
            center, axes, angle = ellipse
            writer.writerow([center[0], center[1], max(axes), min(axes), angle])


def convert_pixel2micrometer(file_path, pixels_per_micrometer):
    rows = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        rows.append(header)

        for row in reader:
            row[2] = float(row[2]) / pixels_per_micrometer  # Convert Major_Axis_Length
            row[3] = float(row[3]) / pixels_per_micrometer  # Convert Minor_Axis_Length
            rows.append(row)

    # Overwrite the original file with the updated data
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)
    print(f"Modified data written back to {file_path}")



#############################################################################
# Code for iridophore crystal detection and measurements
#############################################################################

# Get the absolute path of the current working directory
current_dir = os.getcwd()
detection_dir = os.path.join(current_dir, 'detection')

# set Type and magnification
type = "Type1"  # set to Type1 or Type2 accordingly
mag = "2000x"  # set if magnification/scales are different across groups (eg. "2000x", "2500x", "5000x")

# set input and output directories
input_dir = os.path.join(detection_dir, type, mag, 'images')
output_dir = os.path.join(detection_dir, type, mag, 'out')
os.makedirs(output_dir, exist_ok=True)

# Set to True if masking noisy images
mask_image = False

# Loop through images in input directory
file_paths = [file for ext in ('*.tif', '*.jpg', '*.png') for file in glob.glob(os.path.join(input_dir, ext))]
for file_path in file_paths:
    file_name, ext = os.path.splitext(os.path.basename(file_path))
    print(f'Processing {file_name}...')

    # Read the image
    image = cv2.imread(file_path, flags=0)

    # Mask for noisy images before cropping
    if mask_image:
        mask_path = os.path.join(input_dir, 'masks')
        mask_file = os.path.join(mask_path, f"{file_name}_mask.png")
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Mask for {file_name} not found. Skipping masking.")
            img_crop = image[0:2048, 0:2048]  # Default cropping
        else:
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            plt.imshow(masked_image, cmap='gray')
            img_crop = masked_image[0:5868, 0:5892]  # Adjust according to image size
    else:
        img_crop = image[0:2048, 0:2048]  # adjust according to image size

    # Apply intensity threshold for mask iridophores
    intensity_threshold = 130 # Adjust as needed
    ir_mask = img_crop.copy()
    ir_mask[ir_mask < intensity_threshold] = 0
    # uncomment to visualize:
    # plt.imshow(ir_mask, cmap='gray')
    # plt.savefig(os.path.join(output_dir, f'{file_name}_int{intensity_threshold}.pdf'), bbox_inches='tight', pad_inches=0)
    # plt.close()

    # Use morphological closing to fill small holes in the mask
    kernel = np.ones((5,5), np.uint8)
    filled_image = cv2.morphologyEx(ir_mask, cv2.MORPH_CLOSE, kernel)

    # Remove blobs under a certain area
    min_blob_area = 300  # Adjust as needed
    filtered_image = removeSmallBlobs(filled_image, min_blob_area)
    # uncomment to visualize:
    # plt.imshow(filtered_image, cmap='gray')
    # plt.savefig(os.path.join(output_dir, f'{file_name}_int{intensity_threshold}_min{min_blob_area}.pdf'), bbox_inches='tight', pad_inches=0)
    # plt.close()

    # Calculate long axis of ellipse fit to remaining blobs
    ellipses = calculateLongAxis(filtered_image)

    # Check scale-bar area
    scale_bar_region = [0, 280, 1925, 2048]  # Adjust as needed; eg. coordinates for the bottom left scale bar area
    # uncomment to visualize:
    # image_with_rect = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for colored rectangle
    # cv2.rectangle(image_with_rect, (scale_bar_region[0], scale_bar_region[2]),
    #                (scale_bar_region[1], scale_bar_region[3]), (0, 255, 0), 2)
    # plt.imshow(image_with_rect)
    # plt.close()

    # Filter out ellipses that intersect with the scale bar region
    filtered_ellipses = []
    for ellipse in ellipses:
        (x, y), (major_axis, minor_axis), angle = ellipse
        if not (scale_bar_region[0] <= x <= scale_bar_region[1] and
                scale_bar_region[2] <= y <= scale_bar_region[3]):
            filtered_ellipses.append(ellipse)

    # Draw filtered ellipses on the cropped image
    output_image = cv2.cvtColor(img_crop, cv2.COLOR_GRAY2BGR)
    for ellipse in filtered_ellipses:
        cv2.ellipse(output_image, ellipse, (0, 255, 0), 2)
    # uncomment to visualize:
    # plt.imshow(output_image)
    # plt.savefig(os.path.join(output_dir, f'{file_name}_contours_int{intensity_threshold}_min{min_blob_area}.pdf'), bbox_inches='tight', pad_inches=0)
    # plt.close()

    # write results to csv
    writeEllipsesToCSV(filtered_ellipses, os.path.join(output_dir, f'{file_name}_ellipses_int{intensity_threshold}_min{min_blob_area}.csv'))


# Convert pixels to micrometers for Major and Minor axis lengths
# Set conversion factors
# example:
# 2000x : 208 pixels = 1 µm
# 2500x : 258 pixels = 1 µm
# 5000x : 291 pixels = 1 µm (1455 pixels = 5 µm)

if mag == "2000x":
    pixels_per_micrometer = 208
elif mag == "2500x":
    pixels_per_micrometer = 258
elif mag == "5000x":
    pixels_per_micrometer = 291
else:
    print("Provide valid magnification (pixels_per_micrometer)")

# Convert all CSV files and return updated file
for file_path in glob.glob(os.path.join(output_dir, '*.csv')):
    convert_pixel2micrometer(file_path, pixels_per_micrometer)
print("Length conversion completed.")
