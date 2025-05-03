# Drosophila Biarmipes Tracking Data Structure

## Overview
This dataset contains 25 experimental recordings of Drosophila Biarmipes (strain #84) exposed to 1-hexanol stimulus. The experiments were conducted on two dates (2024.10.10 and 2024.10.14) using a PiVR (Pi-based Virtual Reality) setup.

## Directory Structure
Each experiment follows the naming convention:
```
YYYY.MM.DD_HH-MM-SS_Species_Stimulus
```

## File Contents
Each experiment directory contains:

### Configuration Files
- `experiment_settings.json` and `experiment_settings_updated.json`: Experimental parameters
- `first_frame_data.json`: Initial frame information
- `undistort_matrices.npz`: Camera calibration data

### Raw Data
- `sm_raw.npy`: Raw tracking data
- `sm_thresh.npy`: Thresholded data
- `sm_skeletons.npy`: Skeleton tracking data
- `Background.tiff` and `Undistorted_Background.tiff`: Background images

### Processed Data
- `*_data.csv`: Raw tracking data
- `*_data_level1.csv` and `*_data_level2.csv`: Processed tracking data
- `distance_to_source.csv`: Distance measurements
- `Overview of tracking.png`: Visualization

## Experimental Parameters
- Frame rate: 30 fps
- Recording time: 300 seconds (5 minutes)
- Resolution: 1296x972
- Pixel to mm conversion: ~5.35 pixels/mm
- Animal tracking parameters:
  - Tail speed threshold: 0.532
  - Curvature threshold: 0.624
  - Reorientation threshold: 14.815
  - Average strain length: 3.113

## Tracking Data Structure
The CSV files contain the following columns:
- Frame number and timestamp
- Centroid coordinates (X,Y)
- Head and tail positions
- Midpoint coordinates
- Bounding box information
- Local threshold values
- Stimulation data

## Notable Features
- Consistent stimulus: 1-hexanol at 1:100 concentration
- Multiple data processing levels (raw → level1 → level2)
- Camera calibration and undistortion
- Comprehensive positional tracking (head, tail, centroid, midpoint) 