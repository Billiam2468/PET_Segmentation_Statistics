# This script is a barebones script that is used once the user has all the PET NIFTI files (after conversion) and segmentation files.
# It will convert the segmentations and corresponding PET NIFTIs to generate statistics based on whatever organ/structure the user is interested in

# Segmentation Map
total_segmentator_names = [
    "background",
    "spleen",
    "kidney_right",
    "kidney_left",
    "gallbladder",
    "liver",
    "stomach",
    "pancreas",
    "adrenal_gland_right",
    "adrenal_gland_left",
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
    "esophagus",
    "trachea",
    "thyroid_gland",
    "small_bowel",
    "duodenum",
    "colon",
    "urinary_bladder",
    "prostate",
    "kidney_cyst_left",
    "kidney_cyst_right",
    "sacrum",
    "vertebrae_S1",
    "vertebrae_L5",
    "vertebrae_L4",
    "vertebrae_L3",
    "vertebrae_L2",
    "vertebrae_L1",
    "vertebrae_T12",
    "vertebrae_T11",
    "vertebrae_T10",
    "vertebrae_T9",
    "vertebrae_T8",
    "vertebrae_T7",
    "vertebrae_T6",
    "vertebrae_T5",
    "vertebrae_T4",
    "vertebrae_T3",
    "vertebrae_T2",
    "vertebrae_T1",
    "vertebrae_C7",
    "vertebrae_C6",
    "vertebrae_C5",
    "vertebrae_C4",
    "vertebrae_C3",
    "vertebrae_C2",
    "vertebrae_C1",
    "heart",
    "aorta",
    "pulmonary_vein",
    "brachiocephalic_trunk",
    "subclavian_artery_right",
    "subclavian_artery_left",
    "common_carotid_artery_right",
    "common_carotid_artery_left",
    "brachiocephalic_vein_left",
    "brachiocephalic_vein_right",
    "atrial_appendage_left",
    "superior_vena_cava",
    "inferior_vena_cava",
    "portal_vein_and_splenic_vein",
    "iliac_artery_left",
    "iliac_artery_right",
    "iliac_vena_left",
    "iliac_vena_right",
    "humerus_left",
    "humerus_right",
    "scapula_left",
    "scapula_right",
    "clavicula_left",
    "clavicula_right",
    "femur_left",
    "femur_right",
    "hip_left",
    "hip_right",
    "spinal_cord",
    "gluteus_maximus_left",
    "gluteus_maximus_right",
    "gluteus_medius_left",
    "gluteus_medius_right",
    "gluteus_minimus_left",
    "gluteus_minimus_right",
    "autochthon_left",
    "autochthon_right",
    "iliopsoas_left",
    "iliopsoas_right",
    "brain",
    "skull",
    "rib_left_1",
    "rib_left_2",
    "rib_left_3",
    "rib_left_4",
    "rib_left_5",
    "rib_left_6",
    "rib_left_7",
    "rib_left_8",
    "rib_left_9",
    "rib_left_10",
    "rib_left_11",
    "rib_left_12",
    "rib_right_1",
    "rib_right_2",
    "rib_right_3",
    "rib_right_4",
    "rib_right_5",
    "rib_right_6",
    "rib_right_7",
    "rib_right_8",
    "rib_right_9",
    "rib_right_10",
    "rib_right_11",
    "rib_right_12",
    "sternum",
    "costal_cartilages"
]

joint_names = [
    "Background",
    "Left toe-metatarsal",
    "Left metatarsal-tarsal",
    "Left tarsal-tibial",
    "Left tibial-femoral",
    "Right toe-metatarsal",
    "Right metatarsal-tarsal",
    "Right tarsal-tibial",
    "Right tibial-femoral",
    "Left finger-metacarpal",
    "Left metacarpal-carpal",
    "Left carpal-radial",
    "Left radial-humeral",
    "Left humeral-scapular",
    "Right finger-metacarpal",
    "Right metacarpal-carpal",
    "Right carpal-radial",
    "Right radial-humeral",
    "Right humeral-scapular"
]

import nibabel as nib
import numpy as np
import pydicom
from datetime import datetime
import os
import sys
import gc
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, Dropdown
import IPython
from IPython.display import display, clear_output
from matplotlib.cm import ScalarMappable


# Upscale SUV values (256,256) to segmentation mask resolution (512,512)
def upscale_suv_values_3d(suv_values, new_shape):
    # Get the shape of the original SUV values array
    original_shape = suv_values.shape
    
    # Create arrays of coordinates for the original and new SUV values arrays
    x = np.linspace(0, 1, original_shape[0])
    y = np.linspace(0, 1, original_shape[1])
    z = np.linspace(0, 1, original_shape[2])
    
    new_x = np.linspace(0, 1, new_shape[0])
    new_y = np.linspace(0, 1, new_shape[1])
    new_z = np.linspace(0, 1, new_shape[2])
    
    # Create meshgrids of the coordinates
    x_mesh, y_mesh, z_mesh = np.meshgrid(x, y, z, indexing='ij')
    new_x_mesh, new_y_mesh, new_z_mesh = np.meshgrid(new_x, new_y, new_z, indexing='ij')
    
    # Interpolate the SUV values to the new coordinates in each dimension
    suv_interpolated = np.zeros(new_shape).astype(np.float32)
    for i in range(original_shape[2]):
        suv_interpolated[:, :, i] = interpolate.interpn((x, y), suv_values[:, :, i], (new_x_mesh[:, :, i], new_y_mesh[:, :, i]), method='linear', bounds_error=False, fill_value=None)
    
    return suv_interpolated.astype(np.float32)

def load_nifti_file(filepath):
    """Load a NIfTI file and return the data array."""
    nifti_img = nib.load(filepath)
    data = nifti_img.get_fdata()
    return data
    
def calculate_time_difference(scan_time_str, injection_time_str):
    # Define the correct time format
    time_format_with_microseconds = "%H%M%S.%f"
    time_format_without_microseconds = "%H%M%S"

    # Parse the time strings with and without microseconds
    try:
        scan_time = datetime.strptime(scan_time_str, time_format_with_microseconds)
    except ValueError:
        scan_time = datetime.strptime(scan_time_str, time_format_without_microseconds)

    try:
        injection_time = datetime.strptime(injection_time_str, time_format_with_microseconds)
    except ValueError:
        injection_time = datetime.strptime(injection_time_str, time_format_without_microseconds)

    # Remove the fractional seconds by setting microseconds to zero
    scan_time = scan_time.replace(microsecond=0)
    
    # Subtract the two datetime objects
    time_difference = scan_time - injection_time

    # Get the total difference in seconds
    total_seconds = time_difference.total_seconds()

    return total_seconds

def convert_raw_PET_to_SUV(pet_dicom, pet_nifti):
    PET_data = load_nifti_file(pet_nifti)
    #print(pet_dicom)
    full_path = os.path.join(pet_dicom, os.listdir(pet_dicom)[0])
    ds = pydicom.dcmread(full_path)
    
    # Get Scan time (Osirix uses SeriesTime, but can change to AcquisitionTime. Series time seems more precise)
    # scantime = parse_time(ds.SeriesTime)
    # # Start Time for the Radiopharmaceutical Injection
    # injection_time = parse_time(ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime)
    print("Series time:")
    print(ds.SeriesTime)

    print("Injection time:")
    print(ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime)

    # Hard coded for exception due to bad metadata here for this DICOM. Said injection time was 105900000000. Should probably be 105900.00
    if pet_dicom == "E:/Psoriasis/VIP-S/1002017-199-A/study/PET_SLICES_IR_MAC":
        time_diff = abs(calculate_time_difference(ds.SeriesTime, '105900.00'))
    else:
        time_diff = abs(calculate_time_difference(ds.SeriesTime, ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime))

    
    # Half Life for Radionuclide # seconds
    half_life = float(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife) 
    # Total dose injected for Radionuclide
    injected_dose = float(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
    
    # Calculate dose decay correction factor
    decay_correction_factor = np.exp(-np.log(2)*((time_diff)/half_life))

    print("half_life: ",half_life)
    #print("scan_time: ", scantime)
    #print("injection time: ", injection_time)
    print("minus time", (time_diff))
    
    patient_weight = ds.PatientWeight * 1000
    SUV_factor = (patient_weight) / (injected_dose * decay_correction_factor)


    
    print("decay_correction factor ", decay_correction_factor) 
    print("SUV factor is:", SUV_factor)
    return(PET_data * SUV_factor).astype(np.float32)

def calculate_suv_statistics(roi_data, pet_data, reference):
    """DEPRECATED DUE TO MEMORY ISSUES"""
    """Calculate SUV statistics for each unique ROI."""
    #print(roi_data)
    unique_rois = np.arange(0, 118)
    suv_stats = {}
    
    for roi in unique_rois:
        print("Working on roi: ", total_segmentator_names[roi])
        print(roi)
        if roi == 0:
            continue  # Skip the background

        # Remove this if statement when doing batch statistics. Only here to save time and only calculate lung statistics
        if roi == 52:
            print("shapes:")
            print(roi_data.shape)
            print(pet_data.shape)
            roi_mask = roi_data == roi
            suv_values = pet_data[roi_mask]
            if len(suv_values) == 0:
                mean_suv = None
                max_suv = None
                median_suv = None
                num_val = None
            else:
                mean_suv = np.mean(suv_values)
                max_suv = np.max(suv_values)
                median_suv = np.median(suv_values)
                num_val = len(suv_values)
            
            
            suv_stats[reference[int(roi)]] = {
                'mean': mean_suv,
                'max': max_suv,
                'median': median_suv,
                'num_val': num_val
            }
    return suv_stats

def calculate_suv_statistics_chunked(roi_data, pet_data, reference, chunk_size=(64, 64, 64)):
    unique_rois = np.arange(0, len(reference))
    suv_stats = {}

    # Determine chunk ranges
    def get_chunks(dim_size, chunk_size):
        return [(i, min(i + chunk_size, dim_size)) for i in range(0, dim_size, chunk_size)]
    
    chunks_x = get_chunks(roi_data.shape[0], chunk_size[0])
    chunks_y = get_chunks(roi_data.shape[1], chunk_size[1])
    chunks_z = get_chunks(roi_data.shape[2], chunk_size[2])

    # Initialize stats dictionary for all ROIs
    for roi in unique_rois:
        if roi != 0:  # Skip the background (don't skip background because 0 is left toe metatarsal)
            suv_stats[reference[roi]] = {'mean': 0, 'max': -np.inf, 'median': 0, 'num_val': 0}
    
    # Process each chunk
    for (start_x, end_x) in chunks_x:
        for (start_y, end_y) in chunks_y:
            for (start_z, end_z) in chunks_z:
                roi_chunk = roi_data[start_x:end_x, start_y:end_y, start_z:end_z]
                pet_chunk = pet_data[start_x:end_x, start_y:end_y, start_z:end_z]

                for roi in unique_rois:
                    if roi == 0:
                        continue  # Skip the background
                    roi_mask = (roi_chunk == roi)
                    suv_values = pet_chunk[roi_mask]
                    
                    if suv_values.size > 0:
                        suv_stats[reference[roi]]['mean'] += np.sum(suv_values)
                        suv_stats[reference[roi]]['max'] = max(suv_stats[reference[roi]]['max'], np.max(suv_values))
                        suv_stats[reference[roi]]['num_val'] += suv_values.size

                del roi_chunk, pet_chunk, roi_mask
                gc.collect()

    # Finalize statistics (calculate mean)
    for roi in unique_rois:
        if roi != 0:
            num_val = suv_stats[reference[roi]]['num_val']
            if num_val > 0:
                suv_stats[reference[roi]]['mean'] /= num_val
            else:
                suv_stats[reference[roi]]['mean'] = None
                suv_stats[reference[roi]]['max'] = None
                suv_stats[reference[roi]]['median'] = None

    return suv_stats

# Function to parse time with or without fractional seconds
def parse_time(time_str):
    try:
        # Try parsing with fractional seconds
        return datetime.datetime.strptime(time_str, '%H%M%S.%f')
    except ValueError:
        # Fallback to parsing without fractional seconds
        return datetime.datetime.strptime(time_str, '%H%M%S')

# Helper function that takes in an organ and SUV values of a single patient and extracts the correct statitistics. Adapted to extract total lung weighted avg across lobes (lung) as well as all vertebrae and rib
def extractPatientStats(suvs, organ, stat):
    if organ == "lung":
        val1 = suvs['lung_upper_lobe_left'][stat]
        val1num = suvs['lung_upper_lobe_left']['num_val']
        val2 = suvs['lung_lower_lobe_left'][stat]
        val2num = suvs['lung_lower_lobe_left']['num_val']
        val3 = suvs['lung_upper_lobe_right'][stat]
        val3num = suvs['lung_upper_lobe_right']['num_val']
        val4 = suvs['lung_middle_lobe_right'][stat]
        val4num = suvs['lung_middle_lobe_right']['num_val']
        val5 = suvs['lung_lower_lobe_right'][stat]
        val5num = suvs['lung_lower_lobe_right']['num_val']
        avg = (val1*val1num + val2*val2num + val3*val3num + val4*val4num + val5*val5num)/(val1num+val2num+val3num+val4num+val5num)
        return avg
    if (organ == "vertebrae") or (organ == "rib"):
        rawVals = []
        numVals = []
        for name in total_segmentator_names:
            if name.startswith(organ):
                val = suvs[name][stat]
                rawVals.append(val)
                numVal = suvs[name]['num_val']
                numVals.append(numVal)
        numerator = 0
        denominator = 0
        for idx,val in enumerate(rawVals):
            numerator = numerator + (rawVals[idx] * numVals[idx])
            denominator = denominator + numVals[idx]
        #print("length of vertebrae ", len(numVals))
        return (numerator/denominator)
    else:
        return suvs[organ][stat]

def upscale_suv_values_3d_memmap(suv_values, new_shape, memmap_filename):
    original_shape = suv_values.shape
    x = np.linspace(0, 1, original_shape[0])
    y = np.linspace(0, 1, original_shape[1])
    z = np.linspace(0, 1, original_shape[2])
    
    new_x = np.linspace(0, 1, new_shape[0])
    new_y = np.linspace(0, 1, new_shape[1])
    new_z = np.linspace(0, 1, new_shape[2])
    
    # Create a memory-mapped file for the upscaled SUV values
    suv_interpolated = np.memmap(memmap_filename, dtype=np.float32, mode='w+', shape=new_shape)
    
    for i in range(original_shape[2]):
        suv_interpolated[:, :, i] = interpolate.interpn(
            (x, y), suv_values[:, :, i],
            (new_x[:, None], new_y[:, None]),  # Adjusted for 2D interpolation
            method='linear', bounds_error=False, fill_value=None
        )
    
    suv_interpolated.flush()  # Ensure changes are written to disk
    return np.array(suv_interpolated)  # Convert back to a regular array if needed

# # Usage
#memmap_filename = 'upscaled_suv_values.dat'
#upscaled_suv_values = upscale_suv_values_3d_memmap(suv_array, new_shape, memmap_filename)



nifti_path = "E:/Psoriasis/PET NIFTIs/Site 1012/"
home_path = "E:/Psoriasis/VIP-S/"

SUV_vals = {}

with os.scandir(nifti_path) as entries:
    for entry in entries:
        if entry.is_file():
            # Create reference to DICOM each NIFTI comes from
            split_entry = entry.name.split('_')
            fileName = '_'.join(split_entry[1:])
            fileName = fileName[:-7]
            dicom_ref = split_entry[0] + "/study/" + fileName
            dicom_path = os.path.join(home_path, dicom_ref)
            print(dicom_path)

            SUV_vals[split_entry[0]] = convert_raw_PET_to_SUV(dicom_path, entry)

        # with os.scandir(dicom_path) as scans:
        #     for scanType in scans:
        #         if scanType.is_dir():
        #             #print(split_entry[6][:-7])
        #             #print(scanType.name)
        #             if split_entry[6][:-7] == scanType.name.split('_')[3]:
        #                 #print(scanType)
        #                 dicom_folder = os.path.join(dicom_path, scanType)
        #                 print(entry.name)
        #                 SUV_vals[entry.name] = convert_raw_PET_to_SUV(dicom_folder, entry)

import nrrd
import gc

segmentation_dir = "E:/Psoriasis/Peripheral_AI_Segmentations/Site 1012/Joint Segmentations/"

stats = {}

with os.scandir(segmentation_dir) as segmentations:
    for segmentation in segmentations:
        if segmentation.is_file():
            print(segmentation.name)
            pet_name = segmentation.name.split('_')[4]
            seg_dir = os.path.join(segmentation_dir, segmentation)

            #If files are nifti use below:
            # segmentation_img = nib.load(seg_dir)
            # segmentation_data = segmentation_img.get_fdata()

            #If files are nrrd use below:
            segmentation_data, header_data = nrrd.read(seg_dir)

            # Upscale the PET SUV values to match the shape of the segmentation
            new_shape = (segmentation_data.shape[0], segmentation_data.shape[1], segmentation_data.shape[2])
            print("Shape of segmentation is: ", new_shape)

            print("Shape of PET suv_vals is: ", SUV_vals[pet_name].shape)

            # upscaled_suv_values = upscale_suv_values_3d(suv_array, new_shape)
            upscaled_suv_values = upscale_suv_values_3d(SUV_vals[pet_name], new_shape)

            chunk_size = 50  # Adjust the chunk size based on your available memory
            stats[pet_name] = calculate_suv_statistics_chunked(segmentation_data, upscaled_suv_values, joint_names)

            del segmentation_data, upscaled_suv_values
            gc.collect()


np.save('site_1012_joint_suv_stats.npy', stats)