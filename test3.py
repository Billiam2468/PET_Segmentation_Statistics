
import os
import gc
import nibabel as nib
import numpy as np
import pydicom
from datetime import datetime
import os
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, Dropdown
import IPython
from IPython.display import display, clear_output
from matplotlib.cm import ScalarMappable

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
    print(pet_dicom)
    full_path = os.path.join(pet_dicom, os.listdir(pet_dicom)[0])
    ds = pydicom.dcmread(full_path)
    
    # Get Scan time (Osirix uses SeriesTime, but can change to AcquisitionTime. Series time seems more precise)
    # scantime = parse_time(ds.SeriesTime)
    # # Start Time for the Radiopharmaceutical Injection
    # injection_time = parse_time(ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime)

    time_diff = abs(calculate_time_difference(ds.SeriesTime, ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime))
    print(ds.SeriesTime)
    print(ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime)
    
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

def calculate_suv_statistics_chunked(roi_data, pet_data, chunk_size=(64, 64, 64)):
    unique_rois = np.arange(0, 118)
    suv_stats = {}

    # Determine chunk ranges
    def get_chunks(dim_size, chunk_size):
        return [(i, min(i + chunk_size, dim_size)) for i in range(0, dim_size, chunk_size)]
    
    chunks_x = get_chunks(roi_data.shape[0], chunk_size[0])
    chunks_y = get_chunks(roi_data.shape[1], chunk_size[1])
    chunks_z = get_chunks(roi_data.shape[2], chunk_size[2])

    # Initialize stats dictionary for all ROIs
    for roi in unique_rois:
        if roi != 0:  # Skip the background
            suv_stats[total_segmentator_names[roi]] = {'mean': 0, 'max': -np.inf, 'median': 0, 'num_val': 0}
    
    # Process each chunk
    for (start_x, end_x) in chunks_x:
        for (start_y, end_y) in chunks_y:
            for (start_z, end_z) in chunks_z:
                roi_chunk = roi_data[start_x:end_x, start_y:end_y, start_z:end_z]
                pet_chunk = pet_data[start_x:end_x, start_y:end_y, start_z:end_z]

                for roi in unique_rois:
                    if roi == 0:
                        continue  # Skip the background
                    
                    if roi == 52:
                        roi_mask = (roi_chunk == roi)
                        suv_values = pet_chunk[roi_mask]
                        
                        if suv_values.size > 0:
                            suv_stats[total_segmentator_names[roi]]['mean'] += np.sum(suv_values)
                            suv_stats[total_segmentator_names[roi]]['max'] = max(suv_stats[total_segmentator_names[roi]]['max'], np.max(suv_values))
                            suv_stats[total_segmentator_names[roi]]['num_val'] += suv_values.size

                del roi_chunk, pet_chunk, roi_mask
                gc.collect()

    # Finalize statistics (calculate mean)
    for roi in unique_rois:
        if roi != 0:
            num_val = suv_stats[total_segmentator_names[roi]]['num_val']
            if num_val > 0:
                suv_stats[total_segmentator_names[roi]]['mean'] /= num_val
            else:
                suv_stats[total_segmentator_names[roi]]['mean'] = None
                suv_stats[total_segmentator_names[roi]]['max'] = None
                suv_stats[total_segmentator_names[roi]]['median'] = None

    return suv_stats

def calculate_suv_statistics(roi_data, pet_data):
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
            
            
            suv_stats[total_segmentator_names[int(roi)]] = {
                'mean': mean_suv,
                'max': max_suv,
                'median': median_suv,
                'num_val': num_val
            }
    return suv_stats

nifti_path = "E:/UC Davis DTP Lymphoma/PET NIFTIs/"
home_path = "E:/UC Davis DTP Lymphoma/4_Dicom Images/"

SUV_vals = {}

# with os.scandir(nifti_path) as entries:
#     for entry in entries:
#         split_entry = entry.name.split('_')

#         if split_entry[0] == "INTERIM":
#             dicom_ref = "02_Second_Patch_8Lymphomas_Interim_Scans/" +  split_entry[1] + "_" + split_entry[2] + "_" + split_entry[3] + "/Unnamed - 0/"
#         else:
#             continue
#         # if split_entry[0] == "BASELINE":
#         #     dicom_ref = "1_Baseline/" + split_entry[1] + "_" + split_entry[2] + "_" + split_entry[3] + "/Unnamed - 0/"
#         # else:
#         #     continue
#         # else:
#         #     dicom_ref = "02_Second_Patch_8Lymphomas_Interim_Scans/" + split_entry[1] + "_" + split_entry[2] + "_" + split_entry[3] + "/Unnamed - 0/"
#         dicom_path = os.path.join(home_path, dicom_ref)
#         #print(dicom_path)

#         with os.scandir(dicom_path) as scans:
#             for scanType in scans:
#                 if scanType.is_dir():
#                     #print(split_entry[6][:-7])
#                     #print(scanType.name)
#                     if split_entry[6][:-7] == scanType.name.split('_')[3]:
#                         #print(scanType)
#                         dicom_folder = os.path.join(dicom_path, scanType)
#                         print(entry.name)
#                         SUV_vals[entry.name] = convert_raw_PET_to_SUV(dicom_folder, entry)

pet_directory = "E:/UC Davis DTP Lymphoma/4_Dicom Images/02_Second_Patch_8Lymphomas_Interim_Scans/1470016_Sub0749_Nb/Unnamed - 0/PET_SUV_(20)_120_MIN_302/"
pet_nifti = "E:/UC Davis DTP Lymphoma/PET NIFTIs/INTERIM_1470016_Sub0749_Nb_CT_SOFT_120.nii.gz"
SUV_vals["INTERIM_1470016_Sub0749_Nb_CT_SOFT_120.nii.gz"] = convert_raw_PET_to_SUV(pet_directory, pet_nifti)

segmentation = "E:/UC Davis DTP Lymphoma/Automated Segmentations/INTERIM_1470016_Sub0749_Nb_CT_SOFT_120_MIN_202.nii"

segmentation_img = nib.load(segmentation)
segmentation_data = segmentation_img.get_fdata()

new_shape = (segmentation_data.shape[0], segmentation_data.shape[1], segmentation_data.shape[2])

suv_array = SUV_vals["INTERIM_1470016_Sub0749_Nb_CT_SOFT_120.nii.gz"]
print("shape of PET SUV values is:", suv_array.shape)

memmap_filename = 'testfile.dat'
upscaled_suv = upscale_suv_values_3d(suv_array, new_shape)

suv_array = upscaled_suv

stats = calculate_suv_statistics_chunked(segmentation_data, suv_array)
print(stats['aorta'])