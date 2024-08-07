import nibabel as nib
import numpy as np
import nrrd
from scipy import interpolate

def load_nifti_file(filepath):
    """Load a NIfTI file and return the data array."""
    nifti_img = nib.load(filepath)
    data = nifti_img.get_fdata()
    return data

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

data = load_nifti_file("E:/Psoriasis/PET NIFTIs/Site 1002/1002001-086-A_PET_SLICES_IR_MAC.nii.gz")

segmentation_data, header_data = nrrd.read("E:/Psoriasis/Peripheral_AI_Segmentations/Site 1002/Joint Segmentations/joints_CT_Peripheral-Bones_CT_1002001-086-A_CTAC_HEAD_IN_TRANS_TOMO_0000.nii.gz.nrrd.nrrd")

print(data.shape)
print(segmentation_data.shape)

pet_data = upscale_suv_values_3d(data, segmentation_data.shape)

# print("new")
# print(data.shape)

count = np.count_nonzero(segmentation_data == 9)
print(count)


result = pet_data[segmentation_data == 9]
result_list = result.tolist()
print(result_list)

print("max is: ", np.max(result_list))
print("mean is: ", np.mean(result_list))