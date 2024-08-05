import os
import shutil
import dicom2nifti as d2n

def rename_files_in_folders(parent_directory):
    """Function that that names each file inside a directory of folders the name of the folder. Needed when converting PETs to NIFTIs"""
    # Iterate over all the items in the parent directory
    for folder_name in os.listdir(parent_directory):
        folder_path = os.path.join(parent_directory, folder_name)
        
        # Check if the item is a directory
        if os.path.isdir(folder_path):
            # List files in the directory
            files = os.listdir(folder_path)
            
            # Ensure there is only one file in the folder
            if len(files) == 1:
                print(folder_path)
                file_path = os.path.join(folder_path, files[0])
                new_file_path = os.path.join(folder_path, folder_name)
                
                # Get the file extension of the current file
                file_extension = os.path.splitext(files[0])[1]
                
                # Rename the file to the name of the folder, maintaining the original file extension
                os.rename(file_path, new_file_path + ".nii" + file_extension)
                print(f'Renamed {file_path} to {new_file_path + file_extension}')
            else:
                print(f"Warning: Folder '{folder_path}' does not contain exactly one file.")

# Example usage
parent_directory = f"E:/UC Davis COVID Study/Segmentations and PET NIFTIs/Edited Organs/Brain/"
# rename_files_in_folders(parent_directory)

# Extracts the segmentation files out of the moose file directory format
def extract_segmentation(parent_directory):

    segment_files = []

    for folder_name in os.listdir(parent_directory):
        folder_path = os.path.join(parent_directory, folder_name)

        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            for file in files:
                segment_folder = os.path.join(folder_path, file)
                if os.path.isdir(segment_folder):
                    true_segment_folder = os.path.join(segment_folder, "segmentations")
                    gz_files = [f for f in os.listdir(true_segment_folder) if f.endswith('.gz')]
                    #print(gz_files[0])
                    seg_file = os.path.join(true_segment_folder, gz_files[0])
                    segment_files.append(seg_file)
    
    destination_directory = "E:\Psoriasis\AI_Segmentations\\"

    for file_path in segment_files:
        if os.path.isfile(file_path):
            shutil.move(file_path, destination_directory)
            print(f"Moved {file_path} to {destination_directory}")
        else:
            print(f"{file_path} does not exist or is not a file.")

    #print(segment_files)

def rename_files(directory):
    # String to remove from the filenames
    string_to_remove = 'EditedBrain'
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Construct full file path
        old_filepath = os.path.join(directory, filename)
        
        # Skip if it's not a file
        if not os.path.isfile(old_filepath):
            continue
        
        # New filename after removing the string
        new_filename = filename.replace(string_to_remove, '')
        new_filepath = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_filepath, new_filepath)
        print(f'Renamed: {old_filepath} -> {new_filepath}')

def convert_pet_to_nifti(directory):
    """Remember to include the reference to the specific DICOM folder used as that reference is needed when extracting statistics"""
    PET_path = "E:/Psoriasis/PET NIFTIs/Site 1005"
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_dir() and entry.name[:4] == "1012":
                patient_dir = os.path.join(directory, entry)
                patient_dir = os.path.join(patient_dir, "study")


                # if entry.name == "1002014-668-A":
                #     DICOM = os.path.join(patient_dir, "STL_pet_slices_IR_MAC")
                #     saveOutput = os.path.join(PET_path, entry.name + "_STL_pet_slices_IR_MAC")
                #     os.makedirs(saveOutput, exist_ok=True)
                #     d2n.convert_directory(DICOM, saveOutput)
                # elif entry.name == "1002001-850-A":
                #     DICOM = os.path.join(patient_dir, "PROSP_AC_2D_WB")
                #     saveOutput = os.path.join(PET_path, entry.name + "_PROSP_AC_2D_WB")
                #     os.makedirs(saveOutput, exist_ok=True)
                #     d2n.convert_directory(DICOM, saveOutput)
                # elif entry.name == "1002001-850-B":
                #     DICOM = os.path.join(patient_dir, "PROSP_AC_2D_WB")
                #     saveOutput = os.path.join(PET_path, entry.name + "_PROSP_AC_2D_WB")
                #     os.makedirs(saveOutput, exist_ok=True)
                #     d2n.convert_directory(DICOM, saveOutput)


                # if entry.name == "1003010-269-A":
                #     DICOM = os.path.join(patient_dir, "Corrected_for_SUV_Calculation")
                #     saveOutput = os.path.join(PET_path, entry.name + "_Corrected_for_SUV_Calculation")
                #     os.makedirs(saveOutput, exist_ok=True)
                #     d2n.convert_directory(DICOM, saveOutput)
                # else:
                with os.scandir(patient_dir) as scans:
                    for scan in scans:
                        if scan.is_dir():
                            if scan.name.endswith("(AC)"):
                                print(entry.name)
                                print(scan.name)
                                DICOM = os.path.join(patient_dir, scan)
                                saveOutput = os.path.join(PET_path, entry.name + "_" + scan.name)
                                os.makedirs(saveOutput, exist_ok=True)
                                d2n.convert_directory(DICOM, saveOutput)

def move_files_out_of_folders(directory):
    """Takes each file inside each PET NIFTI folder and moves it out of the folder"""
    # Iterate over each item in the provided directory
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        
        # Check if the current item is a directory
        if os.path.isdir(folder_path):
            # Get the list of files in the current folder
            files = os.listdir(folder_path)
            
            # Move each file out of the folder to the parent directory
            for file_name in files:
                old_file_path = os.path.join(folder_path, file_name)
                new_file_path = os.path.join(directory, file_name)
                
                # Move the file
                shutil.move(old_file_path, new_file_path)
                print(f"Moved '{old_file_path}' to '{new_file_path}'")
            
            # Optionally, remove the now-empty folder
            os.rmdir(folder_path)
            print(f"Removed empty folder '{folder_path}'")

def add_prefix_to_files(directory, prefix):
    # Iterate over each item in the provided directory
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        
        # Check if the current item is a file
        if os.path.isfile(file_path):
            # Create new file name with prefix
            new_file_name = prefix + file_name
            new_file_path = os.path.join(directory, new_file_name)
            
            # Rename the file
            os.rename(file_path, new_file_path)
            print(f"Renamed '{file_path}' to '{new_file_path}'")

directory = "E:/Psoriasis/VIP-S/"
#convert_pet_to_nifti(directory)
#rename_files_in_folders("E:/Psoriasis/PET NIFTIs/Site 1012")
#move_files_out_of_folders("E:/Psoriasis/PET NIFTIs/Site 1012")