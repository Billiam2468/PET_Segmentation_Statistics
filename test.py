import os
import shutil
import dicom2nifti as d2n

def rename_files_in_folders(parent_directory):
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
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_dir():
                patient_dir = os.path.join(directory, entry)
                patient_dir = os.path.join(patient_dir, "Unnamed - 0")
                with os.scandir(patient_dir) as scans:
                    for scan in scans:
                        if scan.is_dir():
                            if scan.name[:3] == "PET":
                                #print(scan.name)
                                DICOM = os.path.join(patient_dir, scan)
                                #print(DICOM)
                                segmentName = patient_dir.split("/")[4].split("\\")[0]
                                print(segmentName)
                                segmentName = segmentName + "_CT_SOFT_" + scan.name.split("_")[3]
                                PET_path = "D:/Documents/Alavi Lab/Lymphoma/PET NIFTIs/Interim/"
                                saveOutput = os.path.join(PET_path, segmentName)
                                os.makedirs(saveOutput, exist_ok=True)
                                d2n.convert_directory(DICOM, saveOutput)

directory = "E:/UC Davis DTP Lymphoma/Automated Segmentations/Interim"

def move_files_out_of_folders(directory):
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

add_prefix_to_files(directory, "INTERIM_")