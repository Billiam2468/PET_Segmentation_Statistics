import os
import shutil

def rename_files_in_folders(parent_directory):
    # Iterate over all the items in the parent directory
    for folder_name in os.listdir(parent_directory):
        folder_path = os.path.join(parent_directory, folder_name)
        
        # Check if the item is a directory
        if os.path.isdir(folder_path):
            # List files in the directory
            files = os.listdir(folder_path)
            
            # Ensure there is only one file in the folder
            if len(files) != 1:
                print(folder_path)
                # file_path = os.path.join(folder_path, files[0])
                # new_file_path = os.path.join(folder_path, "CT_" + folder_name)
                
                # # Get the file extension of the current file
                # file_extension = os.path.splitext(files[0])[1]
                
                # # Rename the file to the name of the folder, maintaining the original file extension
                # os.rename(file_path, new_file_path + ".nii" + file_extension)
                # print(f'Renamed {file_path} to {new_file_path + file_extension}')
            # else:
            #     print(f"Warning: Folder '{folder_path}' does not contain exactly one file.")

# Example usage
parent_directory = f'E:\Psoriasis\Finished Segmentations\\'
# rename_files_in_folders(parent_directory)

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
extract_segmentation(parent_directory)