import os
import subprocess
home_dir = "D:/Documents/Scans/Test COVID Study Files/"
counter = 0
home_dir = "E:/UC Davis COVID Study/"
with os.scandir(home_dir) as entries:
    for entry in entries:
        if entry.is_dir():
            #print(entry.name)
            
            group_dir = os.path.join(home_dir, entry)
            with os.scandir(group_dir) as patients:
                for patient in patients:
                    if patient.is_dir():
                        #print(patient)

                        patient_dir = os.path.join(group_dir, patient)
                        with os.scandir(patient_dir) as scan_times:
                            for scan_time in scan_times:
                                if scan_time.is_dir():
                                    #print(scan_time.name)

                                    scan_dir = os.path.join(patient_dir, scan_time)
                                    with os.scandir(scan_dir) as scans:
                                        for scan in scans:
                                            if scan.is_dir():
                                                if scan.name[:2] == "CT":
                                                    print(os.path.join(scan_dir, scan.name))
                                                    counter = counter + 1
print(counter)
command = 'TotalSegmentator -i "E:/UC Davis COVID Study/Healthy Controls/1341792-Sub023-A01-CJ/20110227/CT_SOFT_BS_512x512" -o "D:/Documents/Scans/notebooktest" --ml'

# Execute the bash command
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Wait for the process to finish and get the output
stdout, stderr = process.communicate()

# Print the output
if stdout:
    print("Output:")
    print(stdout.decode())
if stderr:
    print("Error:")
    print(stderr.decode())