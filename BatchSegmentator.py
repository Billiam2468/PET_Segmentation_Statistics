import os
import subprocess
import dicom2nifti as d2n

def runBash(command):
    # Execute the bash command
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for the process to finish and get the output
    stdout, stderr = process.communicate()

    # Print the output
    if stdout:
        print("Output:")
        print(stdout.decode())

def segment(DICOM, segmentName, task):
    # output = DICOM.split('\\')
    # output_str = output[3] + "_" + output[5]
    # os.makedirs("E:/Psoriasis/AI Segmentations/" + output_str)
    # d2n.convert_directory(DICOM, "E:/Psoriasis/AI Segmentations/" + output_str, compression=True)
    #command = f'TotalSegmentator -i "{DICOM}" -o "./Segmentations/Lymphoma/Interim/{segmentName}" --ml --force_split -ta {task}'
    command = f'moosez -d {DICOM} -m {task}'
    runBash(command)



#home_dir = "/media/billiam/T7 Shield/UC Davis COVID Study/"
home_dir = "E:\Psoriasis\AI_Segmentations\\"
# home_dir = "E:/UC Davis DTP Lymphoma/4_Dicom Images/02_Second_Patch_8Lymphomas_Interim_Scans/"

segment(home_dir, "t", "clin_ct_peripheral_bones")

# for scan in os.listdir(home_dir):
#     if os.path.isdir(os.path.join(home_dir, scan)):
#         scan_dir = os.path.join(home_dir, scan)
#         print(scan)
        

# for patient in os.listdir(home_dir):
#     if os.path.isdir(os.path.join(home_dir, patient)):
#         patient_dir = os.path.join(home_dir, patient)
#         patient_dir = os.path.join(patient_dir, "Unnamed - 0")
#         #print(patient_dir)
#         for scan in os.listdir(patient_dir):
#             scan_dir = os.path.join(patient_dir, scan)
#             if os.path.isdir(scan_dir):
#                 #print(scan)
#                 if scan[:2] == "CT":
#                     seg_name = patient + "_" + scan
#                     print(scan_dir)
#                     segment(scan_dir, seg_name, "total")

# for patient in os.listdir(home_dir):
#     print("Working on patient: ", patient)
#     if os.path.isdir(os.path.join(home_dir, patient)):

#         patient_dir = os.path.join(home_dir, patient)
#         patient_dir = os.path.join(patient_dir, "study")

#         for scan in os.listdir(patient_dir):
#             scan_dir = os.path.join(patient_dir, scan)
#             if os.path.isdir(scan_dir):
#                 site = patient[:4]
#                 if site == "1002":
#                     if scan[:4] == "CTAC":
#                         print(scan_dir)
#                         segment(scan_dir, patient+"-total", "total")
#                         segment(scan_dir, patient+"-appendicular_bones", "appendicular_bones")
#                 elif site == "1003":
#                     if scan[:2] == "CT" or scan == "Standard-Full":
#                         if patient[:11] == "1003010-269":
#                             if scan == "Standard-Full":
#                                 print(scan_dir)
#                                 segment(scan_dir, patient+"-total", "total")
#                                 segment(scan_dir, patient+"-appendicular_bones", "appendicular_bones")
#                         else:
#                             print(scan_dir)
#                             segment(scan_dir, patient+"-total", "total")
#                             segment(scan_dir, patient+"-appendicular_bones", "appendicular_bones")
#                 elif site == "1005":
#                     if scan[:11] == "NON_DIAG_CT":
#                         print(scan_dir)
#                         segment(scan_dir, patient+"-total", "total")
#                         segment(scan_dir, patient+"-appendicular_bones", "appendicular_bones")

#                 # elif site == "1010":
#                 #     print("test")
#                 # elif site == "1011":
#                 #     print("test")

#                 elif site == "1012":                    
#                     if scan[-4:] == "eFoV":
#                         print(scan_dir)
#                         segment(scan_dir, patient+"-total", "total")
#                         segment(scan_dir, patient+"-appendicular_bones", "appendicular_bones")
#                     if patient == "1012006-351-B" or patient == "1012005-444-B" or patient == "1012003-802-B" or patient == "1012003-307-B":
#                         if scan[-4:] == "B31f":
#                             print(scan_dir)
#                             segment(scan_dir, patient+"-total", "total")
#                             segment(scan_dir, patient+"-appendicular_bones", "appendicular_bones")





                    # scan_dir = os.path.join(patient_dir, scan_time)
                    # with os.scandir(scan_dir) as scans:
                    #     for scan in scans:
                    #         if scan.is_dir():
                    #             if scan.name[:2] == "CT":
                    #                 print(counter)
                    #                 DICOM = os.path.join(scan_dir, scan.name)
                    #                 segmentName = scan_dir.replace(home_dir, '').replace('/','_')
                    #                 command = f'TotalSegmentator -i "{DICOM}" -o "./segmentations/{segmentName}" --ml --force_split'
                    #                 runBash(command)
                    #                 counter = counter + 1


# command = 'TotalSegmentator -i "/media/billiam/T7 Shield/UC Davis COVID Study/COVID Patients/1697954_FDG_COVID_Pt002_JP/20121026/CT_SOFT_512x512/" -o "~/Documents/Scans/notebooktest" --ml'

