import pydicom
import os
from datetime import datetime

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


home_dir = "E:/UC Davis COVID Study/COVID Patients/"

for patient in os.listdir(home_dir):
    patient_dir = os.path.join(home_dir, patient)
    #print(patient_dir)
    if os.path.isdir(patient_dir):
        for date in os.listdir(patient_dir):
            date_dir = os.path.join(patient_dir, date)
            if os.path.isdir(date_dir):
                for scan in os.listdir(date_dir):
                    if scan[:2] == "CT":
                        scan_dir = os.path.join(date_dir, scan)
                        #print(scan_dir)
                        full_path = os.path.join(scan_dir, os.listdir(scan_dir)[0])
                        ds = pydicom.dcmread(full_path)
                        time_diff = (calculate_time_difference(ds.SeriesTime, ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime))
                        print(scan_dir)
                        print(time_diff)
                        print(ds.SeriesTime)
                        print(ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime)

# for scan in os.listdir(home_dir):
#     scan_file = os.path.join(home_dir, scan)
#     if os.path.isfile(scan_file):
#         print(scan_file)
#         ds = pydicom.dcmread(scan_file)
#         time_diff = abs(calculate_time_difference(ds.SeriesTime, ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime))
#         print(ds.SeriesTime)
#         print(ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime)
