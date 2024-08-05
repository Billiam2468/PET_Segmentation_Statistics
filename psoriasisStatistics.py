import numpy as np
import pandas as pd

joint_names = [
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

def combine_scans(path):
    """This function takes an npy file and combines the separate A,B,C scans into one and outputs the combined dictionary without the A,B,C suffixes"""
    combined_scans = {}
    scans = np.load(path, allow_pickle='TRUE').item()

    for scan_id, regions in scans.items():
        # Get the base scan ID without the component suffix (A, B, C)
        base_id = '-'.join(scan_id.split('-')[:-1])
        if base_id not in combined_scans:
            combined_scans[base_id] = {}

        for region, metrics in regions.items():
            if region not in combined_scans[base_id]:
                combined_scans[base_id][region] = {
                    'mean': None,
                    'max': None,
                    'median': None,
                    'num_val': 0
                }

            # Check and update metrics if they are not None and num_val is not zero
            for metric in ['mean', 'max', 'median']:
                if metrics[metric] is not None:
                    combined_scans[base_id][region][metric] = metrics[metric]

            # Update num_val if it's greater than the existing value
            if metrics['num_val'] > combined_scans[base_id][region]['num_val']:
                combined_scans[base_id][region]['num_val'] = metrics['num_val']

    return combined_scans

def detectPresentJoints(combinedScans):
    """This function takes a combined A,B,C scan dictionary from a npy statistics file and detects what joints are present. ABC are present when scans are separated into different body regions"""

    present_joints = {}

    for key, val in combinedScans.items():
        #print(key)
        scan_joints = []
        for key2, val2 in val.items():
            #print(key2)
            #print(val2['num_val'])
            if not val2['num_val'] == 0:
                scan_joints.append(joint_names.index(key2))
        present_joints[key] = scan_joints

    return present_joints
    #print(len(present_joints))
    #print(present_joints)


    # This below part of code will combine the A, B, and C scans together (present for some sites)
    # Dictionary to store combined results
    # combined_scans = {}

    # for key, array in present_joints.items():
    #     base_id = key.rsplit('-', 1)[0]  # Get base identifier (before last dash)
    #     if base_id not in combined_scans:
    #         combined_scans[base_id] = set(array)  # Use set to store unique elements
    #     else:
    #         combined_scans[base_id].update(array)  # Add unique elements

    # # Convert sets back to lists (optional)
    # combined_scans = {k: list(v) for k, v in combined_scans.items()}

    # # Print combined scans
    # for k, v in combined_scans.items():
    #     print(f"{k}: {v}")

#detectPresentJoints('D:/Documents/Repos/PET_Segmentation_Statistics/Psoriasis Statistics/site_1012_joint_suv_stats.npy')

def retrieveStats(combined_scans, scan, joint, stat):
    """This function will take a scan and return the requested statistics of the joint of interest"""
    # print(scan)
    # print(joint_names[joint])
    # print(stat)
    # print(combined_scans)
    return combined_scans[scan][joint_names[joint]][stat]

def calculateStatistics(combined_scans, joint, stat):
    #print(combined_scans)
    organizationFiles = "E:/Psoriasis/VIP-S Subject Tracker_Accounting Final.xlsx"
    df = pd.read_excel(organizationFiles)
    
    for index, row in df.iterrows():
        if str(row['Subject #']).startswith('1002'):
            if str(row['Subject #']) == "1002018":
                continue
            print(str(row['Subject #']))
            subject_name = str(row['Subject #'])
            code0 = row['IRC_Week 0 baseline']
            if not np.isnan(code0):
                week_0 = subject_name + "-" + f"{int(code0):03}"
                stat1 = retrieveStats(combined_scans, week_0, joint, stat)
                #print(week_0)
                print(stat1)
            code1 = row['IRC_Week 12']
            if not np.isnan(code1):
                week_1 = subject_name + "-" + f"{int(code1):03}"
                stat2 = retrieveStats(combined_scans, week_1, joint, stat)
                #print(week_1)
                print(stat2)
            code2 = row['IRC_Week 52']
            if not np.isnan(code2):
                week_2 = subject_name + "-" + f"{int(code2):03}"
                stat3 = retrieveStats(combined_scans, week_2, joint, stat)
                #print(week_2)
                print(stat3)
            # print(f"Week 0: {int(row['IRC_Week 0 baseline'])}")
            # print(f"Week 12: {int(row['IRC_Week 12'])}")
            # print(f"Week 52: {int(row['IRC_Week 52'])}")

combined = combine_scans('D:/Documents/Repos/PET_Segmentation_Statistics/Psoriasis Statistics/site_1002_joint_suv_stats.npy')
present_joints = detectPresentJoints(combined)

# For our first test we will calculate 10,11,14,15 for site 1002
calculateStatistics(combined, 10, 'mean')