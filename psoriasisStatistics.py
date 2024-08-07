import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

def combine_scans(path):
    """This function takes an npy file and combines the separate A,B,C scans into one and outputs the combined dictionary without the A,B,C suffixes"""
    combined_scans = {}
    scans = np.load(path, allow_pickle='TRUE').item()
    for key,val in scans.items():
        print(key)

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

def calculateStatistics(combined_scans, joint, stat, site):
    #print(combined_scans)
    organizationFiles = "E:/Psoriasis/VIP-S Subject Tracker_Accounting Final.xlsx"
    df = pd.read_excel(organizationFiles)
    
    # Initiating empty dataframe:
    columns = ["Patient", "Week 0", "Week 12", "Week 52"]
    statistics_df = pd.DataFrame(columns=columns)

    for index, row in df.iterrows():
        if str(row['Subject #']).startswith(site):
            # if str(row['Subject #']) == "1002018":
            #     continue

            # if str(row['Subject #']) == "1003005":
            #    continue


            # if str(row['Subject #']) == "1003007" and row['IRC_Week 0 baseline'] == 531:
            #    continue
            # if str(row['Subject #']) == "1005001" and row['IRC_Week 0 baseline'] == 322:
            #    continue
            # if str(row['Subject #']) == "1005002" and row['IRC_Week 0 baseline'] == 742:
            #    continue
            # if str(row['Subject #']) == "1005004" and row['IRC_Week 0 baseline'] == 295:
            #    continue
            # if str(row['Subject #']) == "1005007" and row['IRC_Week 12'] == 241:
            #    continue
            # if str(row['Subject #']) == "1005011" and row['IRC_Week 0 baseline'] == 764:
            #    continue
            # if str(row['Subject #']) == "1005013" and row['IRC_Week 0 baseline'] == 890:
            #    continue
            
            # if str(row['Subject #']) == "1012001" and row['IRC_Week 0 baseline'] == 980:
            #    continue
            # if str(row['Subject #']) == "1012003" and row['IRC_Week 0 baseline'] == 802:
            #    continue
            # if str(row['Subject #']) == "1012006" and row['IRC_Week 0 baseline'] == 351:
            #    continue

            subject_name = str(row['Subject #'])

            #Add patient data to df:
            patient_data = {"Patient": subject_name}

            code0 = row['IRC_Week 0 baseline']
            if not np.isnan(code0):
                week_0 = subject_name + "-" + f"{int(code0):03}"

                if week_0 in combined_scans:
                    stat1 = retrieveStats(combined_scans, week_0, joint, stat)
                    patient_data["Week 0"] = stat1
                else:
                    patient_data["Week 0"] = np.nan
            else:
                patient_data["Week 0"] = np.nan


            code1 = row['IRC_Week 12']
            if not np.isnan(code1):
                week_1 = subject_name + "-" + f"{int(code1):03}"
                if week_1 in combined_scans:
                    stat2 = retrieveStats(combined_scans, week_1, joint, stat)
                    patient_data["Week 12"] = stat2
                else:
                    patient_data["Week 12"] = np.nan
            else:
                patient_data["Week 12"] = np.nan
            

            code2 = row['IRC_Week 52']
            if not np.isnan(code2):
                week_2 = subject_name + "-" + f"{int(code2):03}"

                if week_2 in combined_scans:
                    stat3 = retrieveStats(combined_scans, week_2, joint, stat)
                    patient_data["Week 52"] = stat3
                else:
                    patient_data["Week 52"] = np.nan
            else:
                patient_data["Week 52"] = np.nan
            # print(f"Week 0: {int(row['IRC_Week 0 baseline'])}")
            # print(f"Week 12: {int(row['IRC_Week 12'])}")
            # print(f"Week 52: {int(row['IRC_Week 52'])}")

            # Append the patient's data to the DataFrame
            patient_data = pd.DataFrame([patient_data])
            statistics_df = pd.concat([statistics_df, patient_data])
            #statistics_df = statistics_df.append(patient_data, ignore_index=True)
    return statistics_df

def plot_patient_statistics(statistics_df, joint, save_path):
    # Melt the DataFrame to have 'Time Point' as a variable and 'Value' as value
    plot_df = statistics_df.melt(id_vars="Patient", var_name="Time Point", value_name="Value")
    
    # Set the style and size of the plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Create a line plot for each patient
    sns.lineplot(data=plot_df, x="Time Point", y="Value", hue="Patient", marker='o', palette='tab10')
    
    # Add a title and labels
    plt.title(f"SUV means of {joint_names[joint]} joint over time")
    plt.xlabel("Time Point")
    plt.ylabel("Value")
    
    # Adjust legend to prevent it from being cut off
    plt.legend(title="Patient", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Adjust layout to make room for the legend
    plt.tight_layout()

    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path)

    # Show the plot
    #plt.show()



combined = combine_scans('D:/Documents/Repos/PET_Segmentation_Statistics/Psoriasis Statistics/site_1003_joint_suv_stats.npy')

# No need to do combine for Site 1005 since no A,B,C scans
#combined = np.load('D:/Documents/Repos/PET_Segmentation_Statistics/Psoriasis Statistics/site_1005_joint_suv_stats.npy', allow_pickle='TRUE').item()
present_joints = detectPresentJoints(combined)
print(present_joints)

# # For our first test we will calculate 10,11,14,15 for site 1002
for i in range(1, 19):
    print(f"For Joint {i}")
    statistics_df = calculateStatistics(combined, i, 'mean', '1003')

    print(statistics_df)
    # Excluding Week 12 from scans
    #statistics_df = statistics_df.drop(columns=['Week 12'])
    save_path = "D:/Documents/Repos/PET_Segmentation_Statistics/Psoriasis Statistics/Figures/Site 1003/" + f"{joint_names[i]}.png"
    plot_patient_statistics(statistics_df, i, save_path)

