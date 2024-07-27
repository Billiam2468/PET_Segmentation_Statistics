import numpy as np

# Helper function that takes in an organ and SUV values of a single patient and extracts the correct statitistics. Adapted to extract total lung weighted avg across lobes (lung) as well as all vertebrae and rib
def extractPatientStats(suvs, organ, stat):
    if organ == "lung":
        val1 = suvs['lung_upper_lobe_left'][stat]
        val1num = suvs['lung_upper_lobe_left']['num_val']
        val2 = suvs['lung_lower_lobe_left'][stat]
        val2num = suvs['lung_lower_lobe_left']['num_val']
        val3 = suvs['lung_upper_lobe_right'][stat]
        val3num = suvs['lung_upper_lobe_right']['num_val']
        val4 = suvs['lung_middle_lobe_right'][stat]
        val4num = suvs['lung_middle_lobe_right']['num_val']
        val5 = suvs['lung_lower_lobe_right'][stat]
        val5num = suvs['lung_lower_lobe_right']['num_val']
        avg = (val1*val1num + val2*val2num + val3*val3num + val4*val4num + val5*val5num)/(val1num+val2num+val3num+val4num+val5num)
        return avg
    if (organ == "vertebrae") or (organ == "rib"):
        rawVals = []
        numVals = []
        for name in total_segmentator_names:
            if name.startswith(organ):
                val = suvs[name][stat]
                rawVals.append(val)
                numVal = suvs[name]['num_val']
                numVals.append(numVal)
        numerator = 0
        denominator = 0
        for idx,val in enumerate(rawVals):
            numerator = numerator + (rawVals[idx] * numVals[idx])
            denominator = denominator + numVals[idx]
        #print("length of vertebrae ", len(numVals))
        return (numerator/denominator)
    else:
        return suvs[organ][stat]



home_dict = "D:/Documents/Repos/PET_Segmentation_Statistics/"

suv_dict1 = np.load(home_dict + "baseline_aorta_suv_stats.npy", allow_pickle='TRUE').item()
suv_dict2 = np.load(home_dict + "interim_aorta_suv_stats.npy", allow_pickle='TRUE').item()

#data = suv_dict1['BASELINE_1470016_Sub0312_As_CT_SOFT_120.nii.gz']
#print(data['aorta']['mean'])


# print(extractPatientStats(data, 'aorta', 'mean'))


baseline_scans = [
    "BASELINE_1470016_Sub0025_Kb_CT_SOFT_120.nii.gz",
    "BASELINE_1470016_Sub0025_Kb_CT_SOFT_60.nii.gz",
    "BASELINE_1470016_Sub0042_Aj_CT_SOFT_120.nii.gz",
    "BASELINE_1470016_Sub0042_Aj_CT_SOFT_60.nii.gz",
    #"BASELINE_1470016_Sub0068_Hh_CT_SOFT_120.nii.gz",
    #"BASELINE_1470016_Sub0068_Hh_CT_SOFT_60.nii.gz",
    "BASELINE_1470016_Sub0125_Js_CT_SOFT_120.nii.gz",
    "BASELINE_1470016_Sub0125_Js_CT_SOFT_60.nii.gz",
    "BASELINE_1470016_Sub0205_Ao_CT_SOFT_120.nii.gz",
    "BASELINE_1470016_Sub0205_Ao_CT_SOFT_60.nii.gz",
    "BASELINE_1470016_Sub0256_Ei_CT_SOFT_120.nii.gz",
    "BASELINE_1470016_Sub0256_Ei_CT_SOFT_60.nii.gz",
    "BASELINE_1470016_Sub0312_As_CT_SOFT_120.nii.gz",
    "BASELINE_1470016_Sub0312_As_CT_SOFT_60.nii.gz",
    "BASELINE_1470016_Sub0329_Mm_CT_SOFT_120.nii.gz",
    "BASELINE_1470016_Sub0329_Mm_CT_SOFT_60.nii.gz",
    "BASELINE_1470016_Sub0343_Ck_CT_SOFT_120.nii.gz",
    "BASELINE_1470016_Sub0343_Ck_CT_SOFT_60.nii.gz",
    "BASELINE_1470016_Sub0350_Rl_CT_SOFT_120.nii.gz",
    "BASELINE_1470016_Sub0350_Rl_CT_SOFT_60.nii.gz",
    "BASELINE_1470016_Sub0403_Tw_CT_SOFT_120.nii.gz",
    "BASELINE_1470016_Sub0403_Tw_CT_SOFT_60.nii.gz",
    "BASELINE_1470016_Sub0498_Lc_CT_SOFT_120.nii.gz",
    "BASELINE_1470016_Sub0498_Lc_CT_SOFT_60.nii.gz",
    "BASELINE_1470016_Sub0642_Ab_CT_SOFT_120.nii.gz",
    "BASELINE_1470016_Sub0642_Ab_CT_SOFT_60.nii.gz",
    "BASELINE_1470016_Sub0647_Dn_CT_SOFT_120.nii.gz",
    "BASELINE_1470016_Sub0647_Dn_CT_SOFT_60.nii.gz",
    "BASELINE_1470016_Sub0688_Jl_CT_SOFT_120.nii.gz",
    "BASELINE_1470016_Sub0688_Jl_CT_SOFT_60.nii.gz",
    "BASELINE_1470016_Sub0701_Gl_CT_SOFT_120.nii.gz",
    "BASELINE_1470016_Sub0701_Gl_CT_SOFT_60.nii.gz",
    "BASELINE_1470016_Sub0744_Kh_CT_SOFT_120.nii.gz",
    "BASELINE_1470016_Sub0744_Kh_CT_SOFT_60.nii.gz",
    "BASELINE_1470016_Sub0749_Nb_CT_SOFT_120.nii.gz",
    "BASELINE_1470016_Sub0749_Nb_CT_SOFT_60.nii.gz",
    "BASELINE_1470016_Sub0820_Tm_CT_SOFT_120.nii.gz",
    "BASELINE_1470016_Sub0820_Tm_CT_SOFT_60.nii.gz",
    "BASELINE_1470016_Sub0869_Cr_CT_SOFT_120.nii.gz",
    "BASELINE_1470016_Sub0869_Cr_CT_SOFT_60.nii.gz",
]

interim_scans = [
    #"INTERIM_1470016_Sub0068_Hh_CT_SOFT_120.nii.gz",
    #"INTERIM_1470016_Sub0068_Hh_CT_SOFT_60.nii.gz",
    "INTERIM_1470016_Sub0205_Ao_CT_SOFT_120.nii.gz",
    "INTERIM_1470016_Sub0205_Ao_CT_SOFT_60.nii.gz",
    "INTERIM_1470016_Sub0498_Lc_CT_SOFT_120.nii.gz",
    "INTERIM_1470016_Sub0498_Lc_CT_SOFT_60.nii.gz",
    "INTERIM_1470016_Sub0647_Dn_CT_SOFT_120.nii.gz",
    "INTERIM_1470016_Sub0647_Dn_CT_SOFT_60.nii.gz",
    "INTERIM_1470016_Sub0688_Jl_CT_SOFT_120.nii.gz",
    "INTERIM_1470016_Sub0688_Jl_CT_SOFT_60.nii.gz",
    "INTERIM_1470016_Sub0701_Gl_CT_SOFT_120.nii.gz",
    "INTERIM_1470016_Sub0701_Gl_CT_SOFT_60.nii.gz",
    "INTERIM_1470016_Sub0744_Kh_CT_SOFT_120.nii.gz",
    "INTERIM_1470016_Sub0744_Kh_CT_SOFT_60.nii.gz",
    "INTERIM_1470016_Sub0749_Nb_CT_SOFT_120.nii.gz",
    "INTERIM_1470016_Sub0749_Nb_CT_SOFT_60.nii.gz"
]

organ = "aorta"
stat = "mean"

first_vals = []
second_vals = []

# for patient, suvs in suv_dict1.items():
#     start = patient.split("_")[0]
#     if start == "BASELINE":
#         val = extractPatientStats(suvs, organ, stat)
#         baseline_vals.append(val)

# for patient, suvs in suv_dict2.items():
#     start = patient.split("_")[0]
#     if start == "INTERIM":
#         val = extractPatientStats(suvs, organ, stat)
#         print(patient)
#         print(val)
#         interim_vals.append(val)

filename1 = "comparison.txt"
filename2 = "comparison2.txt"

with open(filename1, 'r') as file:
    for line in file:
        val = suv_dict1[line.strip()]['aorta']['mean']
        first_vals.append(val)
with open(filename2, 'r') as file:
    for line in file:
        val = suv_dict2[line.strip()]['aorta']['mean']
        second_vals.append(val)

for val in first_vals:
    print(val)

print("text")
for val in second_vals:
    print(val)

import matplotlib.pyplot as plt
import numpy as np

# Example data
covid_vals = second_vals
healthy_vals = first_vals
#covid_second_vals = COVID_second_vals

# Calculate the means
covid_mean = np.mean(covid_vals)
healthy_mean = np.mean(healthy_vals)
#covid_second_mean = np.mean(covid_second_vals)

# Scatter plot data
covid_x = np.ones(len(covid_vals)) * 2
healthy_x = np.ones(len(healthy_vals))
#covid_second_x = np.ones(len(covid_second_vals)) * 3

# Plotting the data
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size if necessary
ax.scatter(covid_x, covid_vals, color='red', label='120 MIN Scan')
ax.scatter(healthy_x, healthy_vals, color='green', label='60 MIN Scan')
#ax.scatter(covid_second_x, covid_second_vals, color='blue', label='COVID Scan 2 Values')

# Plotting the means
ax.axhline(y=covid_mean, color='red', linestyle='--', label=f'120 MIN MEAN: {covid_mean:.2f}')
ax.axhline(y=healthy_mean, color='green', linestyle='--', label=f'60 MIN MEAN: {healthy_mean:.2f}')
#ax.axhline(y=covid_second_mean, color='blue', linestyle='--', label=f'COVID Scan 2 Mean: {covid_second_mean:.2f}')

# Customizing the plot
ax.set_title(f'SUVmeans of {organ} across Baseline 60 and 120 MIN Scans')
ax.set_xticks([1, 2])
ax.set_xticklabels(['60 MIN Scan', '120 MIN Scan'])
ax.set_ylabel('SUVmean')

# Place legend to the side
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])  # Adjust plot size to make room for legend
legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Save the figure as a JPEG file
#plt.savefig(f'{organ}_values_comparison.jpg', format='jpg', bbox_inches='tight')
plt.show()
plt.close()
