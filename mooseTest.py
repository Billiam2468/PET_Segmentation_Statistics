from moosez import moose

import dicom2nifti


if __name__ == '__main__':
    model_name = 'clin_ct_peripheral_bones'
    #input_dir = 'E:/Psoriasis/VIP-S/1002001-086-A/study/CTAC_HEAD_IN_TRANS_TOMO/'
    input_dir = 'E:/Clear_Example_CT/Nifti/'

    #dicom2nifti.convert_directory(input_dir, input_dir, compression=True, reorient=True)

    output_dir = "D:/Documents/Alavi Lab/Moose/"
    accelerator = 'cuda'
    moose(model_name, input_dir, output_dir, accelerator)