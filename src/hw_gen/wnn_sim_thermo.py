import os
import torch
import torch_dwn as dwn
import openml
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from torch import nn

def read_mapping(mapping_filepath):
    #with open(mapping_filepath, 'r') as f:
    #    mapping_str_list = f.readline().strip().split(';')
    
    mapping_str = "660;3116;3185;22;487;3096;13;671;684;3037;479;60;2823;672;3166;683;3126;660;2965;3109;3172;577;1706;657;3125;673;682;2952;773;2290;3130;3175;2816;673;673;2821;3135;3071;6;453;2316;502;664;579;3127;3056;2;3140;2346;3182;3144;664;3102;572;68;475;3080;472;1968;3167;23;2884;2885;2921;23;721;3166;3107;3080;2885;2927;722;2924;75;720;2567;719;674;198;2879;718;2877;3125;719;721;2878;3129;717;720;149;2921;185;720;719;69;2883;721;3128;3140;719;2926;723;2919;722;2882;2923;2199;114;720;2881;718;717;717;718;26;718;2882;2917;135;721;753;2303;2995;51;3098;711;2568;563;0;563;752;123;564;1;752;1735;2565;148;569;563;160;3089;2996;2488;713;756;2922;3194;566;2962;571;2924;714;3185;2997;560;124;556;2997;562;0;715;2005;117;2998;751;750;104;1661;565;3;2253;754;2973;750;988;544;2994;2288;0;754;2884;2914;43;714;258;2956;2918;1978;3151;755;754;755;3139;3123;755;2917;753;97;2954;2914;2954;2921;753;2920;115;683;2954;3131;753;716;2922;2954;753;2913;2955;2918;2921;2919;2832;754;2955;2918;194;2957;755;753;2916;3136;2916;2956;754;2956;3163;2953;2921;59;2956;15;59;184;138;752;2977;2870;156;3095;2876;1782;230;2057;180;186;2690;168;2954;163;2520;2463;2520;2877;441;3025;2881;3119;160;2097;2876;2877;677;3154;144;2879;471;474;144;753;156;2864;2971;2675;2866;3150;119;2875;475;180;1339;131;2878;950;2268;2879;2867;3083;469;2875;3127;2954;203"
    mapping_str_list = mapping_str.split(';')
    mapping = [int(x) for x in mapping_str_list]
    return mapping

def thermometer_encode(sample, thresholds, bits):
    output = 0
    count = 0

    for threshold in thresholds:
        if sample >= threshold:
            output = (output << 1) | 1
            count += 1
        else:
            break

    # Pad with zeros to make full width
    output = output << (bits - count)

    # Convert to binary string and reverse
    bit_str = f"{output:0{bits}b}"[::-1]
    return bit_str

def encode_sample_mapping(sample: torch.Tensor, mapping):
    sample_list = [int(x) for x in sample.cpu().flatten().numpy().tolist()]
    
    #print(f"{len(sample_list)=}")
    #print(f"{len(mapping)=}")
    
    output_str = "".join(str(sample_list[i]) for i in mapping)
    
    return output_str

def encode_samples_thresholds(samples: torch.Tensor, thresholds: torch.Tensor, mapping, bits=200, feature_wise=True) -> list[int]:
    """
    Encode multiple samples using the thermometer thresholds and return a flat list of 0s and 1s.
    
    Args:
        samples: Tensor of shape (N, F) where N = number of samples, F = features
        thresholds: 1D tensor of sorted thresholds
    
    Returns:
        Flat list[int] of thermometer encoded bits
    """
    samples = samples.flatten()
    thresholds = torch.tensor(thresholds, dtype=samples.dtype, device=samples.device)
    
    output = ""
    encoded_all = "" 
    
    print(len(samples))
        
    for sample in samples: 
        
        if feature_wise:
            feature_list_index = 0
            output = output + "" + thermometer_encode(sample, thresholds[feature_list_index], bits)
            
            feature_list_index += 1
        else:
            output = output + "" + thermometer_encode(sample, thresholds, bits)

    
    for i in mapping[::-1]:
        if i < len(output):
            encoded_all += output[i]
        else:
            print(f"Index {i} out of range for output string of length {len(output)}")

    encoded_all = output
    return encoded_all

def main(args):

    num_samples = args.num_of_samples
    mapping = read_mapping(args.mapping)
    dataset_str = args.dataset
    thermometer_bits = args.bits
    scale_factor = args.scaling
    feature_wise = True

    # Load OpenML dataset
    dataset = openml.datasets.get_dataset(42468)
    df_features, df_labels, _, _ = dataset.get_data(
        dataset_format='dataframe', target=dataset.default_target_attribute
    )
    features = df_features.values.astype(np.float32)
    label_names = list(df_labels.unique())
    labels = np.array(df_labels.map(lambda x: label_names.index(x)).values)

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, train_size=0.8, random_state=42
    )

    thermometer = dwn.DistributiveThermometer(thermometer_bits, feature_wise=feature_wise).fit(x_train)
    
    thresholds = (thermometer.thresholds * scale_factor).to(torch.int32).tolist()
    
    #print(f"Thermometer thresholds: {thresholds}")
    input("Press Enter to continue...")
    
    x_train_bin = thermometer.binarize(x_train).flatten(start_dim=1)
    x_test_bin = thermometer.binarize(x_test).flatten(start_dim=1)

    
    data_samples_bin, targets = (x_train_bin, y_train) if args.dataset == 'train' else (x_test_bin, y_test)
    num_samples = min(args.num_of_samples, len(data_samples_bin))

    # Iterate over dataset
    for sample_index in range(num_samples):
        data_sample_bin = data_samples_bin[sample_index:sample_index + 1]
        data_sample_float = x_train[sample_index:sample_index + 1]
        
        # Rescaling: Konvertiere numpy.ndarray zu torch.Tensor, bevor .to() verwendet wird
        data_sample_float = torch.tensor(data_sample_float * scale_factor, dtype=torch.int32)
    
        
        encoded_sample_mapping = encode_sample_mapping(data_sample_bin, mapping)
        encoded_sample_thresholds = encode_samples_thresholds(data_sample_float, thresholds, mapping, thermometer_bits, feature_wise)
        
        print(f"Encoded sample mapping: {data_sample_bin}")
        print(f"Encoded sample thresholds: {encoded_sample_thresholds}")
        
        input("Press Enter to continue...")
        
        
if __name__ == "__main__":

    # Create argparse parser
    parser = argparse.ArgumentParser(description='Run WNN vs PyTorch model')
    parser.add_argument('-m', '--mapping', type=str, default='/src/hw_gen/mapping.txt', help='Input mapping file path')
    parser.add_argument('-n', '--num_of_samples', type=int, default=10, help='Number of samples to use')
    parser.add_argument('-d', '--dataset', type=str, choices=['train', 'test'], default='train', help='Dataset: train or test')
    parser.add_argument('-b', '--bits', type=int, default=200, help='Number of thermometer bits (default: 200)')
    parser.add_argument('-s', '--scaling', type=float, default=10000, help='Scaling factor for thermometer (default: 100000)')
    
    args = parser.parse_args()

    main(args)