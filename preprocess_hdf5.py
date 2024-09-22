import os
import h5py
import numpy as np
import fnmatch
import argparse

def find_all_hdf5_files(dataset_dir, skip_mirrored_data=False):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename:
                continue
            if skip_mirrored_data and 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    return hdf5_files

def filter_excluded_data(loaded_hdf5):
    if 'task' not in loaded_hdf5:
        # No filtering needed, read all data
        data_dict = {}
        for key in loaded_hdf5.keys():
            if isinstance(loaded_hdf5[key], h5py.Dataset):
                data_dict[key] = loaded_hdf5[key][()]
            else:
                # Handle groups like 'observations'
                data_dict[key] = {}
                for subkey in loaded_hdf5[key].keys():
                    if isinstance(loaded_hdf5[key][subkey], h5py.Dataset):
                        data_dict[key][subkey] = loaded_hdf5[key][subkey][()]
                    else:
                        data_dict[key][subkey] = {}
                        for subsubkey in loaded_hdf5[key][subkey].keys():
                            data_dict[key][subkey][subsubkey] = loaded_hdf5[key][subkey][subsubkey][()]
        data_dict['attrs'] = dict(loaded_hdf5.attrs)
        return data_dict

    data = loaded_hdf5['task'][()]
    idx = np.array(data) == b'play'
    indices = np.where(idx)[0]
    length = len(indices)
    
    data_dict = {}
    # Read and filter datasets
    data_dict['action'] = loaded_hdf5['action'][indices]
    data_dict['task'] = data[indices]
    if 'base_action' in loaded_hdf5:
        data_dict['base_action'] = loaded_hdf5['base_action'][indices]
    data_dict['observations'] = {}
    for key in loaded_hdf5['observations'].keys():
        if key != 'images':
            data_dict['observations'][key] = loaded_hdf5['observations'][key][indices]
        else:
            data_dict['observations']['images'] = {}
            for cam in ['cam_high', 'cam_left_wrist', 'cam_low', 'cam_right_wrist']:
                if cam in loaded_hdf5['observations']['images']:
                    data_dict['observations']['images'][cam] = loaded_hdf5['observations']['images'][cam][indices]
    data_dict['attrs'] = dict(loaded_hdf5.attrs)
    return data_dict

def preprocess_hdf5_file(input_path, output_path):
    with h5py.File(input_path, 'r') as infile:
        data_dict = filter_excluded_data(infile)

    with h5py.File(output_path, 'w') as outfile:
        # Copy attributes
        for attr_name, attr_value in data_dict.get('attrs', {}).items():
            outfile.attrs[attr_name] = attr_value

        # Write datasets and groups
        for key in data_dict.keys():
            if key == 'observations':
                observations_group = outfile.create_group('observations')
                for obs_key, obs_value in data_dict['observations'].items():
                    if obs_key != 'images':
                        observations_group.create_dataset(obs_key, data=obs_value)
                    else:
                        images_group = observations_group.create_group('images')
                        for cam, img_data in obs_value.items():
                            images_group.create_dataset(cam, data=img_data)
            elif key != 'attrs':
                outfile.create_dataset(key, data=data_dict[key])

def preprocess_dataset(input_dir, output_dir, skip_mirrored_data=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    hdf5_files = find_all_hdf5_files(input_dir, skip_mirrored_data)
    print(f'Found {len(hdf5_files)} HDF5 files to preprocess.')

    for input_path in hdf5_files:
        # Determine the output path
        relative_path = os.path.relpath(input_path, input_dir)
        output_path = os.path.join(output_dir, relative_path)
        output_dirname = os.path.dirname(output_path)

        if not os.path.exists(output_dirname):
            os.makedirs(output_dirname)

        print(f'Processing {input_path} -> {output_path}')
        preprocess_hdf5_file(input_path, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess HDF5 dataset.')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the original dataset directory.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save the preprocessed dataset.')
    parser.add_argument('--skip_mirrored_data', action='store_true', help='Whether to skip mirrored data.')
    args = parser.parse_args()

    preprocess_dataset(args.input_dir, args.output_dir, args.skip_mirrored_data)
