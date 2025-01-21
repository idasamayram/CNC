import h5py
import os

def get_h5_file_info(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File '{file_path}' does not exist.")
        return

    # Get the file size in bytes
    file_size = os.path.getsize(file_path)

    print(f"File Size: {file_size} bytes")

    # Open the HDF5 file
    with h5py.File(file_path, 'r') as h5_file:
        # Iterate over each dataset in the file
        def print_dataset_info(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}")
                print(f" - Shape: {obj.shape}")
                print(f" - Size: {obj.size}")
                print(f" - Data type: {obj.dtype}")

        h5_file.visititems(print_dataset_info)

# Example usage
file_path = '../data/final/Selected_data_windowed_grouped_normalized/bad/M01_Aug_2019_OP01_000_window_2.h5'
get_h5_file_info(file_path)

