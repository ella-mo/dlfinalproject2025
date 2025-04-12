"""
This file extracts file data from NWB file and saves it to a csv
"""

#!/usr/bin/env python
import sys
import os
import numpy as np
import h5py
import csv
from pynwb import NWBHDF5IO

def extract_external_file(nwb_filepath):
    """
    Extract external_file data from an NWB file and save as CSV
    
    Parameters:
    -----------
    nwb_filepath : str
        Path to the NWB file
    
    Returns:
    --------
    None, but saves the external file data as CSV in the script directory
    """
    print(f"Opening NWB file: {nwb_filepath}")
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = "extracted_external_file.csv"
    output_path = os.path.join(script_dir, output_filename)
    
    # Open the NWB file using h5py for direct access to the HDF5 structure
    try:
        with h5py.File(nwb_filepath, 'r') as f:
            # Based on the file structure in your screenshot
            # Navigate to the external_file location
            try:
                path = "/stimulus/presentation/presentations/indexed_timeseries/external_file"
                if path in f:
                    ext_file_data = f[path][()]
                    
                    # Save data to CSV, handling different data types
                    save_to_csv(ext_file_data, output_path)
                    print(f"External file data saved to {output_path}")
                else:
                    print(f"Path {path} not found in NWB file")
                    print("Available paths at the stimulus level:")
                    if "/stimulus" in f:
                        list_paths(f, "/stimulus", level=3)
                    else:
                        print("No /stimulus group found")
                    
                    # Try using PyNWB as an alternative
                    print("Trying to access data using PyNWB...")
                    try_pynwb_approach(nwb_filepath, output_path)
            except Exception as e:
                print(f"Error accessing external_file data: {str(e)}")
                
                # Try using PyNWB as an alternative
                print("Trying to access data using PyNWB...")
                try_pynwb_approach(nwb_filepath, output_path)
    except Exception as e:
        print(f"Error opening NWB file with h5py: {str(e)}")
        print("Trying to access using PyNWB as fallback...")
        try_pynwb_approach(nwb_filepath, output_path)

def save_to_csv(data, output_path):
    """
    Save data to CSV file format, handling different data types
    """
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Handle different data types
        if isinstance(data, (str, bytes)):
            # For string or bytes data
            if isinstance(data, bytes):
                try:
                    data = data.decode('utf-8')
                except UnicodeDecodeError:
                    data = str(data)
            
            # Write as single row
            writer.writerow(['external_file_data'])
            writer.writerow([data])
            
        elif isinstance(data, np.ndarray):
            # For array data
            if data.ndim == 1:
                # For 1D array, write column header and data
                writer.writerow(['value'])
                for value in data:
                    writer.writerow([value])
            elif data.ndim == 2:
                # For 2D array, write each row
                for row in data:
                    writer.writerow(row)
            else:
                # For higher dimensions, flatten and write with indices
                writer.writerow(['index', 'value'])
                flat_data = data.flatten()
                for i, value in enumerate(flat_data):
                    writer.writerow([i, value])
        
        elif isinstance(data, h5py.h5r.Reference):
            # For HDF5 references, just note it's a reference
            writer.writerow(['reference_info'])
            writer.writerow(['HDF5 Reference - cannot be directly represented in CSV'])
        
        else:
            # For other types, convert to string
            writer.writerow(['data'])
            writer.writerow([str(data)])

def list_paths(h5file, base_path, level=1, current_level=0):
    """List paths in HDF5 file for debugging"""
    if current_level > level:
        return
    
    if base_path in h5file:
        obj = h5file[base_path]
        if isinstance(obj, h5py.Group):
            for key in obj.keys():
                full_path = f"{base_path}/{key}"
                print(full_path)
                list_paths(h5file, full_path, level, current_level + 1)
        else:
            print(f"{base_path} (Dataset)")

def try_pynwb_approach(nwb_filepath, output_path):
    """Try to access external_file using PyNWB and save to CSV"""
    try:
        with NWBHDF5IO(nwb_filepath, 'r') as io:
            nwbfile = io.read()
            
            # Navigate through the stimulus presentations
            if hasattr(nwbfile, 'stimulus'):
                stimulus = nwbfile.stimulus
                if hasattr(stimulus, 'presentation'):
                    for key, timeseries in stimulus.presentation.items():
                        print(f"Found stimulus presentation: {key}")
                        
                        # Check if this is the indexed_timeseries
                        if hasattr(timeseries, 'external_file'):
                            ext_file = timeseries.external_file
                            print(f"External file: {ext_file}")
                            save_to_csv(ext_file, output_path)
                            print(f"External file data saved to {output_path}")
                            return True
                        elif hasattr(timeseries, 'indexed_timeseries'):
                            for idx_ts in timeseries.indexed_timeseries:
                                if hasattr(idx_ts, 'external_file'):
                                    ext_file = idx_ts.external_file
                                    print(f"External file from indexed_timeseries: {ext_file}")
                                    save_to_csv(ext_file, output_path)
                                    print(f"External file data saved to {output_path}")
                                    return True
            
            print("Could not find external_file data using PyNWB approach")
            return False
    except Exception as e:
        print(f"Error with PyNWB approach: {str(e)}")
        return False

def main():
    nwb_filepath = input("Enter the path to the .nwb file: ").strip()

    if not os.path.exists(nwb_filepath):
        print(f"Error: File {nwb_filepath} does not exist")
        sys.exit(1)
        
    extract_external_file(nwb_filepath)

if __name__ == "__main__":
    main()