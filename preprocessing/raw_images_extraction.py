"""
Extract PNG files from Stimuli.zip based on paths in the CSV file.
"""

#!/usr/bin/env python
import os
import csv
import subprocess
import sys

def extract_images_from_zip_using_csv():
    """
    Extract PNG files from Stimuli.zip based on paths in the CSV file.
    Places the extracted files in a 'raw_images' directory in the project folder.
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = script_dir
    
    # Paths for files
    csv_path = os.path.join(project_dir, "extracted_external_file.csv")
    zip_path = os.path.join(project_dir, "Stimuli.zip")
    output_dir = os.path.join(project_dir, "raw_images")
    
    # Check if files exist
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return False
    
    if not os.path.exists(zip_path):
        print(f"Error: Stimuli.zip not found at {zip_path}")
        return False
    
    # Create raw_images directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Read file paths from CSV
    file_paths = []
    try:
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header row
            for row in reader:
                if row and row[0]:
                    # Clean up the path - remove b' and ' characters from byte string representation
                    path = row[0].strip()
                    if path.startswith("b'") and path.endswith("'"):
                        path = path[2:-1]  # Remove b' and ' from beginning and end
                    file_paths.append(path)
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return False
    
    if not file_paths:
        print("No file paths found in the CSV file.")
        return False
    
    print(f"Found {len(file_paths)} file paths in the CSV.")
    
    # Extract each file from the zip
    success_count = 0
    failure_count = 0
    
    for path in file_paths:
        try:
            print(f"Extracting: {path}")
            cmd = ["7z", "e", zip_path, path, "-o" + output_dir, "-y"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                success_count += 1
                # Display progress every 10 files
                if success_count % 10 == 0:
                    print(f"Successfully extracted {success_count} files so far...")
            else:
                print(f"Error extracting {path}:")
                print(result.stderr)
                failure_count += 1
        except Exception as e:
            print(f"Exception while extracting {path}: {str(e)}")
            failure_count += 1
    
    # Report results
    print(f"\nExtraction complete.")
    print(f"Successfully extracted: {success_count} files")
    print(f"Failed to extract: {failure_count} files")
    
    # List some of the extracted files
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        png_files = [f for f in files if f.lower().endswith('.png')]
        print(f"\nFiles in output directory: {len(files)} total, {len(png_files)} PNG files")
        
        if png_files:
            print("Sample of extracted PNG files:")
            for png in png_files[:5]:  # Show first 5 files
                print(f"  - {png}")
            if len(png_files) > 5:
                print(f"  ... and {len(png_files) - 5} more")
        else:
            print("No PNG files found in the output directory.")
            if files:
                print("Other files found:")
                for f in files[:5]:
                    print(f"  - {f}")
    
    return success_count > 0

def main():
    print("Starting extraction of PNG files using paths from CSV file...")
    success = extract_images_from_zip_using_csv()
    
    if success:
        print("Extraction completed successfully.")
    else:
        print("Extraction failed. Please check the errors above.")

if __name__ == "__main__":
    main()