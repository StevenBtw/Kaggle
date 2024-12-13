import os
import subprocess
import sys
import tempfile
import shutil
import zipfile

def create_data_directory():
    """Create data directory if it doesn't exist."""
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    return data_dir

def download_and_extract_dataset():
    """Download the competition dataset to a temp directory and extract to data/."""
    try:
        data_dir = create_data_directory()
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            print("Downloading dataset...")
            zip_path = os.path.join(temp_dir, 'dataset.zip')
            
            # Download to temp directory
            subprocess.check_call([
                'kaggle', 'competitions', 'download',
                '-c', 'playground-series-s4e12',
                '-p', temp_dir
            ])
            
            # Find the downloaded zip file (in case the name is different)
            zip_files = [f for f in os.listdir(temp_dir) if f.endswith('.zip')]
            if not zip_files:
                raise FileNotFoundError("No zip file found in download directory")
            
            zip_path = os.path.join(temp_dir, zip_files[0])
            
            print(f"Extracting dataset to {data_dir}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            print("Dataset downloaded and extracted successfully!")
            
            # Temp directory and its contents will be automatically cleaned up
            
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_and_extract_dataset()
