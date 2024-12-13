import os
import subprocess
import sys

def setup_kaggle_credentials():
    """Setup Kaggle API credentials."""
    # Create .kaggle directory if it doesn't exist
    kaggle_dir = os.path.expanduser('~/.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    
    # Move kaggle.json if it's in the root directory
    root_token = 'kaggle.json'
    if os.path.exists(root_token):
        target_path = os.path.join(kaggle_dir, 'kaggle.json')
        if not os.path.exists(target_path):
            print(f"Moving {root_token} to {target_path}")
            os.rename(root_token, target_path)
            os.chmod(target_path, 0o600)
            print("Kaggle credentials set up successfully")
        else:
            print("Kaggle credentials already exist in ~/.kaggle/")
    else:
        print("\nKaggle API credentials not found!")
        print("Please place your kaggle.json file in the root directory and run this script again")
        sys.exit(1)

def install_kaggle():
    """Install Kaggle CLI if not already installed."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        print("Kaggle CLI installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing Kaggle CLI: {e}")
        sys.exit(1)

def main():
    print("Setting up Kaggle CLI...")
    setup_kaggle_credentials()  # Move credentials first
    install_kaggle()           # Then install/import kaggle
    print("Kaggle setup complete!")

if __name__ == "__main__":
    main()
