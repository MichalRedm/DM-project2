import os
import requests
from zipfile import ZipFile

# Define the URLs and the target directories.
urls = {
    'ml-latest-small': 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
    'ml-latest': 'https://files.grouplens.org/datasets/movielens/ml-latest.zip'
}

# Base directory to extract files.
base_extract_dir = os.path.join("data", "raw")

# Create the base target directory if it doesn't exist.
os.makedirs(base_extract_dir, exist_ok=True)

for name, url in urls.items():
    local_zip_path = f'{name}.zip'

    # Create the specific target directory if it doesn't exist.
    os.makedirs(base_extract_dir, exist_ok=True)

    # Download the zip file.
    print(f"Downloading {name}...")
    response = requests.get(url)
    with open(local_zip_path, 'wb') as file:
        file.write(response.content)
    print(f"Download of {name} complete.")

    # Unzip the file.
    print(f"Unzipping {name}...")
    with ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(base_extract_dir)
    print(f"Files extracted to {base_extract_dir}")

    # Rename README.txt to README.md if it exists.
    readme_txt_path = os.path.join(base_extract_dir, name, 'README.txt')
    print(readme_txt_path, os.path.exists(readme_txt_path))
    readme_md_path = os.path.join(base_extract_dir, name, 'README.md')
    if os.path.exists(readme_txt_path):
        os.rename(readme_txt_path, readme_md_path)

    # Clean up: remove the downloaded zip file.
    os.remove(local_zip_path)
    print(f"Clean up complete. {name} zip file removed.")
