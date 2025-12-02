#!/usr/bin/env python3
"""
Download script for PatentsView bulk data files.

Downloads the following TSV files required for the patent information retrieval system:
- g_cpc_current.tsv (CPC classifications)
- g_patent_abstract.tsv (patent abstracts)
- g_us_patent_citation.tsv (patent citations)
- g_application.tsv (application data including filing dates)

Data source: PatentsView bulk data download
https://patentsview.org/download/data-download-tables
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile
import gzip
import shutil

# Base URL for PatentsView bulk data downloads
PATENTSVIEW_BASE_URL = "https://s3.amazonaws.com/data.patentsview.org/download"

# Files to download (filename -> local directory)
FILES_TO_DOWNLOAD = {
    "g_cpc_current.tsv.zip": "Patent Data/g_cpc_current.tsv",
    "g_patent_abstract.tsv.zip": "Patent Data/g_patent_abstract.tsv",
    "g_us_patent_citation.tsv.zip": "Patent Data/g_us_patent_citation.tsv",
    "g_application.tsv.zip": "Patent Data/g_application.tsv",
}


def download_file(url, destination_path, show_progress=True):
    """
    Download a file from a URL to a destination path with progress bar.
    
    Args:
        url (str): URL to download from
        destination_path (Path): Path to save the downloaded file
        show_progress (bool): Whether to show progress bar
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(destination_path, 'wb') as f:
            if show_progress and total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination_path.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}", file=sys.stderr)
        return False


def extract_zip(zip_path, extract_dir):
    """
    Extract a ZIP file to a directory.
    
    Args:
        zip_path (Path): Path to the ZIP file
        extract_dir (Path): Directory to extract to
    """
    print(f"Extracting {zip_path.name}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"✓ Extracted to {extract_dir}")
        return True
    except zipfile.BadZipFile as e:
        print(f"Error extracting {zip_path}: {e}", file=sys.stderr)
        return False


def extract_gzip(gz_path, extract_path):
    """
    Extract a GZIP file.
    
    Args:
        gz_path (Path): Path to the .gz file
        extract_path (Path): Path to save the extracted file
    """
    print(f"Extracting {gz_path.name}...")
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(extract_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"✓ Extracted to {extract_path}")
        return True
    except Exception as e:
        print(f"Error extracting {gz_path}: {e}", file=sys.stderr)
        return False


def main():
    """Download and extract PatentsView bulk data files."""
    
    # Get the base directory (where this script is located)
    base_dir = Path(__file__).parent.resolve()
    
    print("=" * 80)
    print("PatentsView Bulk Data Downloader")
    print("=" * 80)
    print(f"Working directory: {base_dir}\n")
    
    # Check if requests library is available
    try:
        import requests
    except ImportError:
        print("Error: 'requests' library not found. Install it with:")
        print("  pip install requests")
        sys.exit(1)
    
    # Check if tqdm is available (optional, for progress bars)
    try:
        import tqdm
        show_progress = True
    except ImportError:
        print("Note: 'tqdm' library not found. Install it for progress bars:")
        print("  pip install tqdm")
        print()
        show_progress = False
    
    # Download each file
    for filename, target_dir in FILES_TO_DOWNLOAD.items():
        url = f"{PATENTSVIEW_BASE_URL}/{filename}"
        target_path = base_dir / target_dir
        target_path.mkdir(parents=True, exist_ok=True)
        
        download_path = target_path / filename
        
        print(f"\n📥 Downloading {filename}...")
        print(f"   URL: {url}")
        print(f"   Destination: {target_path}")
        
        # Check if already downloaded
        if download_path.exists():
            print(f"   ℹ File already exists: {download_path}")
            user_input = input("   Redownload? [y/N]: ").strip().lower()
            if user_input not in ['y', 'yes']:
                print("   Skipping download.")
            else:
                if download_file(url, download_path, show_progress):
                    print(f"   ✓ Downloaded successfully")
                else:
                    print(f"   ✗ Download failed", file=sys.stderr)
                    continue
        else:
            if download_file(url, download_path, show_progress):
                print(f"   ✓ Downloaded successfully")
            else:
                print(f"   ✗ Download failed", file=sys.stderr)
                continue
        
        # Extract the file
        if filename.endswith('.zip'):
            if extract_zip(download_path, target_path):
                # Clean up the zip file after extraction
                print(f"   Removing {filename}...")
                download_path.unlink()
        elif filename.endswith('.gz'):
            extracted_name = filename[:-3]  # Remove .gz extension
            extract_path = target_path / extracted_name
            if extract_gzip(download_path, extract_path):
                # Clean up the gz file after extraction
                print(f"   Removing {filename}...")
                download_path.unlink()
    
    print("\n" + "=" * 80)
    print("✓ Download complete!")
    print("=" * 80)
    print("\nDownloaded files are located in the 'Patent Data' directory.")
    print("You can now run data_filtering.py to process the patent data.")


if __name__ == "__main__":
    main()

