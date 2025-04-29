import os
import requests
import re
import argparse
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import random
from tqdm import tqdm

"""
This script helps download car sound files from various online sources.
It contains a list of known sources for car sounds and creates the appropriate
directory structure for training the AI Mechanic model.
"""

# Define the issues matching our model
ISSUES = {
    0: "Engine knocking",
    1: "Belt squealing",
    2: "Brake issues",
    3: "Normal operation",
    4: "Exhaust system problems",
    5: "Transmission problems",
    6: "Suspension issues",
    7: "Wheel bearing problems",
    8: "Power steering issues",
    9: "Alternator problems",
    10: "Fuel system issues",
    11: "Turbocharger problems",
    12: "AC compressor issues",
    13: "Timing belt/chain problems",
    14: "Catalytic converter issues",
    15: "Starter motor problems",
    16: "Unknown issue",
    17: "Multiple issues detected"
}

# Sources for car sounds by category
SOUND_SOURCES = {
    "engine_knocking": [
        "https://www.youtube.com/results?search_query=engine+knocking+sound",
        "https://freesound.org/search/?q=engine+knocking",
        "https://www.soundsnap.com/search/audio/engine+knocking/score"
    ],
    "belt_squealing": [
        "https://www.youtube.com/results?search_query=car+belt+squealing",
        "https://freesound.org/search/?q=belt+squeal",
        "https://www.soundsnap.com/search/audio/belt+squeal/score"
    ],
    "brake_issues": [
        "https://www.youtube.com/results?search_query=brake+squeal+sound",
        "https://freesound.org/search/?q=brake+squeal",
        "https://www.soundsnap.com/search/audio/brake+squeal/score"
    ],
    "normal_operation": [
        "https://www.youtube.com/results?search_query=car+engine+idle+sound",
        "https://freesound.org/search/?q=car+engine+idle",
        "https://www.soundsnap.com/search/audio/car+engine+idle/score"
    ],
    "exhaust_system_problems": [
        "https://www.youtube.com/results?search_query=exhaust+leak+sound",
        "https://freesound.org/search/?q=exhaust+leak",
        "https://www.soundsnap.com/search/audio/exhaust+leak/score"
    ],
    "transmission_problems": [
        "https://www.youtube.com/results?search_query=transmission+noise",
        "https://freesound.org/search/?q=transmission+noise",
        "https://www.soundsnap.com/search/audio/transmission+noise/score"
    ],
    "suspension_issues": [
        "https://www.youtube.com/results?search_query=suspension+noise+car",
        "https://freesound.org/search/?q=suspension+noise",
        "https://www.soundsnap.com/search/audio/suspension+noise/score"
    ],
    "wheel_bearing_problems": [
        "https://www.youtube.com/results?search_query=wheel+bearing+noise",
        "https://freesound.org/search/?q=wheel+bearing",
        "https://www.soundsnap.com/search/audio/wheel+bearing/score"
    ],
    "power_steering_issues": [
        "https://www.youtube.com/results?search_query=power+steering+noise",
        "https://freesound.org/search/?q=power+steering",
        "https://www.soundsnap.com/search/audio/power+steering/score"
    ],
    "alternator_problems": [
        "https://www.youtube.com/results?search_query=alternator+noise",
        "https://freesound.org/search/?q=alternator+noise",
        "https://www.soundsnap.com/search/audio/alternator/score"
    ],
    "fuel_system_issues": [
        "https://www.youtube.com/results?search_query=fuel+pump+noise",
        "https://freesound.org/search/?q=fuel+pump",
        "https://www.soundsnap.com/search/audio/fuel+pump/score"
    ],
    "turbocharger_problems": [
        "https://www.youtube.com/results?search_query=turbo+whine+sound",
        "https://freesound.org/search/?q=turbo+whine",
        "https://www.soundsnap.com/search/audio/turbo+whine/score"
    ],
    "ac_compressor_issues": [
        "https://www.youtube.com/results?search_query=AC+compressor+noise",
        "https://freesound.org/search/?q=ac+compressor",
        "https://www.soundsnap.com/search/audio/ac+compressor/score"
    ],
    "timing_belt_chain_problems": [
        "https://www.youtube.com/results?search_query=timing+belt+noise",
        "https://freesound.org/search/?q=timing+belt",
        "https://www.soundsnap.com/search/audio/timing+belt/score"
    ],
    "catalytic_converter_issues": [
        "https://www.youtube.com/results?search_query=catalytic+converter+rattle",
        "https://freesound.org/search/?q=catalytic+converter",
        "https://www.soundsnap.com/search/audio/catalytic/score"
    ],
    "starter_motor_problems": [
        "https://www.youtube.com/results?search_query=car+starter+noise",
        "https://freesound.org/search/?q=starter+motor",
        "https://www.soundsnap.com/search/audio/starter+motor/score"
    ]
}

def setup_download_directories(base_dir="car_sound_data"):
    """Create directories for each issue category."""
    print("Setting up download directories...")
    
    # Create the base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Create subdirectories for each issue category
    for category in SOUND_SOURCES.keys():
        category_dir = os.path.join(base_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        print(f"Created directory: {category_dir}")
    
    return base_dir

def get_freesound_links(url):
    """Scrape audio links from Freesound.org."""
    print(f"Scanning {url} for audio files...")
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        audio_links = []
        
        # Look for sound elements
        sound_elements = soup.select('.sound_filename')
        for element in sound_elements:
            if 'href' in element.attrs:
                audio_links.append(urljoin("https://freesound.org", element['href']))
        
        print(f"Found {len(audio_links)} potential audio files")
        return audio_links
    
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return []

def download_audio_file(url, target_dir, file_name=None):
    """Download an audio file from a URL to the target directory."""
    try:
        # Generate a filename if not provided
        if not file_name:
            # Extract filename from URL or generate a random one
            url_filename = url.split('/')[-1]
            if '?' in url_filename:
                url_filename = url_filename.split('?')[0]
            
            if url_filename and '.' in url_filename and url_filename.split('.')[-1] in ['mp3', 'wav', 'ogg']:
                file_name = url_filename
            else:
                file_name = f"sound_{int(time.time())}_{random.randint(1000, 9999)}.mp3"
        
        # Ensure file has proper extension
        if not file_name.endswith(('.mp3', '.wav', '.ogg')):
            file_name += '.mp3'
        
        # Construct full path
        file_path = os.path.join(target_dir, file_name)
        
        # Don't download if file already exists
        if os.path.exists(file_path):
            print(f"File {file_name} already exists, skipping")
            return False
        
        # Download the file
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, stream=True)
        
        # Check if request was successful and content is an audio file
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            if 'audio' in content_type or url.endswith(('.mp3', '.wav', '.ogg')):
                # Get file size for progress bar
                total_size = int(response.headers.get('content-length', 0))
                
                # Download with progress bar for larger files
                with open(file_path, 'wb') as f:
                    if total_size > 1024*1024:  # If file is larger than 1MB
                        with tqdm(total=total_size, unit='B', unit_scale=True, desc=file_name) as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                    else:
                        f.write(response.content)
                
                print(f"Downloaded {file_name}")
                return True
            else:
                print(f"URL does not contain audio content: {url}")
                return False
        else:
            print(f"Failed to download {url}, status code: {response.status_code}")
            return False
    
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def get_youtube_audio_instructions():
    """Provide instructions for downloading YouTube audio."""
    instructions = """
To download car sound samples from YouTube:

1. Install youtube-dl or yt-dlp:
   pip install yt-dlp

2. Use it to download and extract audio from videos:
   yt-dlp -x --audio-format mp3 --audio-quality 0 <YouTube URL>

3. Place the downloaded MP3 files in the appropriate category folders:
   - engine_knocking/
   - belt_squealing/
   - brake_issues/
   etc.

4. YouTube search queries for each category have been provided in the script.
   """
    
    print(instructions)

def main():
    parser = argparse.ArgumentParser(description="Download car sound samples for AI Mechanic training")
    parser.add_argument('--data_dir', type=str, default='car_sound_data',
                        help='Directory to store downloaded sound files')
    args = parser.parse_args()
    
    # Setup directories
    base_dir = setup_download_directories(args.data_dir)
    
    # Provide YouTube instructions
    print("\n" + "="*80)
    get_youtube_audio_instructions()
    print("="*80 + "\n")
    
    # Display source links for each category
    print("The following search URLs can be used to find car sound samples:\n")
    
    for category, urls in SOUND_SOURCES.items():
        print(f"\n{category.replace('_', ' ').upper()}:")
        for url in urls:
            print(f"  - {url}")
    
    print("\n" + "="*80)
    print("IMPORTANT: When downloading audio samples, ensure you have the right")
    print("to use them for training, and respect copyright and licensing terms.")
    print("="*80 + "\n")
    
    # Ask if user wants to try automated downloading from FreeSounds
    try_download = input("Would you like to attempt automated downloading from Freesound.org? (y/n): ").strip().lower()
    
    if try_download == 'y':
        # Attempt to download from FreeSounds for each category
        for category, urls in SOUND_SOURCES.items():
            category_dir = os.path.join(base_dir, category)
            
            # Look for Freesound URLs
            for url in urls:
                if 'freesound.org' in url:
                    print(f"\nSearching {url} for {category} sounds...")
                    links = get_freesound_links(url)
                    
                    if links:
                        # Try to download up to 5 files per category
                        downloads = 0
                        for link in links[:10]:  # Limit to first 10 links
                            if download_audio_file(link, category_dir):
                                downloads += 1
                                if downloads >= 5:
                                    break
                            # Small delay to be nice to the server
                            time.sleep(1)
        
        print("\nAutomatic download completed. Check the folders for downloaded sounds.")
        print("You may need to manually download more samples for best results.")
    
    print("\nDone setting up directories and providing download information.")
    print(f"Place your car sound samples in the appropriate folders within {base_dir}/")
    print("Then run train_model.py to train the model.")

if __name__ == "__main__":
    main() 