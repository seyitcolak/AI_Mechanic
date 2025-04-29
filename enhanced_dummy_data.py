import os
import numpy as np
import soundfile as sf
import random
from scipy import signal
import urllib.request
import zipfile
import io
import shutil

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

def add_engine_background(duration=3.0, sr=22050, rpm=None):
    """Create a basic engine background noise with optional RPM setting."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Base engine frequency around 30-80 Hz (or specified by RPM)
    if rpm is None:
        base_freq = random.uniform(30, 80)
    else:
        # RPM / 60 = Hz (frequency of one revolution per second)
        # We divide by 2 for a four-stroke engine where the firing frequency is half the rotation frequency
        base_freq = rpm / 60 / 2
    
    # Create more complex harmonics
    harmonics = []
    for i in range(1, 8):  # Increased from 6 to 8 harmonics
        freq = base_freq * i
        # More realistic falloff for higher harmonics
        amplitude = 1.0 / (i ** (1.2 + random.uniform(0, 0.6)))
        phase = random.uniform(0, 2 * np.pi)
        harmonic = amplitude * np.sin(2 * np.pi * freq * t + phase)
        harmonics.append(harmonic)
    
    # Combine harmonics
    engine = np.sum(harmonics, axis=0)
    
    # Add some random fluctuations
    fluctuation = np.sin(2 * np.pi * 0.5 * t) * 0.1  # Slow fluctuation
    fast_fluctuation = np.sin(2 * np.pi * 2.0 * t) * 0.05  # Faster fluctuation
    engine = engine * (1 + fluctuation + fast_fluctuation)
    
    # Add slight randomness to make each sample unique
    unique_pattern = np.sin(2 * np.pi * random.uniform(0.1, 0.3) * t) * random.uniform(0.02, 0.08)
    engine = engine * (1 + unique_pattern)
    
    # Normalize
    engine = engine / np.max(np.abs(engine)) * 0.7
    
    # Add some colored noise (more realistic than white noise for engines)
    # Pink noise has more energy at lower frequencies
    white_noise = np.random.randn(len(t))
    # Create pink noise using a simple filter
    b, a = signal.butter(1, 0.2, btype='lowpass')
    pink_noise = signal.lfilter(b, a, white_noise) * 0.07
    engine = engine + pink_noise
    
    return engine

def create_engine_knocking(samples=40, duration=3.0, sr=22050):
    """Create realistic engine knocking sounds with increased variety."""
    print(f"Creating {samples} enhanced engine knocking samples...")
    
    data_dir = "car_sound_data"
    class_dir = os.path.join(data_dir, "engine_knocking")
    os.makedirs(class_dir, exist_ok=True)
    
    # Define different types of knocking patterns
    knock_types = [
        "light_detonation",     # Light, occasional knocking
        "severe_detonation",    # Heavy, consistent knocking
        "piston_slap",          # Looser, more hollow sound
        "rod_knock",            # Lower frequency, more serious knock
        "carbon_buildup"        # Random, less rhythmic tapping
    ]
    
    for i in range(samples):
        # Choose a type of knock
        knock_type = random.choice(knock_types)
        
        # Choose an RPM level for context
        rpm = random.choice([800, 1200, 1800, 2500, 3000, 3500])
        
        # Base engine sound
        audio = add_engine_background(duration, sr, rpm)
        
        # Create knocking sound
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        
        # Special parameters based on knock type
        if knock_type == "light_detonation":
            # Fewer, lighter knocks at higher frequencies
            num_knocks = random.randint(8, 15)
            freq_range = (1800, 3000)
            intensity_range = (0.2, 0.4)
            knock_duration_range = (0.008, 0.015)
            
        elif knock_type == "severe_detonation":
            # Many strong knocks
            num_knocks = random.randint(25, 40)
            freq_range = (1500, 2800)
            intensity_range = (0.4, 0.8)
            knock_duration_range = (0.01, 0.025)
            
        elif knock_type == "piston_slap":
            # Hollow, looser sound with lower frequency
            num_knocks = random.randint(15, 25)
            freq_range = (800, 1500)
            intensity_range = (0.3, 0.6)
            knock_duration_range = (0.015, 0.03)
            # Add a slight echo effect for piston slap
            echo_delay = int(0.01 * sr)  # 10ms delay
            echo_level = 0.3
            
        elif knock_type == "rod_knock":
            # Lower frequency, consistent, serious knock
            num_knocks = random.randint(20, 35)
            freq_range = (500, 1200)
            intensity_range = (0.5, 0.8)
            knock_duration_range = (0.02, 0.04)
            
        else:  # carbon_buildup
            # Random, less rhythmic tapping
            num_knocks = random.randint(5, 15)
            freq_range = (1200, 2200)
            intensity_range = (0.2, 0.5)
            knock_duration_range = (0.01, 0.02)
        
        # Calculate interval between knocks based on RPM
        # Higher RPM = more frequent knocks
        if rpm < 1500:
            knock_interval = random.uniform(0.15, 0.25)
        elif rpm < 2500:
            knock_interval = random.uniform(0.08, 0.15)
        else:
            knock_interval = random.uniform(0.04, 0.08)
        
        # Generate more realistic knock timings based on engine cycle
        if knock_type in ["light_detonation", "severe_detonation"]:
            # These types of knock are more rhythmic, tied to combustion cycle
            base_time = random.uniform(0, knock_interval)
            knock_times = [base_time + j * knock_interval for j in range(num_knocks)]
            # Add slight variation to timing
            knock_times = [kt + random.uniform(-0.01, 0.01) for kt in knock_times]
            # Ensure knocks are within audio duration
            knock_times = [kt for kt in knock_times if 0.1 < kt < duration-0.1]
        else:
            # Less rhythmic knocking patterns
            knock_times = sorted(random.uniform(0.1, duration-0.1) for _ in range(num_knocks))
        
        # Generate knocks with varying intensity
        for knock_time in knock_times:
            # Find the index corresponding to the knock time
            idx = int(knock_time * sr)
            
            # Create a short, sharp pulse (metallic knocking sound)
            knock_duration = random.uniform(*knock_duration_range)
            knock_samples = int(knock_duration * sr)
            
            # Create a multi-frequency knock for realism
            knock_t = np.linspace(0, knock_duration, knock_samples)
            knock_freq_main = random.uniform(*freq_range)
            
            # Primary knock component
            knock = np.sin(2 * np.pi * knock_freq_main * knock_t)
            
            # Add upper harmonics
            for harm in range(2, 5):
                harm_amp = 1.0 / (harm * harm)
                knock += harm_amp * np.sin(2 * np.pi * knock_freq_main * harm * knock_t)
            
            # Add some broadband noise for metallic character
            noise = np.random.randn(len(knock_t)) * 0.1
            sos = signal.butter(4, [knock_freq_main*0.7, knock_freq_main*1.5], 'bandpass', fs=sr, output='sos')
            filtered_noise = signal.sosfilt(sos, noise)
            knock = knock + filtered_noise
            
            # Apply envelope to create a sharp attack and decay
            # Different envelope shapes for different knock types
            if knock_type in ["rod_knock", "piston_slap"]:
                # Slower decay for these mechanical knocks
                attack = np.linspace(0, 1, int(knock_samples * 0.1))
                decay = np.exp(-3 * np.linspace(0, 1, knock_samples - len(attack)))
                envelope = np.concatenate([attack, decay])
            else:
                # Sharper attack/decay for detonation knocks
                envelope = np.exp(-5 * knock_t)
                
            knock = knock * envelope[:len(knock)]
            
            # Scale the knock to a random intensity within the appropriate range
            intensity = random.uniform(*intensity_range)
            knock = knock * intensity
            
            # Add the knock to the audio
            end_idx = min(idx + knock_samples, len(audio))
            audio[idx:end_idx] = audio[idx:end_idx] + knock[:end_idx-idx]
            
            # Add echo effect for piston slap
            if knock_type == "piston_slap" and idx + knock_samples + echo_delay < len(audio):
                audio[idx+echo_delay:idx+echo_delay+end_idx-idx] += knock[:end_idx-idx] * echo_level
        
        # Ensure the audio doesn't clip
        audio = audio / max(1.0, np.max(np.abs(audio)))
        
        # Save the audio file
        file_path = os.path.join(class_dir, f"enhanced_knocking_{knock_type}_{rpm}rpm_{i+1}.wav")
        sf.write(file_path, audio, sr)
        print(f"Created {file_path}")
    
    return samples

def create_belt_squealing(samples=15, duration=3.0, sr=22050):
    """Create realistic belt squealing sounds."""
    print(f"Creating {samples} enhanced belt squealing samples...")
    
    data_dir = "car_sound_data"
    class_dir = os.path.join(data_dir, "belt_squealing")
    os.makedirs(class_dir, exist_ok=True)
    
    for i in range(samples):
        # Base engine sound at lower volume
        audio = add_engine_background(duration, sr) * 0.3
        
        # Create squealing sound
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        
        # Belt squeal - high frequency with fluctuations
        squeal_freq = random.uniform(2000, 4000)  # High squeal frequency
        
        # Create frequency modulation for the squeal
        mod_freq = random.uniform(4, 10)  # Modulation frequency in Hz
        mod_depth = random.uniform(50, 150)  # Modulation depth
        freq_mod = squeal_freq + mod_depth * np.sin(2 * np.pi * mod_freq * t)
        
        # Instantaneous phase by integrating frequency
        phase = np.cumsum(2 * np.pi * freq_mod / sr)
        
        # Generate the squeal
        squeal = np.sin(phase)
        
        # Create amplitude modulation for realistic pulsating
        am_freq = random.uniform(10, 20)
        am = 0.5 + 0.5 * np.sin(2 * np.pi * am_freq * t)
        squeal = squeal * am
        
        # Apply random intermittence - squeal isn't constant
        for j in range(random.randint(2, 5)):
            start_idx = random.randint(0, int(duration * sr) - int(0.5 * sr))
            end_idx = min(start_idx + random.randint(int(0.3 * sr), int(1.0 * sr)), int(duration * sr))
            mask = np.zeros(int(duration * sr))
            mask[start_idx:end_idx] = np.linspace(0, 1, end_idx-start_idx) * np.linspace(1, 0, end_idx-start_idx)
            squeal = squeal * mask
        
        # Mix squeal with engine at appropriate level
        squeal_level = random.uniform(0.3, 0.6)
        audio = audio + squeal * squeal_level
        
        # Ensure the audio doesn't clip
        audio = audio / max(1.0, np.max(np.abs(audio)))
        
        # Save the audio file
        file_path = os.path.join(class_dir, f"enhanced_squealing_{i+1}.wav")
        sf.write(file_path, audio, sr)
        print(f"Created {file_path}")
    
    return samples

def create_brake_issues(samples=15, duration=3.0, sr=22050):
    """Create realistic brake issue sounds."""
    print(f"Creating {samples} enhanced brake issue samples...")
    
    data_dir = "car_sound_data"
    class_dir = os.path.join(data_dir, "brake_issues")
    os.makedirs(class_dir, exist_ok=True)
    
    for i in range(samples):
        # Base noise (much quieter for brakes)
        audio = add_engine_background(duration, sr) * 0.1
        
        # Create brake squeal/grinding
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        
        # Decide between squeal and grinding
        brake_type = random.choice(["squeal", "grinding"])
        
        if brake_type == "squeal":
            # High-pitched brake squeal
            squeal_freq = random.uniform(4000, 8000)
            squeal = np.sin(2 * np.pi * squeal_freq * t)
            
            # Add frequency wobble
            mod_freq = random.uniform(5, 15)
            phase_mod = 0.2 * np.sin(2 * np.pi * mod_freq * t)
            squeal = np.sin(2 * np.pi * squeal_freq * t + phase_mod)
            
            # Create an envelope for the squeal
            envelope = np.ones_like(t)
            # Apply random fading in/out to make it more realistic
            fade_points = sorted(random.sample(range(len(t)), 5))
            for j in range(len(fade_points)-1):
                start = fade_points[j]
                end = fade_points[j+1]
                if j % 2 == 0:  # Alternate between fade in and fade out
                    envelope[start:end] = np.linspace(0.1, 1.0, end-start)
                else:
                    envelope[start:end] = np.linspace(1.0, 0.1, end-start)
            
            squeal = squeal * envelope * random.uniform(0.4, 0.7)
            audio = audio + squeal
            
        else:  # grinding
            # Create brake grinding sound (broadband noise with resonances)
            noise = np.random.randn(len(t)) * 0.5
            
            # Apply bandpass filtering to simulate metal-on-metal grinding
            sos = signal.butter(10, [800, 3000], 'bandpass', fs=sr, output='sos')
            grinding = signal.sosfilt(sos, noise)
            
            # Add resonant peaks at random frequencies
            for _ in range(3):
                freq = random.uniform(1000, 2500)
                q = random.uniform(30, 50)  # Q factor (sharpness of resonance)
                sos_peak = signal.butter(2, [freq-freq/q, freq+freq/q], 'bandpass', fs=sr, output='sos')
                resonance = signal.sosfilt(sos_peak, noise) * 2.0
                grinding = grinding + resonance
            
            # Create a pulsating effect
            pulse_freq = random.uniform(2, 5)  # Wheel rotation frequency
            pulse = 0.5 + 0.5 * np.cos(2 * np.pi * pulse_freq * t)
            grinding = grinding * pulse * random.uniform(0.3, 0.6)
            
            audio = audio + grinding
        
        # Ensure the audio doesn't clip
        audio = audio / max(1.0, np.max(np.abs(audio)))
        
        # Save the audio file
        file_path = os.path.join(class_dir, f"enhanced_brakes_{i+1}.wav")
        sf.write(file_path, audio, sr)
        print(f"Created {file_path}")
    
    return samples

def create_normal_operation(samples=15, duration=3.0, sr=22050):
    """Create realistic normal engine operation sounds."""
    print(f"Creating {samples} enhanced normal operation samples...")
    
    data_dir = "car_sound_data"
    class_dir = os.path.join(data_dir, "normal_operation")
    os.makedirs(class_dir, exist_ok=True)
    
    for i in range(samples):
        # Just create clean engine noise with slight variations
        audio = add_engine_background(duration, sr)
        
        # Add subtle resonances that don't indicate issues
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        
        # Add slight rpm variations for realism
        rpm_var = np.sin(2 * np.pi * 0.3 * t) * 0.05  # Slow, gentle rpm changes
        audio = audio * (1 + rpm_var)
        
        # Add random mild resonances that don't indicate problems
        for _ in range(2):
            freq = random.uniform(100, 600)  # Lower frequencies for normal resonances
            res_amp = random.uniform(0.05, 0.1)  # Low amplitude
            resonance = res_amp * np.sin(2 * np.pi * freq * t)
            audio = audio + resonance
        
        # Ensure the audio doesn't clip
        audio = audio / max(1.0, np.max(np.abs(audio)))
        
        # Save the audio file
        file_path = os.path.join(class_dir, f"enhanced_normal_{i+1}.wav")
        sf.write(file_path, audio, sr)
        print(f"Created {file_path}")
    
    return samples

def download_engine_knock_samples():
    """
    Download real engine knock samples from online sources to supplement synthetic data.
    Returns number of files downloaded.
    """
    print("Attempting to download real engine knocking samples...")
    
    # Create directory for downloaded samples
    data_dir = "car_sound_data"
    download_dir = os.path.join(data_dir, "engine_knocking", "downloaded")
    os.makedirs(download_dir, exist_ok=True)
    
    # List of URLs to freesound.org or similar free sound repositories
    # These are example URLs and should be replaced with real URLs to engine knocking sounds
    sound_urls = [
        "https://freesound.org/data/previews/256/256552_3263906-lq.mp3",  # Example: engine knock sound
        "https://freesound.org/data/previews/442/442358_8300117-lq.mp3",  # Example: metallic knocking
        "https://freesound.org/data/previews/369/369158_6898081-lq.mp3"   # Example: engine sound
    ]
    
    downloaded_count = 0
    
    for i, url in enumerate(sound_urls):
        try:
            print(f"Downloading sample from {url}")
            # Create a file name for the downloaded sound
            file_name = f"downloaded_knock_{i+1}.wav"
            output_path = os.path.join(download_dir, file_name)
            
            # Download the file
            urllib.request.urlretrieve(url, output_path)
            downloaded_count += 1
            
            print(f"Successfully downloaded to {output_path}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    
    print(f"Downloaded {downloaded_count} real engine knock samples")
    return downloaded_count

def create_enhanced_dummy_data():
    """Create enhanced dummy data for key categories with improved variety."""
    print("Creating enhanced dummy training data...")
    
    # Create enhanced data for the most important categories
    total_samples = 0
    
    # Generate more engine knocking samples with greater variety
    total_samples += create_engine_knocking(samples=40)
    
    # Try to download some real samples to complement synthetic data
    try:
        total_samples += download_engine_knock_samples()
    except Exception as e:
        print(f"Could not download real samples: {e}")
        print("Continuing with synthetic data only")
    
    # Create other needed sounds
    total_samples += create_belt_squealing(samples=20)
    total_samples += create_brake_issues(samples=20)
    total_samples += create_normal_operation(samples=20)
    
    print(f"Created {total_samples} enhanced audio samples for training")
    print("You can now run the training script to improve the model.")

if __name__ == "__main__":
    create_enhanced_dummy_data() 