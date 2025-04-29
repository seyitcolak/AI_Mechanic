import os
import numpy as np
import soundfile as sf

def create_timing_belt_data(data_dir="car_sound_data", samples=5):
    """Create specific data for the timing belt/chain problems class."""
    print("Creating dummy data for timing belt/chain problems...")
    
    # Create the proper directory name
    class_dir = os.path.join(data_dir, "timing_belt_chain_problems")
    os.makedirs(class_dir, exist_ok=True)
    print(f"Creating directory: {class_dir}")
    
    # Class index for timing belt problems is 13
    class_idx = 13
    
    # Generate dummy audio files
    for i in range(samples):
        # Generate synthetic audio (white noise)
        sr = 22050
        duration = 2.0  # 2 seconds
        audio = np.random.randn(int(sr * duration))
        
        # Add some structure to make it distinctive
        time = np.arange(0, duration, 1/sr)
        freq = 100 + class_idx * 50  # Different frequency based on class
        sin_wave = 0.3 * np.sin(2 * np.pi * freq * time)
        audio = audio * 0.1 + sin_wave
        
        # Save the audio file
        file_path = os.path.join(class_dir, f"sample_{i+1}.wav")
        sf.write(file_path, audio, sr)
        print(f"Created {file_path}")
    
    print("Completed creating timing belt/chain problems data")

if __name__ == "__main__":
    create_timing_belt_data(samples=5) 