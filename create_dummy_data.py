import os
import numpy as np
import librosa
import soundfile as sf

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

def create_dummy_data(data_dir="car_sound_data", samples_per_class=3):
    """Create dummy data for testing if no real data is available."""
    print("Creating dummy training data for all 18 classes...")
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Create subdirectories for each class
    for class_idx, class_name in ISSUES.items():
        class_dir = os.path.join(data_dir, class_name.replace(" ", "_").lower())
        os.makedirs(class_dir, exist_ok=True)
        print(f"Creating directory: {class_dir}")
        
        # Generate dummy audio files per class
        for i in range(samples_per_class):
            # Generate synthetic audio (white noise)
            sr = 22050
            duration = 2.0  # 2 seconds
            audio = np.random.randn(int(sr * duration))
            
            # Add some structure to make it somewhat distinctive per class
            time = np.arange(0, duration, 1/sr)
            freq = 100 + class_idx * 50  # Different frequency per class
            sin_wave = 0.3 * np.sin(2 * np.pi * freq * time)
            audio = audio * 0.1 + sin_wave
            
            # Save the audio file
            file_path = os.path.join(class_dir, f"sample_{i+1}.wav")
            sf.write(file_path, audio, sr)
            print(f"Created {file_path}")
    
    print(f"Created dummy data with {len(ISSUES)} classes, {samples_per_class} samples per class")
    print(f"Data saved to {data_dir}")
    print("Now you can run: python train_model.py")

if __name__ == "__main__":
    create_dummy_data(samples_per_class=5)  # Create 5 samples per class 