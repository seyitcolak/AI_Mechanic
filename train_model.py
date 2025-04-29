import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import librosa
import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import argparse
import soundfile as sf

# Define the model architecture (same as in car_diagnostic_app.py)
class CarSoundCNN(nn.Module):
    def __init__(self, num_classes=18):
        super(CarSoundCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 10 * 25, 128)  # Adjusted for input (1, 1, 40, 100)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)  # Add dropout for regularization

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x

# Define issues matching the main application
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

class CarSoundDataset(Dataset):
    def __init__(self, audio_files, labels, transform=None):
        self.audio_files = audio_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        # Load and process the audio file
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=22050, mono=True)
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            
            # Pad or truncate to fixed length (100 frames)
            fixed_length = 100
            if mfccs.shape[1] < fixed_length:
                pad_width = fixed_length - mfccs.shape[1]
                mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mfccs = mfccs[:, :fixed_length]
            
            # Add channel dimension for CNN input
            mfccs = mfccs[np.newaxis, :, :]
            
            # Convert to PyTorch tensor
            mfccs_tensor = torch.from_numpy(mfccs).float()
            
            return mfccs_tensor, label
            
        except Exception as e:
            print(f"Error processing file {audio_path}: {e}")
            # Return a zero tensor and the same label if there's an error
            mfccs_tensor = torch.zeros((1, 40, 100), dtype=torch.float32)
            return mfccs_tensor, label

def find_audio_files(data_dir):
    """Find all audio files in the data directory structure."""
    audio_files = []
    labels = []
    
    # Check if the directory structure exists
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found!")
        return audio_files, labels
    
    # If each class has its own folder
    for class_idx, class_name in ISSUES.items():
        class_folder = os.path.join(data_dir, class_name.replace(" ", "_").lower())
        if os.path.exists(class_folder):
            for audio_file in glob.glob(os.path.join(class_folder, "*.wav")) + glob.glob(os.path.join(class_folder, "*.mp3")):
                audio_files.append(audio_file)
                labels.append(class_idx)
    
    # If using a flat directory structure with naming convention
    if len(audio_files) == 0:
        for audio_file in glob.glob(os.path.join(data_dir, "*.wav")) + glob.glob(os.path.join(data_dir, "*.mp3")):
            file_name = os.path.basename(audio_file)
            for class_idx, class_name in ISSUES.items():
                class_token = class_name.replace(" ", "_").lower()
                if class_token in file_name.lower():
                    audio_files.append(audio_file)
                    labels.append(class_idx)
                    break
    
    return audio_files, labels

def data_augmentation(audio_files, labels):
    """Augment the dataset with variations of existing files."""
    augmented_files = audio_files.copy()
    augmented_labels = labels.copy()
    
    # Only augment if we have actual data
    if len(audio_files) == 0:
        return augmented_files, augmented_labels
    
    print(f"Performing data augmentation on {len(audio_files)} files...")
    
    # Create a temporary directory for augmented files
    aug_dir = "augmented_audio"
    os.makedirs(aug_dir, exist_ok=True)
    
    # Generate augmented versions
    for i, (file_path, label) in enumerate(zip(audio_files, labels)):
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=22050, mono=True)
            
            # Skip very short audio files
            if len(audio) < sr * 0.5:  # Shorter than 0.5 seconds
                continue
                
            # Generate variations
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # 1. Pitch shift (up)
            shifted_audio_up = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
            shifted_path_up = os.path.join(aug_dir, f"{file_name}_shifted_up.wav")
            sf.write(shifted_path_up, shifted_audio_up, sr)
            augmented_files.append(shifted_path_up)
            augmented_labels.append(label)
            
            # 2. Pitch shift (down)
            shifted_audio_down = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2)
            shifted_path_down = os.path.join(aug_dir, f"{file_name}_shifted_down.wav")
            sf.write(shifted_path_down, shifted_audio_down, sr)
            augmented_files.append(shifted_path_down)
            augmented_labels.append(label)
            
            # 3. Time stretch (faster)
            stretched_audio_fast = librosa.effects.time_stretch(audio, rate=1.2)
            stretched_path_fast = os.path.join(aug_dir, f"{file_name}_stretched_fast.wav")
            sf.write(stretched_path_fast, stretched_audio_fast, sr)
            augmented_files.append(stretched_path_fast)
            augmented_labels.append(label)
            
            # 4. Time stretch (slower)
            stretched_audio_slow = librosa.effects.time_stretch(audio, rate=0.8)
            stretched_path_slow = os.path.join(aug_dir, f"{file_name}_stretched_slow.wav")
            sf.write(stretched_path_slow, stretched_audio_slow, sr)
            augmented_files.append(stretched_path_slow)
            augmented_labels.append(label)
            
            # 5. Add some noise
            noise_factor = 0.01
            noise = np.random.randn(len(audio))
            noisy_audio = audio + noise_factor * noise
            noisy_path = os.path.join(aug_dir, f"{file_name}_noisy.wav")
            sf.write(noisy_path, noisy_audio, sr)
            augmented_files.append(noisy_path)
            augmented_labels.append(label)
            
        except Exception as e:
            print(f"Error augmenting file {file_path}: {e}")
    
    print(f"Dataset expanded from {len(audio_files)} to {len(augmented_files)} examples")
    return augmented_files, augmented_labels

def train_model(data_dir, model_save_path, epochs=30, batch_size=16, learning_rate=0.001, augment=True):
    """Train the car sound classification model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find audio files
    print(f"Looking for audio files in {data_dir}...")
    audio_files, labels = find_audio_files(data_dir)
    
    if len(audio_files) == 0:
        print("No audio files found! Please add training data.")
        print("Expected folder structure:")
        print("data_directory/")
        print("├── engine_knocking/")
        print("│   ├── sample1.wav")
        print("│   ├── sample2.wav")
        print("├── belt_squealing/")
        print("│   ├── sample1.wav")
        print("│   └── ...")
        print("Or use naming convention like: engine_knocking_sample1.wav")
        return
    
    print(f"Found {len(audio_files)} audio files across {len(set(labels))} classes")
    
    # Augment data if required
    if augment:
        audio_files, labels = data_augmentation(audio_files, labels)
    
    # Split into training and validation sets
    train_files, val_files, train_labels, val_labels = train_test_split(
        audio_files, labels, test_size=0.2, random_state=42, stratify=labels if len(set(labels)) > 1 else None
    )
    
    print(f"Training set: {len(train_files)}, Validation set: {len(val_files)}")
    
    # Create datasets
    train_dataset = CarSoundDataset(train_files, train_labels)
    val_dataset = CarSoundDataset(val_files, val_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model, loss function, and optimizer
    model = CarSoundCNN(num_classes=len(ISSUES))
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100.0 * correct / total
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = 100.0 * correct / total
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print statistics
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save the model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Saving improved model to {model_save_path}")
            torch.save(model.state_dict(), model_save_path)
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png')
    plt.close()
    
    print(f"Training complete! Model saved to {model_save_path}")
    print(f"Training curves saved to training_curves.png")

def create_dummy_data(data_dir):
    """Create dummy data for testing if no real data is available."""
    print("Creating dummy training data for demonstration...")
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Create subdirectories for each class
    for class_idx, class_name in ISSUES.items():
        class_dir = os.path.join(data_dir, class_name.replace(" ", "_").lower())
        os.makedirs(class_dir, exist_ok=True)
        
        # Generate 3 dummy audio files per class
        for i in range(3):
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
            librosa.output.write_wav(file_path, audio, sr)
    
    print(f"Created dummy data with {len(ISSUES)} classes, 3 samples per class")
    print(f"Data saved to {data_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train car sound classification model")
    parser.add_argument("--data_dir", type=str, default="car_sound_data", 
                        help="Directory containing audio files grouped by class")
    parser.add_argument("--model_path", type=str, default="car_sound_classifier.pt",
                        help="Path to save the trained model")
    parser.add_argument("--epochs", type=int, default=30, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--no_augment", action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--create_dummy", action="store_true",
                        help="Create dummy data for testing if no real data exists")
    
    args = parser.parse_args()
    
    # Create dummy data if requested
    if args.create_dummy:
        create_dummy_data(args.data_dir)
    
    # Train the model
    train_model(
        data_dir=args.data_dir,
        model_save_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        augment=not args.no_augment
    ) 