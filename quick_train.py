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

# Define the model architecture (same as in car_diagnostic_app.py)
class CarSoundCNN(nn.Module):
    def __init__(self, num_classes=18):
        super(CarSoundCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 10 * 25, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Define issues
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
    def __init__(self, audio_files, labels):
        self.audio_files = audio_files
        self.labels = labels

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
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
            mfccs_tensor = torch.zeros((1, 40, 100), dtype=torch.float32)
            return mfccs_tensor, label

def find_audio_files(data_dir="car_sound_data"):
    """Find all audio files in the data directory structure."""
    audio_files = []
    labels = []
    
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
    
    return audio_files, labels

def main():
    print("Starting simplified training process...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find audio files
    audio_files, labels = find_audio_files("car_sound_data")
    print(f"Found {len(audio_files)} audio files across {len(set(labels))} classes")
    
    if len(audio_files) == 0:
        print("No audio files found! Please run create_dummy_data.py first.")
        return
    
    # Split into training and validation sets
    train_files, val_files, train_labels, val_labels = train_test_split(
        audio_files, labels, test_size=0.2, random_state=42, stratify=labels if len(set(labels)) > 1 else None
    )
    
    print(f"Training set: {len(train_files)}, Validation set: {len(val_files)}")
    
    # Create datasets and dataloaders
    train_dataset = CarSoundDataset(train_files, train_labels)
    val_dataset = CarSoundDataset(val_files, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    model = CarSoundCNN(num_classes=len(ISSUES))
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 5  # Reduced for quicker training
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        total_train = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, targets = inputs.to(device), torch.tensor(targets, dtype=torch.long).to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            train_acc += (predicted == targets).sum().item()
            train_loss += loss.item() * inputs.size(0)
        
        avg_train_loss = train_loss / total_train
        avg_train_acc = 100.0 * train_acc / total_train
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        total_val = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs, targets = inputs.to(device), torch.tensor(targets, dtype=torch.long).to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_val += targets.size(0)
                val_acc += (predicted == targets).sum().item()
                val_loss += loss.item() * inputs.size(0)
        
        avg_val_loss = val_loss / total_val
        avg_val_acc = 100.0 * val_acc / total_val
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.2f}%")
        
        # Save best model
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            print(f"Saving improved model with accuracy: {best_val_acc:.2f}%")
            torch.save(model.state_dict(), "car_sound_classifier.pt")
    
    print(f"Training complete! Final accuracy: {best_val_acc:.2f}%")
    print("Model saved as car_sound_classifier.pt")

if __name__ == "__main__":
    main() 