# AI Mechanic - Car Sound Diagnostic

An AI-powered car diagnostic tool that identifies potential car issues by analyzing engine sounds.

## Features

- Diagnoses 18 different car issues from audio recordings
- Web-based interface for easy upload or direct recording
- Real-time analysis and recommendations
- Mobile-friendly design

## Training the Model

The system uses a convolutional neural network (CNN) to classify car sounds into different problem categories. To train or retrain the model with your own data:

### Option 1: Using Real Car Sound Data

1. Organize your audio files in the following structure:
   ```
   car_sound_data/
   ├── engine_knocking/
   │   ├── sample1.wav
   │   ├── sample2.wav
   ├── belt_squealing/
   │   ├── sample1.wav
   │   └── ...
   ```

   (A folder for each issue category, with audio samples inside)

2. Run the training script:
   ```
   python train_model.py --data_dir car_sound_data
   ```

### Option 2: Generate Dummy Data for Testing

If you don't have real car sound data but want to test the system:

```
python train_model.py --create_dummy --data_dir car_sound_data
```

This will generate synthetic audio data for all 18 problem categories.

### Training Options

```
python train_model.py --help
```

Available options:
- `--data_dir`: Directory containing training data (default: "car_sound_data")
- `--model_path`: Where to save the trained model (default: "car_sound_classifier.pt")
- `--epochs`: Number of training epochs (default: 30)
- `--batch_size`: Batch size for training (default: 16)
- `--learning_rate`: Learning rate (default: 0.001)
- `--no_augment`: Disable data augmentation
- `--create_dummy`: Create dummy data for testing

## Running the Application

After training the model:

```
python car_diagnostic_app.py
```

Then open your browser to http://127.0.0.1:5000/

## Supported Car Issues

The system can diagnose the following car problems:

1. Engine knocking
2. Belt squealing
3. Brake issues
4. Normal operation (no issues)
5. Exhaust system problems
6. Transmission problems
7. Suspension issues
8. Wheel bearing problems
9. Power steering issues
10. Alternator problems
11. Fuel system issues
12. Turbocharger problems
13. AC compressor issues
14. Timing belt/chain problems
15. Catalytic converter issues
16. Starter motor problems
17. Unknown issue
18. Multiple issues detected

## Requirements

- Python 3.6+
- PyTorch
- Librosa
- Flask
- NumPy
- Matplotlib
- scikit-learn

Install dependencies with:
```
pip install torch librosa flask numpy matplotlib scikit-learn tqdm
```

## Setup Instructions

1. Install the required dependencies:
   ```
   pip install torch torchvision torchaudio numpy librosa flask scipy matplotlib sklearn tqdm soundfile
   ```

2. The project consists of the following files:
   - `car_diagnostic_app.py` - The main Flask application for diagnosing car sounds
   - `generate_training_data.py` - Script to generate synthetic car sound data for training
   - `train_model.py` - Script to train the neural network model on car sound data
   - `README.md` - This documentation file

## How to Use

### Step 1: Generate Training Data (Optional)

If you don't have real car sound recordings, you can generate synthetic training data:

```
python generate_training_data.py
```

The script will create a `data` directory with subdirectories for each car issue category, and populate them with synthetic audio files. You can specify the number of samples to generate for each category.

### Step 2: Train the Model

Train the neural network model on the available data:

```
python train_model.py
```

This will:
- Process the audio files in the `data` directory
- Train a CNN model to classify car sounds
- Save the trained model as `car_sound_classifier.pt`
- Generate a training metrics visualization as `training_metrics.png`

The enhanced training process includes:
- Early stopping to prevent overfitting
- Learning rate scheduling for better convergence
- Advanced visualization of training metrics

### Step 3: Run the Diagnostic App

Start the web application to diagnose car sounds:

```
python car_diagnostic_app.py
```

This will start a Flask server at http://127.0.0.1:5000. Open this URL in your browser to:
- Upload car sound recordings
- Record car sounds directly in the browser
- Get instant diagnoses of potential car issues with confidence scores
- Receive recommendations based on the diagnosis

## Features

### Enhanced User Interface
- Modern, responsive design that works on mobile devices
- Tabbed interface for both uploading and recording audio
- Real-time audio recording capability directly in the browser
- Loading indicators and clear result presentation

### Advanced Diagnostics
- Accurate classification of different car issues
- Confidence scoring to indicate the reliability of diagnosis
- Specific recommendations based on the detected issue
- User-friendly presentation of results

### Synthetic Data Generation
- Realistic synthetic engine sounds for training
- Customizable number of training samples
- Accurately simulates different car issues using acoustic principles
- Ensures balanced dataset for better model training

## Technical Details

### Model Architecture

The system uses a Convolutional Neural Network (CNN) that processes Mel-frequency cepstral coefficients (MFCCs) extracted from audio. The model architecture includes:

- 2 convolutional layers with ReLU activation and max pooling
- Fully connected layers for classification
- Softmax output providing confidence scores for each issue type

### Training Process

The enhanced training process:
- Extracts MFCC features from audio files
- Uses a train/validation split of 80/20
- Employs cross-entropy loss and Adam optimizer
- Implements learning rate scheduling with ReduceLROnPlateau
- Utilizes early stopping to prevent overfitting
- Saves the best-performing model based on validation accuracy
- Plots training/validation loss and accuracy curves

### Data Preprocessing

Audio files are:
- Loaded at a 22050Hz sample rate
- Converted to MFCC features (40 coefficients)
- Padded or truncated to a fixed length of 100 frames
- Normalized before feeding into the model

## Troubleshooting

- If you encounter `ModuleNotFoundError`, make sure all required packages are installed
- If the app reports missing model file, ensure you've run the training script first
- For issues with audio processing, check that your audio files are in WAV or MP3 format
- If browser recording doesn't work, ensure you've granted microphone permissions

## Future Improvements

- Add more car issue categories for broader diagnostic capabilities
- Implement transfer learning to improve accuracy with smaller datasets
- Add a detailed explanation of what causes each issue
- Develop a mobile app version for on-the-go diagnostics

## License

This project is provided for educational purposes. 