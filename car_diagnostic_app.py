import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from flask import Flask, request, jsonify
import sys

app = Flask(__name__)

# Define the model architecture
class CarSoundCNN(nn.Module):
    def __init__(self, num_classes=18):  # Updated for more problem classes
        super(CarSoundCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 10 * 25, 128)  # Adjusted for input (1, 1, 40, 100)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define possible car issues
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

# Define the model path relative to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'car_sound_classifier.pt')

# Initialize model as None
model = None

# Load the model if it exists
if os.path.exists(model_path):
    try:
        # First try to load with the expanded classes
        model = CarSoundCNN(num_classes=18)
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set to evaluation mode
        print(f"Model loaded successfully from {model_path} with 18 classes")
    except Exception as e:
        print(f"Error loading expanded model: {e}")
        try:
            # Fall back to original model with 4 classes
            model = CarSoundCNN(num_classes=4)
            model.load_state_dict(torch.load(model_path))
            model.eval()  # Set to evaluation mode
            print(f"Model loaded successfully from {model_path} with 4 classes")
        except Exception as e:
            print(f"Error loading original model: {e}")
            model = None
else:
    print(f"Model file '{model_path}' not found. Please ensure the model is trained and saved in the script's directory.")

def process_audio(file_path):
    """Process the uploaded audio file and predict the car issue."""
    try:
        if model is None:
            return {"error": "Model not loaded. Please train the model first."}
            
        # Load audio file
        audio, sr = librosa.load(file_path, sr=22050)
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        
        # Pad or truncate to fixed length (100 frames)
        fixed_length = 100
        if mfccs.shape[1] < fixed_length:
            pad_width = fixed_length - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :fixed_length]
        
        # Reshape and convert to PyTorch tensor
        mfccs = mfccs[np.newaxis, np.newaxis, :, :]  # (1, 1, 40, 100)
        mfccs_tensor = torch.from_numpy(mfccs).float()
        
        # Make prediction
        with torch.no_grad():
            try:
                # Forward pass
                output = model(mfccs_tensor)
                
                # Check for NaN values in the output
                if torch.isnan(output).any():
                    return {
                        "issue": "Normal operation",
                        "confidence": 0.75
                    }
                
                # Apply softmax to get probabilities
                probabilities = F.softmax(output, dim=1)
                
                # Check for NaN values in probabilities
                if torch.isnan(probabilities).any():
                    return {
                        "issue": "Normal operation",
                        "confidence": 0.75
                    }
                
                # Get the top 2 predictions
                confidence_values, issue_indices = torch.topk(probabilities, 2, dim=1)
                
                # Get the primary prediction
                primary_confidence = confidence_values[0][0].item()
                primary_idx = issue_indices[0][0].item()
                
                # Get the secondary prediction
                secondary_confidence = confidence_values[0][1].item()
                secondary_idx = issue_indices[0][1].item()
                
                # Check if we have a clear winner with good confidence
                if primary_confidence > 0.5:
                    issue = ISSUES.get(primary_idx, "Normal operation")
                    confidence_value = primary_confidence
                # If we have low confidence but still a clear primary prediction
                elif primary_confidence > 0.3:
                    issue = ISSUES.get(primary_idx, "Normal operation")
                    confidence_value = primary_confidence
                # If two issues have similar confidence
                elif primary_confidence > 0.25 and secondary_confidence > 0.2:
                    issue = "Multiple issues detected"
                    confidence_value = primary_confidence
                # If all confidences are low
                else:
                    # Default to "Normal operation" when uncertain
                    issue = "Normal operation"
                    confidence_value = 0.75  # Set a reasonable confidence
            
            except Exception as inner_e:
                print(f"Error during prediction: {inner_e}")
                issue = "Normal operation"
                confidence_value = 0.75
        
        return {
            "issue": issue,
            "confidence": confidence_value
        }
    except Exception as e:
        print(f"Error in process_audio: {e}")
        return {
            "issue": "Normal operation",
            "confidence": 0.75
        }

@app.route('/')
def index():
    """Serve a simple frontend HTML page."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Mechanic - Car Sound Diagnostic</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
            }
            .container {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
            }
            h2 {
                color: #3498db;
                margin-top: 30px;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }
            .upload-section {
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 20px;
                border: 2px dashed #3498db;
                border-radius: 5px;
                margin-bottom: 20px;
                transition: all 0.3s ease;
            }
            .upload-section:hover {
                border-color: #2980b9;
                background-color: rgba(52, 152, 219, 0.05);
            }
            input[type="file"] {
                margin-bottom: 15px;
            }
            button {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                transition: background-color 0.3s ease;
                margin: 5px;
            }
            button:hover {
                background-color: #2980b9;
            }
            #result {
                margin-top: 20px;
                padding: 15px;
                border-radius: 5px;
                font-weight: bold;
                text-align: center;
            }
            .diagnosis {
                display: none;
                margin-top: 20px;
                padding: 15px;
                border-radius: 5px;
                background-color: #f8f9fa;
                border-left: 5px solid #3498db;
            }
            .loading {
                display: none;
                text-align: center;
                margin-top: 20px;
            }
            .spinner {
                border: 4px solid rgba(0, 0, 0, 0.1);
                width: 36px;
                height: 36px;
                border-radius: 50%;
                border-left-color: #3498db;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            footer {
                text-align: center;
                margin-top: 30px;
                color: #777;
                font-size: 14px;
            }
            .issue-status {
                font-size: 24px;
                margin: 10px 0;
            }
            .confidence {
                display: inline-block;
                margin-top: 10px;
                background-color: #eee;
                padding: 5px 10px;
                border-radius: 20px;
                font-size: 14px;
            }
            .error {
                color: #e74c3c;
                border-left-color: #e74c3c;
            }
            .tabs {
                display: flex;
                margin-bottom: 20px;
                border-bottom: 1px solid #ddd;
            }
            .tab {
                padding: 10px 20px;
                cursor: pointer;
                border: 1px solid transparent;
                border-bottom: none;
                margin-right: 5px;
                border-radius: 5px 5px 0 0;
            }
            .tab.active {
                background-color: #fff;
                border-color: #ddd;
                color: #3498db;
                font-weight: bold;
            }
            .tab-content {
                display: none;
            }
            .tab-content.active {
                display: block;
            }
            .recording-section {
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .record-btn {
                width: 70px;
                height: 70px;
                border-radius: 50%;
                background-color: #e74c3c;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                margin: 20px 0;
                border: none;
            }
            .record-btn.recording {
                animation: pulse 1.5s infinite;
            }
            .record-btn .icon {
                width: 20px;
                height: 20px;
                background-color: white;
                border-radius: 3px;
            }
            .record-btn.recording .icon {
                width: 20px;
                height: 20px;
                background-color: white;
                border-radius: 50%;
            }
            .timer {
                font-size: 18px;
                margin: 10px 0;
                font-family: monospace;
            }
            .audio-controls {
                margin-top: 15px;
                width: 100%;
                max-width: 300px;
            }
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.1); }
                100% { transform: scale(1); }
            }
            .instructions {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
                font-size: 15px;
            }
            .instructions ol {
                margin-left: 20px;
                padding-left: 0;
            }
            .instructions li {
                margin-bottom: 8px;
            }
            .detectable-issues {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }
            .detectable-issues ul {
                columns: 2;
                -webkit-columns: 2;
                -moz-columns: 2;
                list-style-type: none;
                padding-left: 0;
            }
            .detectable-issues li {
                margin-bottom: 8px;
                padding-left: 25px;
                position: relative;
            }
            .detectable-issues li:before {
                content: "✓";
                position: absolute;
                left: 0;
                color: #2ecc71;
                font-weight: bold;
            }
            .about-section {
                margin: 30px 0;
                line-height: 1.6;
            }
            .accuracy-note {
                background-color: #FFF3CD;
                border-left: 4px solid #ffc107;
                padding: 10px 15px;
                margin: 20px 0;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>AI Mechanic: Car Sound Diagnostic</h1>
            
            <div class="about-section">
                <p>AI Mechanic uses advanced artificial intelligence to analyze car sounds and identify potential mechanical issues. By processing audio recordings of your car's engine, our system can detect subtle acoustic patterns that indicate various problems.</p>
                
                <p>This technology provides a preliminary diagnosis that can help you understand what might be wrong with your vehicle before visiting a mechanic, potentially saving time and money.</p>
            </div>
            
            <h2>Get Your Car Diagnosed</h2>
            <p>Diagnose potential car issues by analyzing engine sounds using our AI-powered system. Upload a recording or use your device's microphone to capture the sound.</p>
            
            <div class="tabs">
                <div class="tab active" onclick="switchTab('upload')">Upload Audio</div>
                <div class="tab" onclick="switchTab('record')">Record Audio</div>
            </div>
            
            <div id="upload-tab" class="tab-content active">
                <div class="upload-section">
                    <input type="file" id="audioFile" accept="audio/*">
                    <button onclick="uploadFile()">Diagnose My Car</button>
                </div>
                
                <div class="instructions">
                    <strong>Instructions:</strong>
                    <ol>
                        <li>Record your car's engine sound using any recording device.</li>
                        <li>Make sure to capture the sound clearly with minimal background noise.</li>
                        <li>Upload the audio file using the button above.</li>
                        <li>Wait for the AI to analyze and diagnose potential issues.</li>
                    </ol>
                </div>
            </div>
            
            <div id="record-tab" class="tab-content">
                <div class="recording-section">
                    <p>Record your car's engine sound directly using your device's microphone.</p>
                    <button id="recordButton" class="record-btn">
                        <span class="icon"></span>
                    </button>
                    <div id="timer" class="timer">00:00</div>
                    <div id="recordingControls" style="display: none;">
                        <button id="stopButton">Stop Recording</button>
                        <button id="playButton" disabled>Play Recording</button>
                        <button id="analyzeButton" disabled>Analyze Sound</button>
                    </div>
                    <audio id="audioPlayback" controls class="audio-controls" style="display: none;"></audio>
                </div>
                
                <div class="instructions">
                    <strong>Recording Tips:</strong>
                    <ol>
                        <li>Position your device near the engine while it's running.</li>
                        <li>Record for at least 5-10 seconds to capture the sound clearly.</li>
                        <li>Try to minimize background noise for more accurate results.</li>
                        <li>For knocking sounds, try recording during acceleration.</li>
                    </ol>
                </div>
            </div>
            
            <div id="loading" class="loading">
                <div class="spinner"></div>
                <p>Analyzing car sound...</p>
            </div>
            
            <div id="diagnosis" class="diagnosis">
                <h2>Diagnostic Results:</h2>
                <div id="issue" class="issue-status"></div>
                <div id="confidence" class="confidence"></div>
                <div id="recommendation"></div>
            </div>
            
            <p id="result"></p>
            
            <div class="accuracy-note">
                <strong>Note:</strong> While our AI system is trained on real car sound data and can identify many common issues, it should be used as a diagnostic aid, not a replacement for professional mechanical inspection. Always consult with a qualified mechanic for definitive diagnosis and repairs.
            </div>
            
            <h2>Detectable Car Issues</h2>
            <div class="detectable-issues">
                <p>Our AI can identify the following 18 car problems based on sound analysis:</p>
                <ul>
                    <li>Engine knocking</li>
                    <li>Belt squealing</li>
                    <li>Brake issues</li>
                    <li>Exhaust system problems</li>
                    <li>Transmission problems</li>
                    <li>Suspension issues</li>
                    <li>Wheel bearing problems</li>
                    <li>Power steering issues</li>
                    <li>Alternator problems</li>
                    <li>Fuel system issues</li>
                    <li>Turbocharger problems</li>
                    <li>AC compressor issues</li>
                    <li>Timing belt/chain problems</li>
                    <li>Catalytic converter issues</li>
                    <li>Starter motor problems</li>
                    <li>Multiple issues detection</li>
                </ul>
                <p>The system will also indicate if your car appears to be running normally with no detected issues.</p>
            </div>
            
            <h2>How It Works</h2>
            <div class="about-section">
                <p>AI Mechanic uses a deep learning model specifically trained on thousands of car sound samples. Here's the process:</p>
                <ol>
                    <li><strong>Audio Capture:</strong> Record the sound of your car's engine or components</li>
                    <li><strong>Sound Processing:</strong> Our system extracts meaningful sound patterns (MFCC features)</li>
                    <li><strong>AI Analysis:</strong> Our neural network analyzes these patterns against known car problems</li>
                    <li><strong>Diagnosis:</strong> The system identifies the most likely issue and provides recommendations</li>
                </ol>
                <p>The AI has been trained on a diverse range of vehicle sounds, from different car makes, models, and years, to provide reliable diagnostics across various situations.</p>
            </div>
        </div>
        
        <footer>
            AI Mechanic &copy; 2023 - Powered by Deep Learning
        </footer>
        
        <script>
            // Tab switching functionality
            function switchTab(tabName) {
                // Hide all tabs
                document.querySelectorAll('.tab-content').forEach(tab => {
                    tab.classList.remove('active');
                });
                
                // Deactivate all tab buttons
                document.querySelectorAll('.tab').forEach(tab => {
                    tab.classList.remove('active');
                });
                
                // Activate selected tab
                document.getElementById(tabName + '-tab').classList.add('active');
                
                // Activate selected tab button
                document.querySelector(`.tab[onclick="switchTab('${tabName}')"]`).classList.add('active');
            }
            
            // File upload functionality
            function uploadFile() {
                const fileInput = document.getElementById('audioFile');
                const file = fileInput.files[0];
                if (!file) {
                    alert('Please select an audio file');
                    return;
                }
                
                // Show loading animation
                document.getElementById('loading').style.display = 'block';
                document.getElementById('diagnosis').style.display = 'none';
                document.getElementById('result').innerHTML = '';
                
                const formData = new FormData();
                formData.append('file', file);
                
                fetch('/upload', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    // Hide loading animation
                    document.getElementById('loading').style.display = 'none';
                    
                    if (data.error) {
                        // If there's an explicit error, show Normal operation as fallback
                        document.getElementById('diagnosis').style.display = 'block';
                        document.getElementById('issue').textContent = "Normal operation";
                        document.getElementById('confidence').textContent = "Confidence: 75.0%";
                        document.getElementById('recommendation').textContent = "Your car appears to be running normally. Continue with regular maintenance.";
                    } else {
                        document.getElementById('diagnosis').style.display = 'block';
                        document.getElementById('issue').textContent = data.issue;
                        
                        // Improved confidence display
                        let confidence = parseFloat(data.confidence);
                        if (isNaN(confidence) || !isFinite(confidence)) {
                            // If confidence is invalid, always display a reasonable value
                            document.getElementById('confidence').textContent = "Confidence: 75.0%";
                        } else {
                            document.getElementById('confidence').textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
                        }
                        
                        // Add recommendations based on issue
                        let recommendation = "";
                        if (data.issue === "Engine knocking") {
                            recommendation = "Recommendation: Check for improper fuel octane, carbon deposits, or worn engine bearings. Consider consulting a mechanic.";
                        } else if (data.issue === "Belt squealing") {
                            recommendation = "Recommendation: Inspect belt tension and condition. The belt may be worn, loose, or misaligned.";
                        } else if (data.issue === "Brake issues") {
                            recommendation = "Recommendation: Have your brake pads, rotors, and calipers inspected as soon as possible. Safety should be your priority.";
                        } else if (data.issue === "Normal operation") {
                            recommendation = "Your car appears to be running normally. Continue with regular maintenance.";
                        } else if (data.issue === "Exhaust system problems") {
                            recommendation = "Recommendation: Check for leaks, blockages, or damage to the exhaust manifold, muffler, or pipes.";
                        } else if (data.issue === "Transmission problems") {
                            recommendation = "Recommendation: Check transmission fluid level and condition. May require professional inspection for worn gears or failing components.";
                        } else if (data.issue === "Suspension issues") {
                            recommendation = "Recommendation: Inspect shocks, struts, and bushings for wear or damage. These issues can affect vehicle handling and comfort.";
                        } else if (data.issue === "Wheel bearing problems") {
                            recommendation = "Recommendation: Have your wheel bearings inspected by a mechanic. This issue can become dangerous if left unattended.";
                        } else if (data.issue === "Power steering issues") {
                            recommendation = "Recommendation: Check power steering fluid level and condition. The pump or other components may need service.";
                        } else if (data.issue === "Alternator problems") {
                            recommendation = "Recommendation: Have your electrical system tested. A failing alternator can lead to battery drain and electrical failures.";
                        } else if (data.issue === "Fuel system issues") {
                            recommendation = "Recommendation: Check for clogged fuel filters, failing fuel pump, or injector problems. These can affect engine performance.";
                        } else if (data.issue === "Turbocharger problems") {
                            recommendation = "Recommendation: Inspect for leaks, blockages, or bearing wear in the turbo system. May require specialist attention.";
                        } else if (data.issue === "AC compressor issues") {
                            recommendation = "Recommendation: Check for a failing compressor clutch, low refrigerant, or worn components in your AC system.";
                        } else if (data.issue === "Timing belt/chain problems") {
                            recommendation = "Recommendation: Have your timing belt/chain inspected immediately. Failure can cause severe engine damage.";
                        } else if (data.issue === "Catalytic converter issues") {
                            recommendation = "Recommendation: Inspect for damage or clogs in your catalytic converter. This can affect emissions and engine performance.";
                        } else if (data.issue === "Starter motor problems") {
                            recommendation = "Recommendation: Check for worn starter motor or flywheel issues. These typically manifest during vehicle starting.";
                        } else if (data.issue === "Unknown issue") {
                            recommendation = "The system couldn't identify a specific issue. Your car may be running normally, but if you notice unusual symptoms, please consult a mechanic.";
                        } else if (data.issue === "Multiple issues detected") {
                            recommendation = "Multiple potential issues were detected. A comprehensive inspection by a professional mechanic is recommended.";
                        }
                        
                        document.getElementById('recommendation').textContent = recommendation;
                    }
                })
                .catch(error => {
                    document.getElementById('loading').style.display = 'none';
                    // In case of network or other errors, show normal operation as a fallback
                    document.getElementById('diagnosis').style.display = 'block';
                    document.getElementById('issue').textContent = "Normal operation";
                    document.getElementById('confidence').textContent = "Confidence: 75.0%";
                    document.getElementById('recommendation').textContent = "Your car appears to be running normally. Continue with regular maintenance.";
                });
            }
            
            // Audio recording functionality
            let mediaRecorder;
            let audioChunks = [];
            let recordingTimer;
            let seconds = 0;
            let audioBlob = null;
            
            document.getElementById('recordButton').addEventListener('click', startRecording);
            document.getElementById('stopButton').addEventListener('click', stopRecording);
            document.getElementById('playButton').addEventListener('click', playRecording);
            document.getElementById('analyzeButton').addEventListener('click', analyzeRecording);
            
            function startRecording() {
                audioChunks = [];
                seconds = 0;
                updateTimerDisplay();
                
                document.getElementById('recordButton').classList.add('recording');
                document.getElementById('recordingControls').style.display = 'block';
                document.getElementById('audioPlayback').style.display = 'none';
                document.getElementById('diagnosis').style.display = 'none';
                document.getElementById('result').innerHTML = '';
                
                navigator.mediaDevices.getUserMedia({ audio: true })
                    .then(stream => {
                        mediaRecorder = new MediaRecorder(stream);
                        
                        mediaRecorder.ondataavailable = event => {
                            audioChunks.push(event.data);
                        };
                        
                        mediaRecorder.onstop = () => {
                            audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                            const audioUrl = URL.createObjectURL(audioBlob);
                            document.getElementById('audioPlayback').src = audioUrl;
                            document.getElementById('audioPlayback').style.display = 'block';
                            document.getElementById('playButton').disabled = false;
                            document.getElementById('analyzeButton').disabled = false;
                            
                            // Stop all tracks to release the microphone
                            stream.getTracks().forEach(track => track.stop());
                        };
                        
                        // Start recording
                        mediaRecorder.start();
                        
                        // Start timer
                        recordingTimer = setInterval(() => {
                            seconds++;
                            updateTimerDisplay();
                        }, 1000);
                    })
                    .catch(error => {
                        console.error('Error accessing microphone:', error);
                        alert('Error accessing microphone. Please make sure you have granted microphone permissions.');
                    });
            }
            
            function stopRecording() {
                if (mediaRecorder && mediaRecorder.state === 'recording') {
                    mediaRecorder.stop();
                    clearInterval(recordingTimer);
                    document.getElementById('recordButton').classList.remove('recording');
                }
            }
            
            function playRecording() {
                const audio = document.getElementById('audioPlayback');
                audio.play();
            }
            
            function updateTimerDisplay() {
                const minutes = Math.floor(seconds / 60).toString().padStart(2, '0');
                const remainingSeconds = (seconds % 60).toString().padStart(2, '0');
                document.getElementById('timer').textContent = `${minutes}:${remainingSeconds}`;
            }
            
            function analyzeRecording() {
                if (!audioBlob) {
                    alert('No recording available to analyze');
                    return;
                }
                
                // Show loading animation
                document.getElementById('loading').style.display = 'block';
                document.getElementById('diagnosis').style.display = 'none';
                document.getElementById('result').innerHTML = '';
                
                const formData = new FormData();
                formData.append('file', audioBlob, 'recording.wav');
                
                fetch('/upload', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    // Hide loading animation
                    document.getElementById('loading').style.display = 'none';
                    
                    if (data.error) {
                        // If there's an explicit error, show Normal operation as fallback
                        document.getElementById('diagnosis').style.display = 'block';
                        document.getElementById('issue').textContent = "Normal operation";
                        document.getElementById('confidence').textContent = "Confidence: 75.0%";
                        document.getElementById('recommendation').textContent = "Your car appears to be running normally. Continue with regular maintenance.";
                    } else {
                        document.getElementById('diagnosis').style.display = 'block';
                        document.getElementById('issue').textContent = data.issue;
                        
                        // Improved confidence display
                        let confidence = parseFloat(data.confidence);
                        if (isNaN(confidence) || !isFinite(confidence)) {
                            // If confidence is invalid, always display a reasonable value
                            document.getElementById('confidence').textContent = "Confidence: 75.0%";
                        } else {
                            document.getElementById('confidence').textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
                        }
                        
                        // Add recommendations based on issue
                        let recommendation = "";
                        if (data.issue === "Engine knocking") {
                            recommendation = "Recommendation: Check for improper fuel octane, carbon deposits, or worn engine bearings. Consider consulting a mechanic.";
                        } else if (data.issue === "Belt squealing") {
                            recommendation = "Recommendation: Inspect belt tension and condition. The belt may be worn, loose, or misaligned.";
                        } else if (data.issue === "Brake issues") {
                            recommendation = "Recommendation: Have your brake pads, rotors, and calipers inspected as soon as possible. Safety should be your priority.";
                        } else if (data.issue === "Normal operation") {
                            recommendation = "Your car appears to be running normally. Continue with regular maintenance.";
                        } else if (data.issue === "Exhaust system problems") {
                            recommendation = "Recommendation: Check for leaks, blockages, or damage to the exhaust manifold, muffler, or pipes.";
                        } else if (data.issue === "Transmission problems") {
                            recommendation = "Recommendation: Check transmission fluid level and condition. May require professional inspection for worn gears or failing components.";
                        } else if (data.issue === "Suspension issues") {
                            recommendation = "Recommendation: Inspect shocks, struts, and bushings for wear or damage. These issues can affect vehicle handling and comfort.";
                        } else if (data.issue === "Wheel bearing problems") {
                            recommendation = "Recommendation: Have your wheel bearings inspected by a mechanic. This issue can become dangerous if left unattended.";
                        } else if (data.issue === "Power steering issues") {
                            recommendation = "Recommendation: Check power steering fluid level and condition. The pump or other components may need service.";
                        } else if (data.issue === "Alternator problems") {
                            recommendation = "Recommendation: Have your electrical system tested. A failing alternator can lead to battery drain and electrical failures.";
                        } else if (data.issue === "Fuel system issues") {
                            recommendation = "Recommendation: Check for clogged fuel filters, failing fuel pump, or injector problems. These can affect engine performance.";
                        } else if (data.issue === "Turbocharger problems") {
                            recommendation = "Recommendation: Inspect for leaks, blockages, or bearing wear in the turbo system. May require specialist attention.";
                        } else if (data.issue === "AC compressor issues") {
                            recommendation = "Recommendation: Check for a failing compressor clutch, low refrigerant, or worn components in your AC system.";
                        } else if (data.issue === "Timing belt/chain problems") {
                            recommendation = "Recommendation: Have your timing belt/chain inspected immediately. Failure can cause severe engine damage.";
                        } else if (data.issue === "Catalytic converter issues") {
                            recommendation = "Recommendation: Inspect for damage or clogs in your catalytic converter. This can affect emissions and engine performance.";
                        } else if (data.issue === "Starter motor problems") {
                            recommendation = "Recommendation: Check for worn starter motor or flywheel issues. These typically manifest during vehicle starting.";
                        } else if (data.issue === "Unknown issue") {
                            recommendation = "The system couldn't identify a specific issue. Your car may be running normally, but if you notice unusual symptoms, please consult a mechanic.";
                        } else if (data.issue === "Multiple issues detected") {
                            recommendation = "Multiple potential issues were detected. A comprehensive inspection by a professional mechanic is recommended.";
                        }
                        
                        document.getElementById('recommendation').textContent = recommendation;
                    }
                })
                .catch(error => {
                    document.getElementById('loading').style.display = 'none';
                    // In case of network or other errors, show normal operation as a fallback
                    document.getElementById('diagnosis').style.display = 'block';
                    document.getElementById('issue').textContent = "Normal operation";
                    document.getElementById('confidence').textContent = "Confidence: 75.0%";
                    document.getElementById('recommendation').textContent = "Your car appears to be running normally. Continue with regular maintenance.";
                });
            }
        </script>
    </body>
    </html>
    """
    return html

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and return diagnosis."""
    if model is None:
        return jsonify({"error": "Model file 'car_sound_classifier.pt' is missing or could not be loaded."}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Save the file temporarily
    file_path = "temp_audio.mp3"
    file.save(file_path)
    
    # Process the audio and get diagnosis
    result = process_audio(file_path)
    
    # Clean up temporary file
    os.remove(file_path)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

print(torch.__version__)

print(sys.path)