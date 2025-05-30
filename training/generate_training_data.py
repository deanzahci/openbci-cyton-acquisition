import numpy as np
import os
import random
from datetime import datetime
from scipy.signal import butter, filtfilt

# Import filtering functions from dataFiltering.py
from dataFiltering import butter_bandpass_filter, scale_data

# Constants
CHANNEL_COUNT = 8
SAMPLE_RATE = 255
WINDOW_SIZE = 255  # 1 second of data at 255 Hz

# Dataset paths organized by concentration level
DATASETS = {
    0: [  # Natural state
        'raw_data/0/Arithmetic/natural-1.txt',
        'raw_data/0/Arithmetic/natural-2.txt',
        'raw_data/0/Arithmetic/natural-3.txt',
        'raw_data/0/Arithmetic/natural-4.txt',
        'raw_data/0/Arithmetic/natural-5.txt',
        'raw_data/0/Arithmetic/natural-6.txt',
        'raw_data/0/Arithmetic/natural-7.txt',
        'raw_data/0/Arithmetic/natural-8.txt',
        'raw_data/0/Arithmetic/natural-9.txt',
        'raw_data/0/Arithmetic/natural-10.txt',
        'raw_data/0/Arithmetic/natural-11.txt',
        'raw_data/0/Arithmetic/natural-12.txt',
        'raw_data/0/Arithmetic/natural-13.txt',
        'raw_data/0/Arithmetic/natural-14.txt',
        'raw_data/0/Arithmetic/natural-15.txt'
    ],
    1: [  # Low concentration
        'raw_data/1/Arithmetic/lowlevel-1.txt',
        'raw_data/1/Arithmetic/lowlevel-2.txt',
        'raw_data/1/Arithmetic/lowlevel-3.txt',
        'raw_data/1/Arithmetic/lowlevel-4.txt',
        'raw_data/1/Arithmetic/lowlevel-5.txt',
        'raw_data/1/Arithmetic/lowlevel-6.txt',
        'raw_data/1/Arithmetic/lowlevel-7.txt',
        'raw_data/1/Arithmetic/lowlevel-8.txt'
    ],
    2: [  # Medium concentration
        'raw_data/2/Arithmetic/midlevel-1.txt',
        'raw_data/2/Arithmetic/midlevel-2.txt',
        'raw_data/2/Arithmetic/midlevel-3.txt',
        'raw_data/2/Arithmetic/midlevel-4.txt',
        'raw_data/2/Arithmetic/midlevel-5.txt',
        'raw_data/2/Arithmetic/midlevel-6.txt',
        'raw_data/2/Arithmetic/midlevel-7.txt',
        'raw_data/2/Arithmetic/midlevel-8.txt'
    ],
    3: [  # High concentration
        'raw_data/3/Arithmetic/highlevel-1.txt',
        'raw_data/3/Arithmetic/highlevel-2.txt',
        'raw_data/3/Arithmetic/highlevel-3.txt',
        'raw_data/3/Arithmetic/highlevel-4.txt',
        'raw_data/3/Arithmetic/highlevel-5.txt',
        'raw_data/3/Arithmetic/highlevel-6.txt',
        'raw_data/3/Arithmetic/highlevel-7.txt',
        'raw_data/3/Arithmetic/highlevel-8.txt'
    ]
}

def load_data_from_file(file_path):
    """Load data from a dataset file."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return None

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        channel_data = []
        timestamps = []

        for line in lines:
            if not line.strip():
                continue
            values = [v.strip() for v in line.strip().split(',')]
            try:
                timestamp = datetime.strptime(values[-1], '%Y-%m-%d %H:%M:%S.%f')
                timestamps.append(timestamp)
                channel_values = [float(x) for x in values[1:1+CHANNEL_COUNT]]
                channel_data.append(channel_values)
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping malformed line: {e}")
                continue

        return np.array(channel_data), timestamps

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def extract_random_epoch(data, timestamps, window_size=WINDOW_SIZE):
    """Extract a random epoch from the data."""
    if len(data) < window_size:
        return None, None
    
    # Choose a random starting point
    start_idx = random.randint(0, len(data) - window_size)
    
    # Extract the epoch
    epoch_data = data[start_idx:start_idx + window_size]
    epoch_timestamps = timestamps[start_idx:start_idx + window_size]
    
    return epoch_data, epoch_timestamps

def process_epoch(epoch_data):
    """Process an epoch with the same filtering as in the main code."""
    # Scale the data
    scaled_data = scale_data(epoch_data)
    
    # Apply bandpass filter (1-30 Hz)
    filtered_data = butter_bandpass_filter(scaled_data, 1, 30, SAMPLE_RATE)
    
    return filtered_data

def generate_training_data(epochs_per_class=100):
    """Generate training data from random epochs."""
    training_data = []
    training_labels = []
    
    for concentration_level, file_paths in DATASETS.items():
        print(f"\nGenerating data for concentration level {concentration_level}...")
        epochs_generated = 0
        
        while epochs_generated < epochs_per_class:
            # Randomly select a file
            file_path = random.choice(file_paths)
            
            # Load data from file
            result = load_data_from_file(file_path)
            if result is None:
                continue
                
            data, timestamps = result
            
            # Extract random epoch
            epoch_data, epoch_timestamps = extract_random_epoch(data, timestamps)
            if epoch_data is None:
                continue
            
            # Process the epoch
            #processed_epoch = process_epoch(epoch_data)
            
            # Add to training data
            training_data.append(epoch_data)
            training_labels.append(concentration_level)
            
            epochs_generated += 1
            if epochs_generated % 10 == 0:
                print(f"Generated {epochs_generated}/{epochs_per_class} epochs")
    
    return np.array(training_data), np.array(training_labels)

def save_training_data(data, labels, filename='training_data.txt'):
    """Save the training data and labels to a text file."""
    with open(filename, 'w') as f:
        # Write header with data shape and label distribution
        f.write(f"# Data shape: {data.shape}\n")
        f.write(f"# Labels shape: {labels.shape}\n")
        f.write("# Label distribution:\n")
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            f.write(f"# Level {label}: {count} epochs\n")
        f.write("#\n")
        f.write("# Format: [label] [channel1_data] [channel2_data] ... [channel8_data]\n")
        f.write("# Each channel data is a space-separated list of 255 values\n")
        f.write("#\n")
        
        # Write data
        for i, (epoch, label) in enumerate(zip(data, labels)):
            # Write label
            f.write(f"{label}")
            
            # Write each channel's data
            for channel in range(CHANNEL_COUNT):
                f.write(" " + " ".join(f"{x:.6f}" for x in epoch[:, channel]))
            
            f.write("\n")
            
            # Print progress
            if (i + 1) % 100 == 0:
                print(f"Saved {i + 1} epochs...")
    
    print(f"\nSaved training data to {filename}")
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    print("\nLabel distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"Level {label}: {count} epochs")

if __name__ == "__main__":
    # Generate training data
    print("Generating training data...")
    training_data, training_labels = generate_training_data(epochs_per_class=100)
    
    # Save the data
    save_training_data(training_data, training_labels) 