import numpy as np
import threading
import queue
import time
import random
from pyOpenBCI import OpenBCICyton
import serial
import sys
import os

# Add parent directory to Python path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PORT, CHANNEL_COUNT, SAMPLE_RATE, WINDOW_SIZE, OVERLAP

SCALE_FACTOR = (4500000) / 24 / (2 ** 23 - 1)  # µV/count for Cyton
BUFFER = []  # Will store tuples of (timestamp, data_array)

# Queue for passing epoched data to preprocessing
Queue1 = queue.Queue(maxsize=50)  # Adjust maxsize based on your needs

def process_window(window_data, timestamps):
    """
    Placeholder for your processing pipeline.
    'window_data' is a NumPy array of shape (WINDOW_SIZE, CHANNEL_COUNT)
    'timestamps' is a NumPy array of shape (WINDOW_SIZE,) with timestamp for each sample
    """
    print(f"Received window of shape: {window_data.shape}")
    print(f"Time range: {timestamps[0]:.3f} - {timestamps[-1]:.3f} seconds")
    print(f"Duration: {timestamps[-1] - timestamps[0]:.3f} seconds")
    
    # Example: Compute mean across time for each channel
    mean_per_channel = window_data.mean(axis=0)
    print("Mean per channel (µV):", mean_per_channel)

def epoch_and_send_data():
    """
    Check buffer and create epochs when enough data is available.
    Send epoched data to Queue1 for preprocessing.
    """
    global BUFFER
    
    while True:
        if len(BUFFER) >= WINDOW_SIZE:
            # Extract the most recent window (epoch)
            window_buffer = BUFFER[-WINDOW_SIZE:]
            
            # Separate timestamps and data
            timestamps = np.array([sample[0] for sample in window_buffer])
            window_data = np.array([sample[1] for sample in window_buffer])
            
            # Create epoched data structure with timestamps
            epoched_data = {
                'data': window_data.copy(),
                'timestamps': timestamps.copy(),
                'start_time': timestamps[0],
                'end_time': timestamps[-1],
                'duration': timestamps[-1] - timestamps[0],
                'sample_count': len(timestamps)
            }
            
            # Send epoched data to preprocessing via Queue1
            try:
                Queue1.put_nowait(epoched_data)
                print(f"Epoched data sent to Queue1. Shape: {window_data.shape}, Duration: {epoched_data['duration']:.3f}s")
            except queue.Full:
                print("Warning: Queue1 is full, dropping epoch")
            
            # Process window for monitoring (optional)
            process_window(window_data, timestamps)
            
            # Remove samples to achieve desired overlap
            keep = int(WINDOW_SIZE * OVERLAP)
            del BUFFER[:WINDOW_SIZE - keep]
        
        time.sleep(0.001)  # Small sleep to prevent busy waiting

def stream_callback(sample):
    """Callback for real OpenBCI data"""
    global BUFFER
    # Convert raw data to microvolts
    scaled_data = np.array(sample.channels_data[:CHANNEL_COUNT]) * SCALE_FACTOR
    timestamp = time.time()  # High precision timestamp
    BUFFER.append((timestamp, scaled_data))

def generate_dataset_data():
    """
    Read data from a dataset file and feed it into the acquisition pipeline.
    The file should be CSV-like, with each row containing channel values and a timestamp.
    """
    global BUFFER

    import os
    from datetime import datetime

    dataset_file = 'raw_data/0/Arithmetic/natural-1.txt'

    if not os.path.exists(dataset_file):
        print(f"Error: File {dataset_file} not found")
        return

    print(f"Reading data from {dataset_file}...")

    try:
        with open(dataset_file, 'r') as f:
            lines = f.readlines()

        channel_data = []
        timestamps = []

        for line in lines:
            if not line.strip():
                continue
            values = [v.strip() for v in line.strip().split(',')]
            try:
                # The last column is the timestamp string
                timestamp = datetime.strptime(values[-1], '%Y-%m-%d %H:%M:%S.%f')
                timestamps.append(timestamp)
                # The channel data is columns 1 to 1+CHANNEL_COUNT (skip index at 0)
                channel_values = [float(x) for x in values[1:1+CHANNEL_COUNT]]
                channel_data.append(channel_values)
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping malformed line: {e}")
                continue

        channel_data = np.array(channel_data)
        start_time = timestamps[0]
        relative_times = [(t - start_time).total_seconds() for t in timestamps]

        print(f"Successfully loaded {len(channel_data)} samples")
        print(f"Data shape: {channel_data.shape}")
        print(f"Time range: {relative_times[0]:.2f} to {relative_times[-1]:.2f} seconds")

        # Feed data into the acquisition pipeline
        for i in range(len(channel_data)):
            BUFFER.append((relative_times[i], channel_data[i]))
            if i < len(channel_data) - 1:
                time.sleep((relative_times[i+1] - relative_times[i]))

        print("Finished feeding data into acquisition pipeline")

    except Exception as e:
        print(f"Error processing file: {str(e)}")

def generate_random_data():
    """Generate random EEG-like data when OpenBCI is not available"""
    global BUFFER
    
    print("Generating random EEG-like data...")
    
    start_time = time.time()
    sample_count = 0
    
    while True:
        # Generate random data that mimics EEG signals
        # Typical EEG amplitude range: -100 to +100 µV
        random_sample = np.random.normal(0, 20, CHANNEL_COUNT)  # µV
        
        # Add some sinusoidal components to make it more EEG-like
        current_time = time.time()
        for i in range(CHANNEL_COUNT):
            # Add alpha waves (8-13 Hz) and some noise
            alpha_freq = 8 + i * 0.5  # Slightly different freq per channel
            random_sample[i] += 10 * np.sin(2 * np.pi * alpha_freq * current_time)
        
        # Store with precise timestamp
        BUFFER.append((current_time, random_sample))
        sample_count += 1
        
        # Sleep to simulate sampling rate (more precise timing)
        target_time = start_time + sample_count / SAMPLE_RATE
        sleep_time = target_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)

def try_connect_openbci():
    """Try to connect to OpenBCI Cyton"""
    try:
        print(f"Attempting to connect to OpenBCI on port {PORT}...")
        board = OpenBCICyton(port=PORT, daisy=False)
        print("OpenBCI Cyton connected successfully!")
        return board
    except Exception as e:
        print(f"Failed to connect to OpenBCI: {e}")
        return None

def main():
    print("Starting EEG Data Acquisition System...")
    
    # Start the epoching thread
    epoch_thread = threading.Thread(target=epoch_and_send_data, daemon=True)
    epoch_thread.start()
    print("Epoching thread started.")
    
    # Try to connect to OpenBCI
    board = try_connect_openbci()
    
    if board is not None:
        # Real OpenBCI data acquisition
        print("Starting OpenBCI data stream...")
        try:
            board.start_stream(stream_callback)
        except KeyboardInterrupt:
            print("Data acquisition stopped by user.")
            board.stop_stream()
            board.disconnect()
    else:
        # Fallback to random data generation
        print("Using random data generation as fallback...")
        random_thread = threading.Thread(target=generate_dataset_data, daemon=True)
        random_thread.start()
        
        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
                # Optional: Print queue status and timing info
                if len(BUFFER) > 0:
                    latest_timestamp = BUFFER[-1][0]
                    print(f"Queue1 size: {Queue1.qsize()}, Buffer size: {len(BUFFER)}, Latest timestamp: {latest_timestamp:.3f}")
        except KeyboardInterrupt:
            print("Random data generation stopped by user.")

def get_epoched_data(timeout=1.0):
    """
    Function for other modules to get epoched data from Queue1.
    Returns: Dictionary with 'data', 'timestamps', 'start_time', 'end_time', 'duration', 'sample_count'
             or None if timeout
    """
    try:
        return Queue1.get(timeout=timeout)
    except queue.Empty:
        return None


def get_latest_samples(n_samples=1):
    """
    Get the latest n samples from the buffer with timestamps.
    Returns: tuple of (timestamps_array, data_array) or None if not enough data
    """
    global BUFFER
    if len(BUFFER) >= n_samples:
        latest_samples = BUFFER[-n_samples:]
        timestamps = np.array([sample[0] for sample in latest_samples])
        data = np.array([sample[1] for sample in latest_samples])
        return timestamps, data
    return None

if __name__ == "__main__":
    main()