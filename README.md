# OpenBCI Cyton Data Acquisition

A Python-based real-time EEG data acquisition system for the OpenBCI Cyton board. This system provides continuous data streaming, epoching, and preprocessing capabilities with automatic fallback to simulated data when hardware is unavailable.

## Features

- Real-time EEG data acquisition from OpenBCI Cyton (8 channels)
- Automatic data scaling to microvolts (µV)
- Windowed data epoching with configurable overlap
- Thread-safe data buffering and queue management
- Fallback to simulated EEG-like data when hardware is unavailable
- High-precision timestamping for each sample
- Configurable sampling parameters

## Hardware Requirements

- OpenBCI Cyton board (8-channel)
- USB-to-Serial adapter or compatible connection
- Computer with available USB port

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd openbci-cyton-acquisition
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Update the serial port in `config.py` to match your system:
```python
PORT = '/dev/tty.usbserial-XXXXXXXX'  # Update with your port
```

## Configuration

Edit [`config.py`](config.py) to customize acquisition parameters:

- `PORT`: Serial port for OpenBCI connection
- `CHANNEL_COUNT`: Number of EEG channels (default: 8)
- `SAMPLE_RATE`: Sampling frequency in Hz (default: 250)
- `WINDOW_SIZE`: Epoch size in samples (default: 250, equivalent to 1 second)
- `OVERLAP`: Overlap between consecutive epochs (default: 0.5 for 50% overlap)

## Usage

### Basic Usage

Run the acquisition system:
```bash
python acquisition.py
```

The system will attempt to connect to the OpenBCI Cyton board. If connection fails, it automatically switches to simulated data generation.

### Integration with Other Modules

The system provides two main functions for external integration:

#### Get Epoched Data
```python
from acquisition import get_epoched_data

# Get the next available epoch (blocks until data is available)
epoch = get_epoched_data(timeout=1.0)
if epoch:
    data = epoch['data']          # Shape: (WINDOW_SIZE, CHANNEL_COUNT)
    timestamps = epoch['timestamps']  # Timestamp for each sample
    duration = epoch['duration']     # Epoch duration in seconds
```

#### Get Latest Samples
```python
from acquisition import get_latest_samples

# Get the most recent 10 samples
timestamps, data = get_latest_samples(n_samples=10)
if timestamps is not None:
    # Process the latest samples
    print(f"Latest data shape: {data.shape}")
```

## Data Format

### Epoched Data Structure
Each epoch returned by `get_epoched_data()` contains:
- `data`: NumPy array of shape (WINDOW_SIZE, CHANNEL_COUNT) in µV
- `timestamps`: NumPy array of timestamps for each sample
- `start_time`: Epoch start timestamp
- `end_time`: Epoch end timestamp
- `duration`: Epoch duration in seconds
- `sample_count`: Number of samples in the epoch

### Data Scaling
Raw OpenBCI data is automatically converted to microvolts (µV) using the Cyton's scaling factor.

## System Architecture

The system uses a multi-threaded architecture:

1. **Main Thread**: Handles OpenBCI connection and user interaction
2. **Data Acquisition Thread**: Collects samples from OpenBCI or generates simulated data
3. **Epoching Thread**: Processes buffered data into epochs and manages the output queue

## Troubleshooting

### Connection Issues
- Verify the correct serial port in [`config.py`](config.py)
- Ensure OpenBCI board is powered and properly connected
- Check that no other applications are using the serial port

### Performance Issues
- Adjust `Queue1` maxsize in [`acquisition.py`](acquisition.py) based on your processing speed
- Monitor queue size and buffer length during operation
- Consider reducing `WINDOW_SIZE` or increasing `OVERLAP` for faster processing

### Dependencies
If you encounter import errors, ensure all dependencies are installed:
```bash
pip install --upgrade -r requirements.txt
```

## Technical Specifications

- **Sampling Rate**: 250 Hz (configurable)
- **Resolution**: 24-bit ADC
- **Input Range**: ±187.5 µV (with default gain settings)
- **Channels**: 8 EEG channels
- **Data Format**: 32-bit floating point (µV)