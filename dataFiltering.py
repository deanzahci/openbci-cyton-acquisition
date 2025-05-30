import threading
import time
import matplotlib.pyplot as plt
import numpy as np
import acquisition
import queue
from scipy.signal import butter, filtfilt
import focusAlgo


#if using the dataset, well need to change these
datamin = -130000
datamax = 30000

newmax = 1
newmin = -1

data_queue = queue.Queue()
focus_queue = queue.Queue(maxsize=300)

plotChannel = 0
samplingRate = 255
# --- GETTING DATA ---- 
def epoch_getting_thread():
    while True:
        epoch = acquisition.get_epoched_data(timeout=1.0)
        if epoch:
            # Ensure timestamps are unique
            epoch['timestamps'] = continuous_timestamps(epoch)
            epoch['data'] = scale_data(epoch['data'])
            
            # Apply filters
            try:
                # Finally apply bandpass filter to focus on specific frequency range (e.g., 1-30 Hz)
                filtered_data = butter_bandpass_filter(epoch['data'], 1, 30, samplingRate)
                epoch['data'] = filtered_data

            except Exception as e:
                print(f"Error applying filters: {e}")
                continue
                
            data_queue.put(epoch)  # Send to main thread for plotting
        else:
            time.sleep(0.1)

def continuous_timestamps(epoch):
    """
    takes in epoch['timestamps']
    Ensures timestamps are unique by remapping them based on the duration
    Returns a new array with unique timestamps.

    This solves a problem in the data, where very small differences in timestamps are considered equal
    becuase of round off.
    """
    newTimes = []
    offset = epoch['duration']/epoch['sample_count']
    myTime = epoch['start_time']
    for i in range(epoch['sample_count']):
        newTimes.append(myTime)
        myTime += offset

    return newTimes

def scale_data(data):
    """
    Scale the input data to range [-1, 1] using min-max normalization.
    
    Args:
        data: Input data array
        
    Returns:
        Scaled data array with values between -1 and 1
    """
    # Convert to numpy array if not already
    data = np.array(data)
    
    # Find min and max values
    data_min = np.min(data)
    data_max = np.max(data)
    
    # Scale to [-1, 1]
    scaled_data = 2 * ((data - data_min) / (data_max - data_min)) - 1
    
    return scaled_data

#------ FILTERING ------- 

def butter_highpass_filter(data, cutoff_freq, sampling_rate, order=5):
    """
    Applies a Butterworth high-pass filter to the data.

    Args:
        data (np.ndarray): The input signal (2D array of shape [samples, channels]).
        cutoff_freq (float): The cutoff frequency of the filter in Hz.
        sampling_rate (float): The sampling rate of the signal in Hz.
        order (int): The order of the filter. Higher orders provide a sharper
                     rolloff but can introduce more ringing. Common values are 4-6.

    Returns:
        np.ndarray: The filtered signal with same shape as input.
    """
    if cutoff_freq >= sampling_rate/2:
        raise ValueError("Cutoff frequency must be less than Nyquist frequency (sampling_rate/2)")
    
    nyquist_freq = 0.5 * sampling_rate
    normalized_cutoff = cutoff_freq / nyquist_freq
    
    # Design the Butterworth filter
    b, a = butter(order, normalized_cutoff, btype='highpass', analog=False)
    
    # Apply filter to each channel separately
    filtered_data = np.zeros_like(data)
    for channel in range(data.shape[1]):
        filtered_data[:, channel] = filtfilt(b, a, data[:, channel])
    
    return filtered_data

def butter_lowpass_filter(data, cutoff_freq, sampling_rate, order=5):
    """
    Applies a Butterworth low-pass filter to the data.

    Args:
        data (np.ndarray): The input signal (2D array of shape [samples, channels]).
        cutoff_freq (float): The cutoff frequency of the filter in Hz.
        sampling_rate (float): The sampling rate of the signal in Hz.
        order (int): The order of the filter. Higher orders provide a sharper
                     rolloff but can introduce more ringing. Common values are 4-6.

    Returns:
        np.ndarray: The filtered signal with same shape as input.
    """
    if cutoff_freq >= sampling_rate/2:
        raise ValueError("Cutoff frequency must be less than Nyquist frequency (sampling_rate/2)")
    
    nyquist_freq = 0.5 * sampling_rate
    normalized_cutoff = cutoff_freq / nyquist_freq
    
    # Design the Butterworth filter
    b, a = butter(order, normalized_cutoff, btype='lowpass', analog=False)
    
    # Apply filter to each channel separately
    filtered_data = np.zeros_like(data)
    for channel in range(data.shape[1]):
        filtered_data[:, channel] = filtfilt(b, a, data[:, channel])
    
    return filtered_data

def butter_notch_filter(data, notch_freq, sampling_rate, quality_factor=30.0):
    """
    Applies a notch (band-stop) filter to remove power line interference.

    Args:
        data (np.ndarray): The input signal (2D array of shape [samples, channels]).
        notch_freq (float): The frequency to remove (e.g., 50 or 60 Hz for power line).
        sampling_rate (float): The sampling rate of the signal in Hz.
        quality_factor (float): Quality factor of the notch filter. Higher values create
                              a narrower notch. Default is 30.0.

    Returns:
        np.ndarray: The filtered signal with same shape as input.
    """
    if notch_freq >= sampling_rate/2:
        raise ValueError("Notch frequency must be less than Nyquist frequency (sampling_rate/2)")
    
    # Calculate the normalized frequency
    nyquist_freq = 0.5 * sampling_rate
    normalized_freq = notch_freq / nyquist_freq
    
    # Design the notch filter
    b, a = butter(2, [normalized_freq - 1.0/quality_factor, 
                     normalized_freq + 1.0/quality_factor], 
                 btype='bandstop')
    
    # Apply filter to each channel separately
    filtered_data = np.zeros_like(data)
    for channel in range(data.shape[1]):
        filtered_data[:, channel] = filtfilt(b, a, data[:, channel])
    
    return filtered_data

def butter_bandpass_filter(data, lowcut, highcut, sampling_rate, order=5):
    """
    Applies a Butterworth band-pass filter to the data.

    Args:
        data (np.ndarray): The input signal (2D array of shape [samples, channels]).
        lowcut (float): The lower cutoff frequency in Hz.
        highcut (float): The upper cutoff frequency in Hz.
        sampling_rate (float): The sampling rate of the signal in Hz.
        order (int): The order of the filter. Higher orders provide a sharper
                     rolloff but can introduce more ringing. Common values are 4-6.

    Returns:
        np.ndarray: The filtered signal with same shape as input.
    """
    if highcut >= sampling_rate/2:
        raise ValueError("High cutoff frequency must be less than Nyquist frequency (sampling_rate/2)")
    if lowcut >= highcut:
        raise ValueError("Low cutoff frequency must be less than high cutoff frequency")
    
    nyquist_freq = 0.5 * sampling_rate
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    
    # Design the Butterworth filter
    b, a = butter(order, [low, high], btype='band')
    
    # Apply filter to each channel separately
    filtered_data = np.zeros_like(data)
    for channel in range(data.shape[1]):
        filtered_data[:, channel] = filtfilt(b, a, data[:, channel])
    
    return filtered_data

# ------- PLOTTING ---------
def plot_epoch(epoch, channel=0):
    """
    Plot a single channel of EEG data.
    Args:
        epoch: Dictionary containing 'data' and 'timestamps'
        channel: Channel number to plot (0-7 for 8 channels)
    """
    data = epoch['data']  # Shape: (WINDOW_SIZE, CHANNEL_COUNT)
    timestamps = epoch['timestamps']
    
    
    plt.figure(figsize=(12, 8))
    plt.plot(timestamps, data[:, channel], label=f'Channel {channel}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.title(f'EEG Epoch - Channel {channel}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    start_acquisition_in_thread()
    thread = threading.Thread(target=epoch_getting_thread, daemon=True)
    thread.start()
    print("Epoch plotting thread started. Press Ctrl+C to exit.")
    try:
        while True:
            try:
                epoch = data_queue.get(timeout=0.1)
                # Plot the time domain signal
                plot_epoch(epoch, channel=plotChannel)
                # Perform and plot FFT analysis
                band_powers = focusAlgo.analyze_epoch(epoch, channel=plotChannel)
                focus_queue.put(focusAlgo.getFocus(band_powers))
                #print("Queue size: ", data_queue.qsize())
            except queue.Empty:
                pass
            time.sleep(0.01)
    except KeyboardInterrupt:
        focusvals = list(focus_queue.queue)
        print(sum(focusvals)/len(focusvals))
        print("Exiting...")


def start_acquisition_in_thread():
    # Start acquisition.main() in a background thread - only if running dataFiltering as main
    thread = threading.Thread(target=acquisition.main, daemon=True)
    thread.start()


if __name__ == "__main__":
    main()
