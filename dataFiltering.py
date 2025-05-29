import threading
import time
import matplotlib.pyplot as plt
import numpy as np
import acquisition
import queue

plot_queue = queue.Queue()

def start_acquisition_in_thread():
    # Start acquisition.main() in a background thread
    thread = threading.Thread(target=acquisition.main, daemon=True)
    thread.start()

def plot_epoch(epoch):
    data = epoch['data']  # Shape: (WINDOW_SIZE, CHANNEL_COUNT)
    timestamps = epoch['timestamps']
    plt.figure(figsize=(12, 8))
    for i in range(data.shape[1]):
        plt.plot(timestamps, data[:, i], label=f'Channel {i}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('EEG Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()

def epoch_plotting_thread():
    while True:
        epoch = acquisition.get_epoched_data(timeout=1.0)
        if epoch:
            plot_queue.put(epoch)  # Send to main thread for plotting
        else:
            time.sleep(0.1)

def main():
    start_acquisition_in_thread()
    thread = threading.Thread(target=epoch_plotting_thread, daemon=True)
    thread.start()
    print("Epoch plotting thread started. Press Ctrl+C to exit.")
    try:
        while True:
            try:
                epoch = plot_queue.get(timeout=0.1)
                print("DATAlol")
                print(epoch['data'][1])
                plot_epoch(epoch)
            except queue.Empty:
                pass
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    main()
