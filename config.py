# Configuration for OpenBCI data processing
PORT = '/dev/tty.usbserial-DM0258IG' 
CHANNEL_COUNT = 8 # Number of channels (Cyton has 8)
SAMPLE_RATE = 250  # Hz (Cyton default)
WINDOW_SIZE = 250  # Samples (1 second)
OVERLAP = 0.5  # 50% overlap