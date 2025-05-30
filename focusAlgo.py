# ---- FOCUS Extraction Algo (Non ML) -----

import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

def compute_fft(data, sampling_rate):
    """
    Compute FFT for each channel in the data.
    
    Args:
        data (np.ndarray): EEG data of shape (samples, channels)
        sampling_rate (float): Sampling rate in Hz
        
    Returns:
        tuple: (frequencies, fft_results) where:
            - frequencies: array of frequency values
            - fft_results: array of FFT results for each channel
    """
    n_samples = data.shape[0]
    
    # Compute FFT for each channel
    fft_results = np.zeros((n_samples//2 + 1, data.shape[1]), dtype=complex)
    for channel in range(data.shape[1]):
        fft_results[:, channel] = fft(data[:, channel])[:n_samples//2 + 1]
    
    # Compute frequency array
    frequencies = fftfreq(n_samples, 1/sampling_rate)[:n_samples//2 + 1]
    
    return frequencies, fft_results

def get_band_powers(frequencies, fft_results):
    """
    Extract power in different frequency bands.
    
    Args:
        frequencies (np.ndarray): Array of frequency values
        fft_results (np.ndarray): FFT results for each channel
        
    Returns:
        dict: Dictionary containing power in each frequency band for each channel
    """
    # Define frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100)
    }
    
    # Compute power spectrum
    power_spectrum = np.abs(fft_results) ** 2
    
    # Initialize results dictionary
    band_powers = {band: np.zeros(fft_results.shape[1]) for band in bands}
    
    # Calculate power in each band for each channel
    for band_name, (low_freq, high_freq) in bands.items():
        # Find indices corresponding to frequency band
        band_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
        
        # Sum power in the band for each channel
        for channel in range(fft_results.shape[1]):
            band_powers[band_name][channel] = np.sum(power_spectrum[band_mask, channel])
    
    return band_powers

def plot_fft_analysis(frequencies, fft_results, band_powers, channel=0):
    """
    Plot FFT analysis results for a specific channel.
    
    Args:
        frequencies (np.ndarray): Array of frequency values
        fft_results (np.ndarray): FFT results for each channel
        band_powers (dict): Dictionary of band powers
        channel (int): Channel to plot
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot power spectrum
    power_spectrum = np.abs(fft_results[:, channel]) ** 2
    ax1.plot(frequencies, power_spectrum)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power')
    ax1.set_title(f'Power Spectrum - Channel {channel}')
    ax1.grid(True)
    ax1.set_xlim(0, 50)  # Show frequencies up to 50 Hz
    
    # Plot band powers
    bands = list(band_powers.keys())
    powers = [band_powers[band][channel] for band in bands]
    powers = [x // 1000 for x in powers]
    ax2.bar(bands, powers)
    ax2.set_xlabel('Frequency Band')
    ax2.set_ylabel('Power')
    ax2.set_title(f'Band Powers - Channel {channel}')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def analyze_epoch(epoch, channel=0):
    """
    Analyze an epoch of EEG data using FFT.
    
    Args:
        epoch (dict): Dictionary containing 'data' and 'timestamps'
        channel (int): Channel to analyze
    """
    data = epoch['data']
    sampling_rate = 255  # Hz
    
    # Compute FFT
    frequencies, fft_results = compute_fft(data, sampling_rate)
    
    # Get band powers
    band_powers = get_band_powers(frequencies, fft_results)
    
    # Plot results
    plot_fft_analysis(frequencies, fft_results, band_powers, channel)
    
    return band_powers

# Example usage:
# band_powers = analyze_epoch(epoch, channel=0)

def getFocus(bandpowers):
    # Fp1, Fp2, F7, F3, FZ, F4, F8, C2
    delta = bandpowers['delta']
    theta = bandpowers['theta']
    alpha = bandpowers['alpha']
    beta = bandpowers['beta']
    gamma = bandpowers['gamma']

    #We are primarily gonna use channels 0 and 1 becuase they correlate to
    #prefontal cortex nodes

    #Beta/Alpha Ratio: An increase in this ratio is a very common indicator of increased 
    # concentration and mental effort. 
    # This means more beta activity (focused) relative to less alpha activity (relaxed).
    betaAlpha = sum(beta) / sum(alpha)

    #Theta/Alpha Ratio: An increase in this ratio can sometimes be seen in states of 
    # internal focus or cognitive overload, but a decrease might be desired for external 
    # attention.
    thetaAlpha = (theta[0]+theta[1]) / (alpha[0] + alpha[1])

    #im gonna work with betaAlpha for now but thetaAlpga and other are also valid
    return int(100*(betaAlpha))