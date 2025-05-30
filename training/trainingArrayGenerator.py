import numpy as np
import random

def load_training_data(filename='training_data.txt'):
    """
    Load raw training data from the text file.
    Returns:
        data: numpy array of shape (n_epochs, window_size, n_channels)
        labels: numpy array of shape (n_epochs,)
    """
    data = []
    labels = []
    
    with open(filename, 'r') as f:
        # Skip header lines (lines starting with #)
        for line in f:
            if not line.startswith('#'):
                # Parse the line
                values = line.strip().split()
                if not values:  # Skip empty lines
                    continue
                    
                # First value is the label
                label = int(values[0])
                labels.append(label)
                
                # Remaining values are the raw channel data
                # Reshape into (window_size, n_channels)
                channel_data = np.array([float(x) for x in values[1:]]).reshape(-1, 8)
                data.append(channel_data)
    
    return np.array(data), np.array(labels)

def generate_random_batch(data, labels, batch_size=32, epochs_per_class=8):
    """
    Generate a random batch of raw epochs with balanced class distribution.
    The data within each batch is shuffled to prevent sequential patterns.
    
    Args:
        data: numpy array of shape (n_epochs, window_size, n_channels)
        labels: numpy array of shape (n_epochs,)
        batch_size: number of epochs in the batch (must be divisible by number of classes)
        epochs_per_class: number of epochs to include from each class
        
    Returns:
        batch_data: numpy array of shape (batch_size, window_size, n_channels)
        batch_labels: numpy array of shape (batch_size,)
    """
    # Get unique classes
    classes = np.unique(labels)
    n_classes = len(classes)
    
    # Ensure batch_size is divisible by number of classes
    if batch_size % n_classes != 0:
        raise ValueError(f"batch_size must be divisible by number of classes ({n_classes})")
    
    # Initialize batch arrays
    batch_data = []
    batch_labels = []
    
    # For each class, randomly select epochs_per_class samples
    for class_label in classes:
        # Get indices for this class
        class_indices = np.where(labels == class_label)[0]
        
        # Randomly select epochs_per_class samples
        selected_indices = random.sample(list(class_indices), epochs_per_class)
        
        # Add to batch
        batch_data.extend(data[selected_indices])
        batch_labels.extend([class_label] * epochs_per_class)
    
    # Convert to numpy arrays
    batch_data = np.array(batch_data)
    batch_labels = np.array(batch_labels)
    
    # Create a random permutation of indices
    indices = np.random.permutation(len(batch_data))
    
    # Shuffle both data and labels using the same permutation
    batch_data = batch_data[indices]
    batch_labels = batch_labels[indices]
    
    return batch_data, batch_labels

def main():
    # Load the training data
    print("Loading raw training data...")
    data, labels = load_training_data()
    print(f"Loaded {len(data)} epochs with shape {data.shape}")
    print("Label distribution:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Level {label}: {count} epochs")
    
    # Generate a few random batches
    print("\nGenerating random batches of raw data...")
    for i in range(3):
        batch_data, batch_labels = generate_random_batch(data, labels, batch_size=32, epochs_per_class=8)
        print(f"\nBatch {i+1}:")
        print(f"Shape: {batch_data.shape}")
        print("Label distribution:")
        unique_labels, counts = np.unique(batch_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"Level {label}: {count} epochs")
        
        # Print some statistics about the raw data
        print("\nRaw data statistics:")
        print(f"Mean: {np.mean(batch_data):.2f}")
        print(f"Std: {np.std(batch_data):.2f}")
        print(f"Min: {np.min(batch_data):.2f}")
        print(f"Max: {np.max(batch_data):.2f}")

if __name__ == "__main__":
    main()
