from trainingArrayGenerator import generate_random_batch, load_training_data


data, labels = load_training_data()

batchData, batchLabels = generate_random_batch(data, labels)

print(batchLabels)