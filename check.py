import numpy as np
labels = np.load('BTXRD/annotations/IMG000024.npy', allow_pickle=True)
print(type(labels), labels.shape)
print(labels)