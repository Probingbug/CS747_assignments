import numpy as np

# Ignore comment lines starting with '%'
params = np.loadtxt("outcmaes/xrecentbest.dat", comments='%')

# Save as .npy
np.save("outcmaes/xbest.npy", params)
print("Saved as outcmaes/xbest.npy")