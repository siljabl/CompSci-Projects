import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, ExpSineSquared as E

# Franke function
def franke_function(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4

# Generate random points in functino space
np.random.seed(0)
n = 100
X = np.random.rand(n, 2)
y = franke_function(X[:, 0], X[:, 1]) + np.random.randn(n) * 0.1

# Kernel for GPR
kernel1 = RBF(0.1, (1e-3, 1e3))
kernel2 = C(1.0, (1e-3, 1e3))
kernel3 = C(1.0, (1e-3, 1e3)) * RBF(0.1, (1e-3, 1e3))
kernel4 = E(0.1,  1.0)

# Create model
gpr = GaussianProcessRegressor(kernel=kernel4, n_restarts_optimizer=9)

# Fit the model 
gpr.fit(X, y)

# Generate grid of points for prediction
x1, x2 = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
X_pred = np.array([x1.flatten(), x2.flatten()]).T

# Predict the function values 
y_pred, sigma = gpr.predict(X_pred, return_std=True)

# Reshape for plotting
y_pred = y_pred.reshape(x1.shape)
sigma = sigma.reshape(x1.shape)

# Plot Franke function
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(x1, x2, franke_function(x1, x2), cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Franke Function')

# Plot predicted function values
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(x1, x2, y_pred, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('GPR Prediction - ExpSineSq kernel')

plt.tight_layout()
plt.show()
