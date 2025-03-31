import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# Generate sample data
X, y = make_blobs(n_samples=100, centers=2, random_state=20)

# Create an SVM classifier
clf = svm.SVC(kernel='linear', C=1.0)

# Train the classifier
clf.fit(X, y)

# Get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

# Plot the data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

# Plot the decision boundary
plt.plot(xx, yy, 'k-', label='Decision Boundary')

# Plot the margins
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_up = yy + a * margin
yy_down = yy - a * margin
plt.plot(xx, yy_up, 'k--', label='Margins')
plt.plot(xx, yy_down, 'k--')

# Plot support vectors
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
            facecolors='none', edgecolors='k', label='Support Vectors')

# Add labels and title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM with Linear Kernel')
plt.legend()
plt.show()
