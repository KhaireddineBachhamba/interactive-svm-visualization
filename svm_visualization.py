import numpy as np
import plotly.graph_objects as go
from sklearn import svm

# Generate synthetic data for two classes
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
y = [0] * 20 + [1] * 20

# Fit the SVM model
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X, y)

# Create a mesh grid to plot decision boundary
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot data points and decision boundary
trace_points = go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=y, colorscale='Viridis'))

trace_boundary = go.Contour(x=np.linspace(-5, 5, 50), y=np.linspace(-5, 5, 50), z=Z, 
                            contours_coloring='lines', showscale=False)

# Create a figure to contain the plot
fig = go.Figure(data=[trace_points, trace_boundary])

# Add labels and title
fig.update_layout(title='SVM Decision Boundary', xaxis_title='Feature 1', yaxis_title='Feature 2')

# Show the interactive plot
fig.show()
fig.write_html("svm_interactive.html")
