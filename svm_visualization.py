import streamlit as st
from sklearn import svm
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


# Set the page configuration
st.set_page_config(
    page_title="Interactive SVM Visualization",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title of the app
st.title("Interactive Support Vector Machine (SVM) Visualization")


# Sidebar for parameter controls
st.sidebar.header("SVM Parameters")

# Regularization parameter C
C = st.sidebar.slider("C (Regularization Parameter)", 0.1, 10.0, 1.0, step=0.1)

# Kernel coefficient gamma
gamma = st.sidebar.slider("Gamma (Kernel Coefficient)", 0.1, 10.0, 1.0, step=0.1)

# Select SVM type
svm_type = st.sidebar.selectbox("SVM Type", ["SVC (Classification)", "SVR (Regression)"])

# Epsilon parameter for SVR
if svm_type == "SVR (Regression)":
    epsilon = st.sidebar.slider("Epsilon (SVR Parameter)", 0.1, 5.0, 0.5, step=0.1)
else:
    epsilon = None


# Option to generate synthetic data or upload
data_option = st.sidebar.radio("Data Option", ["Generate Synthetic Data", "Upload CSV Data"])

if data_option == "Generate Synthetic Data":
    # Parameters for data generation
    n_samples = st.sidebar.slider("Number of Samples", 50, 300, 100, step=50)
    centers = st.sidebar.slider("Number of Centers", 2, 5, 2, step=1)
    cluster_std = st.sidebar.slider("Cluster Standard Deviation", 0.5, 3.0, 1.0, step=0.1)
    
    # Generate synthetic data
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=42)
    X = StandardScaler().fit_transform(X)
    data = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
    data['Label'] = y
else:
    # Upload CSV data
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        if 'Label' not in data.columns:
            st.error("CSV must contain a 'Label' column for classification.")
    else:
        st.warning("Please upload a CSV file.")
        st.stop()


st.sidebar.header("Add a New Data Point")

with st.sidebar.form(key='add_point_form'):
    new_x1 = st.number_input("Feature 1", value=0.0)
    new_x2 = st.number_input("Feature 2", value=0.0)
    if svm_type == "SVC (Classification)":
        new_label = st.selectbox("Label", [0, 1])
    else:
        new_label = st.number_input("Target Value", value=0.0)
    submit_button = st.form_submit_button(label='Add Data Point')

if submit_button:
    new_data = {'Feature 1': new_x1, 'Feature 2': new_x2, 'Label': new_label}
    data = data.append(new_data, ignore_index=True)
    st.success("New data point added!")


from sklearn.metrics import accuracy_score, mean_squared_error

# Separate features and target
X = data[['Feature 1', 'Feature 2']].values
y = data['Label'].values

# Choose the SVM model type
if svm_type == "SVC (Classification)":
    model = svm.SVC(C=C, gamma=gamma, kernel='rbf')
else:
    model = svm.SVR(C=C, gamma=gamma, epsilon=epsilon)

# Train the model
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Calculate performance metrics
if svm_type == "SVC (Classification)":
    accuracy = accuracy_score(y, predictions)
else:
    mse = mean_squared_error(y, predictions)



def plot_svm(model, X, y, svm_type):
    # Create a grid to plot decision boundary
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)
    
    # Initialize Plotly figure
    fig = go.Figure()
    
    # Add contour for decision boundary
    if svm_type == "SVC (Classification)":
        fig.add_trace(go.Contour(
            x=np.arange(x_min, x_max, h),
            y=np.arange(y_min, y_max, h),
            z=Z,
            showscale=False,
            colorscale='RdBu',
            opacity=0.5,
            contours=dict(
                start=0,
                end=1,
                size=1,
                coloring='fill'
            )
        ))
    else:
        # For SVR, plot the regression line
        fig.add_trace(go.Contour(
            x=np.arange(x_min, x_max, h),
            y=np.arange(y_min, y_max, h),
            z=Z,
            showscale=False,
            colorscale='Greens',
            opacity=0.5,
            contours=dict(
                start=min(y),
                end=max(y),
                size=(max(y) - min(y)) / 20,
                coloring='fill'
            )
        ))
    
    # Add scatter points
    if svm_type == "SVC (Classification)":
        fig.add_trace(go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode='markers',
            marker=dict(
                color=y,
                colorscale='RdBu',
                showscale=True,
                colorbar=dict(title='Label')
            )
        ))
    else:
        fig.add_trace(go.Scatter(
            x=X[:, 0],
            y=y,
            mode='markers',
            marker=dict(
                color=y,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Target')
            )
        ))
    
    # Highlight support vectors
    if svm_type == "SVC (Classification)":
        support_vectors = model.support_vectors_
        fig.add_trace(go.Scatter(
            x=support_vectors[:, 0],
            y=support_vectors[:, 1],
            mode='markers',
            marker=dict(
                color='yellow',
                size=12,
                symbol='circle-open',
                line=dict(width=2, color='black')
            ),
            name='Support Vectors'
        ))
    
    # Update layout
    fig.update_layout(
        title="SVM Decision Boundary and Data Points",
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        width=800,
        height=600
    )
    
    return fig

# Generate the plot
svm_plot = plot_svm(model, X, y, svm_type)

# Display the plot
st.plotly_chart(svm_plot, use_container_width=True)



st.sidebar.header("Performance Metrics")

if svm_type == "SVC (Classification)":
    st.sidebar.write(f"**Accuracy:** {accuracy * 100:.2f}%")
else:
    st.sidebar.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
