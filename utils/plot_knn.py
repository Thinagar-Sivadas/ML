import plotly.graph_objects as go
import numpy as np

class PlotKNN(object):
    """Plot KMeans

    Args:
        object (class): Module for various dimension knn plot
    """

    def __init__(self):
        """Intialise variable
        """

        # Declare aesthetic parameters
        self.color_coding = ['mediumblue', 'darkcyan', 'deeppink', 'coral']
        self.opacity = 0.65
        self.data_size = 3.5
        self.centroid_size = 9

        # Declare figure parameters
        self.autorange = False
        self.title_x = 'Feature 1'
        self.title_y = 'Feature 2'
        self.title_z = 'Feature 3'
        self.title_train = "KNN Training"
        self.title_test = "KNN Testing"
        self.title_x_loc = 0.5
        self.hovermode = "closest"
        self.width = 1200
        self.height = 600

    def plot_training_1d(self, train_data, n_clusters, train_labels):
        """Performs knn training plot 1d

        Args:
            train_data (numpy): Data to visualise
            n_clusters (int): Number of cluster
            train_labels (numpy): Train labels
        """

        # Range of feature space to visualise
        x_min, y_min = train_data.min(axis=0)[0] - 1, -1
        x_max, y_max = train_data.max(axis=0)[0] + 1, 1

        # Plots final frame
        frame = []
        for n_cluster in range(n_clusters):

            # Plot training points with cluster labels
            x = list(np.array(train_data[train_labels == n_cluster])[:,0])
            y = [0]*sum(train_labels == n_cluster)
            frame.append(
                go.Scatter(
                    x = x,
                    y = y,
                    mode="markers",
                    marker_size=self.data_size,
                    marker_color=self.color_coding[n_cluster],
                    marker_opacity=self.opacity,
                    name=f'Training Points Cluster {n_cluster+1}',
                    )
                )

        # Initalise and display figure object
        fig = go.Figure(
            data=frame,
            layout=go.Layout(
                xaxis=dict(range=[x_min, x_max], autorange=self.autorange, title=self.title_x),
                yaxis=dict(range=[y_min, y_max], autorange=self.autorange, title=self.title_y),
                title_text=self.title_train, title_x=self.title_x_loc, hovermode=self.hovermode,
                width=self.width, height=self.height,
                )
            )

        fig.show()

    def plot_training_2d(self, train_data, n_clusters, train_labels):
        """Performs knn training plot 2d

        Args:
            train_data (numpy): Data to visualise
            n_clusters (int): Number of cluster
            train_labels (numpy): Train labels
        """

        # Range of feature space to visualise
        x_min, y_min = train_data.min(axis=0) - 1
        x_max, y_max = train_data.max(axis=0) + 1

        # Plots final frame
        frame = []
        for n_cluster in range(n_clusters):

            # Plot training points with cluster labels
            x = list(np.array(train_data[train_labels == n_cluster])[:,0])
            y = list(np.array(train_data[train_labels == n_cluster])[:,1])
            frame.append(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker_size=self.data_size,
                    marker_color=self.color_coding[n_cluster],
                    marker_opacity=self.opacity,
                    name=f'Training Points Cluster {n_cluster+1}',
                    )
                )

        # Initalise and display figure object
        fig = go.Figure(
            data=frame,
            layout=go.Layout(
                xaxis=dict(range=[x_min, x_max], autorange=self.autorange, title=self.title_x),
                yaxis=dict(range=[y_min, y_max], autorange=self.autorange, title=self.title_y),
                title_text=self.title_train, title_x=self.title_x_loc, hovermode=self.hovermode,
                width=self.width, height=self.height,
                )
            )

        fig.show()

    def plot_training_3d(self, train_data, n_clusters, train_labels):
        """Performs knn training plot 3d

        Args:
            train_data (numpy): Data to visualise
            n_clusters (int): Number of cluster
            train_labels (numpy): Train labels
        """

        # Range of feature space to visualise
        x_min, y_min, z_min = train_data.min(axis=0) - 1
        x_max, y_max, z_max = train_data.max(axis=0) + 1

        # Plots final frame
        frame = []
        for n_cluster in range(n_clusters):

            # Plot training points with cluster labels
            x = list(np.array(train_data[train_labels == n_cluster])[:,0])
            y = list(np.array(train_data[train_labels == n_cluster])[:,1])
            z = list(np.array(train_data[train_labels == n_cluster])[:,2])
            frame.append(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker_size=self.data_size - 1,
                    marker_color=self.color_coding[n_cluster],
                    marker_opacity=self.opacity,
                    name=f'Training Points Cluster {n_cluster+1}',
                    )
                )

        # Initalise and display figure object
        fig = go.Figure(
            data=frame,
            layout=go.Layout(
                scene = dict(xaxis=dict(range=[x_min, x_max], autorange=self.autorange, title=self.title_x),
                             yaxis=dict(range=[y_min, y_max], autorange=self.autorange, title=self.title_y),
                             zaxis=dict(range=[z_min, z_max], autorange=self.autorange, title=self.title_z)),
                title_text=self.title_train, title_x=self.title_x_loc, hovermode=self.hovermode,
                width=self.width + 300, height=self.height + 100
                )
            )

        fig.show()

    def plot_testing_1d(self, train_data, test_data, n_clusters, train_labels, test_labels):
        """Performs knn testing plot 1d

        Args:
            train_data (numpy): Train data to visualise
            test_data (numpy): Test data to visualise
            n_clusters (int): Number of cluster
            train_labels (numpy): Train target labels
            test_labels (numpy): Test target labels
        """

        # Range of feature space to visualise
        x_min, y_min = np.vstack((train_data,test_data)).min(axis=0) - 1, -1
        x_max, y_max = np.vstack((train_data,test_data)).max(axis=0) + 1, 1

        # Plots final frame
        frame = []
        for n_cluster in range(n_clusters):

            # Plot training points with cluster labels
            x = list(np.array(train_data[train_labels == n_cluster])[:,0])
            y = [0]*sum(train_labels == n_cluster)
            frame.append(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker_size=self.data_size,
                    marker_color=self.color_coding[n_cluster],
                    marker_opacity=self.opacity,
                    name=f'Training Points Cluster {n_cluster+1}',
                    )
                )

            # Plot testing points with cluster labels
            x = list(np.array(test_data[test_labels == n_cluster])[:,0])
            y = [0]*sum(test_labels == n_cluster)
            frame.append(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker_size=self.data_size + 1.5,
                    marker_color=self.color_coding[n_cluster],
                    marker_symbol='cross',
                    marker_opacity=self.opacity,
                    name=f'Testing Points Cluster {n_cluster+1}',
                    )
                )

        # Initalise and display figure object
        fig = go.Figure(
            data=frame,
            layout=go.Layout(
                xaxis=dict(range=[x_min, x_max], autorange=self.autorange, title=self.title_x),
                yaxis=dict(range=[y_min, y_max], autorange=self.autorange, title=self.title_y),
                title_text=self.title_test, title_x=self.title_x_loc, hovermode=self.hovermode,
                width=self.width, height=self.height)
            )

        fig.show()

    def plot_testing_2d(self, train_data, test_data, n_clusters, train_labels, test_labels):
        """Performs knn testing plot 2d

        Args:
            train_data (numpy): Train data to visualise
            test_data (numpy): Test data to visualise
            n_clusters (int): Number of cluster
            train_labels (numpy): Train target labels
            test_labels (numpy): Test target labels
        """

        # Range of feature space to visualise
        x_min, y_min = np.vstack((train_data,test_data)).min(axis=0) - 1
        x_max, y_max = np.vstack((train_data,test_data)).max(axis=0) + 1

        # Plots final frame
        frame = []
        for n_cluster in range(n_clusters):

            # Plot training points with cluster labels
            x = list(np.array(train_data[train_labels == n_cluster])[:,0])
            y = list(np.array(train_data[train_labels == n_cluster])[:,1])
            frame.append(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker_size=self.data_size,
                    marker_color=self.color_coding[n_cluster],
                    marker_opacity=self.opacity,
                    name=f'Training Points Cluster {n_cluster+1}',
                    )
                )

            # Plot testing points with cluster labels
            x = list(np.array(test_data[test_labels == n_cluster])[:,0])
            y = list(np.array(test_data[test_labels == n_cluster])[:,1])
            frame.append(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker_size=self.data_size + 1.5,
                    marker_color=self.color_coding[n_cluster],
                    marker_symbol='cross',
                    marker_opacity=self.opacity,
                    name=f'Testing Points Cluster {n_cluster+1}',
                    )
                )

        # Initalise and display figure object
        fig = go.Figure(
            data=frame,
            layout=go.Layout(
                xaxis=dict(range=[x_min, x_max], autorange=self.autorange, title=self.title_x),
                yaxis=dict(range=[y_min, y_max], autorange=self.autorange, title=self.title_y),
                title_text=self.title_test, title_x=self.title_x_loc, hovermode=self.hovermode,
                width=self.width, height=self.height)
            )

        fig.show()

    def plot_testing_3d(self, train_data, test_data, n_clusters, train_labels, test_labels):
        """Performs knn testing plot 3d

        Args:
            train_data (numpy): Train data to visualise
            test_data (numpy): Test data to visualise
            n_clusters (int): Number of cluster
            train_labels (numpy): Train target labels
            test_labels (numpy): Test target labels
        """

        # Range of feature space to visualise
        x_min, y_min, z_min = np.vstack((train_data,test_data)).min(axis=0) - 1
        x_max, y_max, z_max = np.vstack((train_data,test_data)).max(axis=0) + 1

        # Plots final frame
        frame = []
        for n_cluster in range(n_clusters):

            # Plot training points with cluster labels
            x = list(np.array(train_data[train_labels == n_cluster])[:,0])
            y = list(np.array(train_data[train_labels == n_cluster])[:,1])
            z = list(np.array(train_data[train_labels == n_cluster])[:,2])
            frame.append(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker_size=self.data_size - 1,
                    marker_color=self.color_coding[n_cluster],
                    marker_opacity=self.opacity,
                    name=f'Training Points Cluster {n_cluster+1}',
                    )
                )

            # Plot testing points with cluster labels
            x = list(np.array(test_data[test_labels == n_cluster])[:,0])
            y = list(np.array(test_data[test_labels == n_cluster])[:,1])
            z = list(np.array(test_data[test_labels == n_cluster])[:,2])
            frame.append(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker_size=self.data_size + 1.5,
                    marker_color=self.color_coding[n_cluster],
                    marker_symbol='cross',
                    marker_opacity=self.opacity,
                    name=f'Testing Points Cluster {n_cluster+1}',
                    )
                )

        # Initalise and display figure object
        fig = go.Figure(
            data=frame,
            layout=go.Layout(
                scene = dict(xaxis=dict(range=[x_min, x_max], autorange=self.autorange, title=self.title_x),
                             yaxis=dict(range=[y_min, y_max], autorange=self.autorange, title=self.title_y),
                             zaxis=dict(range=[z_min, z_max], autorange=self.autorange, title=self.title_z)),
                title_text=self.title_test, title_x=self.title_x_loc, hovermode=self.hovermode,
                width=self.width + 300, height=self.height + 100)
            )

        fig.show()