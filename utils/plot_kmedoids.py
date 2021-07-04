import plotly.graph_objects as go
import numpy as np

class PlotKMedoids(object):
    """Plot KMedoids

    Args:
        object (class): Module for various dimension kmeans plot
    """

    def __init__(self):
        """Intialise variable
        """

        # Initalise location at where buttons and sliders will be
        self.updatemenus = [
            dict(type="buttons",
                 buttons=[dict(label="Play", method="animate",
                               args=[None, {
                                   "frame": {
                                       "duration": 700,
                                       "redraw": True
                                       },
                                   "fromcurrent": True,
                                   "transition": {
                                       "duration": 0,
                                       }
                                   }]),
                          dict(label="Pause", method="animate",
                               args=[[None],{
                                   "frame": {
                                       "duration": 0,
                                       "redraw": True
                                       },
                                   "mode": "immediate",
                                   "transition": {
                                       "duration": 0
                                       }
                                   }]),
                          ],
                 direction="left",
                 pad={"r": 10, "t": 87},
                 showactive= False,
                 x= 0.1,
                 xanchor= "right",
                 y= 0,
                 yanchor="top"
                 ),
            ]

        # Initialise slider
        self.sliders_dict = {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Iteration:",
                "visible": True,
                "xanchor": "right"
            },
            "pad": {
                "b": 10,
                "t": 50
                },
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": []
            }

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
        self.title_train = "KMedoids Training"
        self.title_test = "KMedoids Testing"
        self.title_x_loc = 0.5
        self.hovermode = "closest"
        self.width = 1200
        self.height = 600

    def plot_training_1d(self, train_data, n_clusters, hist_labels, hist_centroids):
        """Performs kmedoids training plot 1d

        Args:
            train_data (numpy): Train data to visualise
            n_clusters (int): Number of cluster
            hist_labels (nump): Historical list of predicted labels at each iteration
            hist_centroids (numpy): Histroical list of centroids at each iteration
        """

        # Range of feature space to visualise
        x_min, y_min = train_data.min(axis=0)[0] - 1, -1
        x_max, y_max = train_data.max(axis=0)[0] + 1, 1

        # Plots animation frames
        frame = []
        for k in range(len(hist_centroids)):
            frame_ind = []
            for n_cluster in range(n_clusters):

                # Plot training points with cluster labels
                x = list(np.array(train_data[hist_labels[k] == n_cluster])[:,0])
                y = [0]*sum(hist_labels[k] == n_cluster)
                frame_ind.append(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="markers",
                        marker_size=self.data_size,
                        marker_color=self.color_coding[n_cluster],
                        marker_opacity=self.opacity,
                        name=f'Training Points'
                        )
                    )

                # Plot centroids with cluster labels
                x = list(hist_centroids[k][n_cluster][[0]])
                y = [0]
                frame_ind.append(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="markers",
                        marker_size=self.centroid_size,
                        marker_color=self.color_coding[n_cluster],
                        marker_symbol="x",
                        name=f"Centroid {n_cluster+1}"
                        )
                    )

            self.sliders_dict["steps"].append({
                "args": [
                    [k],
                    {"frame": {"duration": 700, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0}},
                    ],
                "label": k,
                "method": "animate",
                })

            frame.append(
                go.Frame(
                    data=frame_ind,
                    name=k
                ))

        # Display first frame when plot is displayed
        frame_1 = frame[0]['data']

        # Initalise and display figure object
        fig = go.Figure(
            data=frame_1,
            layout=go.Layout(
                xaxis=dict(range=[x_min, x_max], autorange=self.autorange, title=self.title_x),
                yaxis=dict(range=[y_min, y_max], autorange=self.autorange, title=self.title_y),
                title_text=self.title_train, title_x=self.title_x_loc, hovermode=self.hovermode,
                width=self.width, height=self.height,
                updatemenus=self.updatemenus
                ),
            frames=frame,
            )

        # Intialise slider parameter to figure
        fig["layout"]["sliders"] = [self.sliders_dict]

        fig.show()

    def plot_training_2d(self, train_data, n_clusters, hist_labels, hist_centroids):
        """Performs kmedoids training plot 2d

        Args:
            train_data (numpy): Train data to visualise
            n_clusters (int): Number of cluster
            hist_labels (numpy): Historical list of predicted labels at each iteration
            hist_centroids (numpy): Histroical list of centroids at each iteration
        """

        # Range of feature space to visualise
        x_min, y_min = train_data.min(axis=0) - 1
        x_max, y_max = train_data.max(axis=0) + 1

        # Plots animation frames
        frame = []
        for k in range(len(hist_centroids)):
            frame_ind = []
            for n_cluster in range(n_clusters):

                # Plot training points with cluster labels
                x = list(np.array(train_data[hist_labels[k] == n_cluster])[:,0])
                y = list(np.array(train_data[hist_labels[k] == n_cluster])[:,1])
                frame_ind.append(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="markers",
                        marker_size=self.data_size,
                        marker_color=self.color_coding[n_cluster],
                        marker_opacity=self.opacity,
                        name=f'Training Points'
                        )
                    )

                # Plot centroids with cluster labels
                x = list(hist_centroids[k][n_cluster][[0]])
                y = list(hist_centroids[k][n_cluster][[1]])
                frame_ind.append(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="markers",
                        marker_size=self.centroid_size,
                        marker_color=self.color_coding[n_cluster],
                        marker_symbol="x",
                        name=f"Centroid {n_cluster+1}"
                        )
                    )

            self.sliders_dict["steps"].append({
                "args": [
                    [k],
                    {"frame": {"duration": 700, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0}},
                    ],
                "label": k,
                "method": "animate",
                })

            frame.append(
                go.Frame(
                    data=frame_ind,
                    name=k
                ))

        # Display first frame when plot is displayed
        frame_1 = frame[0]['data']

        # Initalise and display figure object
        fig = go.Figure(
            data=frame_1,
            layout=go.Layout(
                xaxis=dict(range=[x_min, x_max], autorange=self.autorange, title=self.title_x),
                yaxis=dict(range=[y_min, y_max], autorange=self.autorange, title=self.title_y),
                title_text=self.title_train, title_x=self.title_x_loc, hovermode=self.hovermode,
                width=self.width, height=self.height,
                updatemenus=self.updatemenus
                ),
            frames=frame,
            )

        # Intialise slider parameter to figure
        fig["layout"]["sliders"] = [self.sliders_dict]

        fig.show()

    def plot_training_3d(self, train_data, n_clusters, hist_labels, hist_centroids):
        """Performs kmedoids training plot 3d

        Args:
            train_data (numpy): Train data to visualise
            n_clusters (int): Number of cluster
            hist_labels (nump): Historical list of predicted labels at each iteration
            hist_centroids (numpy): Histroical list of centroids at each iteration
        """

        # Range of feature space to visualise
        x_min, y_min, z_min = train_data.min(axis=0) - 1
        x_max, y_max, z_max = train_data.max(axis=0) + 1

        # Plots animation frames
        frame = []
        for k in range(len(hist_centroids)):
            frame_ind = []
            for n_cluster in range(n_clusters):

                # Plot training points with cluster labels
                x = list(np.array(train_data[hist_labels[k] == n_cluster])[:,0])
                y = list(np.array(train_data[hist_labels[k] == n_cluster])[:,1])
                z = list(np.array(train_data[hist_labels[k] == n_cluster])[:,2])
                frame_ind.append(
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode="markers",
                        marker_size=self.data_size - 1,
                        marker_color=self.color_coding[n_cluster],
                        marker_opacity=self.opacity,
                        name=f'Training Points'
                        )
                    )

                # Plot centroids with cluster labels
                x = list(hist_centroids[k][n_cluster][[0]])
                y = list(hist_centroids[k][n_cluster][[1]])
                z = list(hist_centroids[k][n_cluster][[2]])
                frame_ind.append(
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode="markers",
                        marker_size=self.centroid_size - 4,
                        marker_color=self.color_coding[n_cluster],
                        marker_symbol="x",
                        name=f"Centroid {n_cluster+1}"
                        )
                    )

            self.sliders_dict["steps"].append({
                "args": [
                    [k],
                    {"frame": {"duration": 700, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0}},
                    ],
                "label": k,
                "method": "animate",
                })

            frame.append(
                go.Frame(
                    data=frame_ind,
                    name=k
                ))

        # Display first frame when plot is displayed
        frame_1 = frame[0]['data']

        # Initalise and display figure object
        fig = go.Figure(
            data=frame_1,
            layout=go.Layout(
                scene = dict(xaxis=dict(range=[x_min, x_max], autorange=self.autorange, title=self.title_x),
                             yaxis=dict(range=[y_min, y_max], autorange=self.autorange, title=self.title_y),
                             zaxis=dict(range=[z_min, z_max], autorange=self.autorange, title=self.title_z)),
                title_text=self.title_train, title_x=self.title_x_loc, hovermode=self.hovermode,
                width=self.width + 300, height=self.height + 100,
                updatemenus=self.updatemenus
                ),
            frames=frame,
            )

        # Intialise slider parameter to figure
        fig["layout"]["sliders"] = [self.sliders_dict]

        fig.show()

    def plot_testing_1d(self, train_data, test_data, n_clusters, train_labels, test_labels, centroids):
        """Performs kmedoids testing plot 1d

        Args:
            train_data (numpy): Train data to visualise
            test_data (numpy): Test data to visualise
            n_clusters (int): Number of cluster
            train_labels (numpy): Training target labels
            test_labels (numpy): Testing target labels
            centroids (numpy): Centroids
        """

        # Range of feature space to visualise
        x_min, y_min = np.vstack((train_data, test_data)).min(axis=0)[0] - 1, -1
        x_max, y_max = np.vstack((train_data, test_data)).max(axis=0)[0] + 1, 1

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
                    name=f'Training Points'
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
                    marker_symbol="cross",
                    marker_opacity=self.opacity,
                    name=f'Testing Points'
                    )
                )

            # Plot centroids with cluster labels
            x = list(centroids[n_cluster][[0]])
            y = [0]
            frame.append(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker_size=self.centroid_size,
                    marker_color=self.color_coding[n_cluster],
                    marker_symbol='x',
                    name=f"Centroid {n_cluster+1}"
                    )
                )

        # Initalise and display figure object
        fig = go.Figure(
            data=frame,
            layout=go.Layout(
                xaxis=dict(range=[x_min, x_max], autorange=self.autorange, title=self.title_x),
                yaxis=dict(range=[y_min, y_max], autorange=self.autorange, title=self.title_y),
                title_text=self.title_test, title_x=self.title_x_loc, hovermode=self.hovermode,
                width=self.width, height=self.height
                )
            )

        fig.show()

    def plot_testing_2d(self, train_data, test_data, n_clusters, train_labels, test_labels, centroids):
        """Performs kmedoids testing plot 2d

        Args:
            train_data (numpy): Train data to visualise
            test_data (numpy): Test data to visualise
            n_clusters (int): Number of cluster
            train_labels (numpy): Training target labels
            test_labels (numpy): Testing target labels
            centroids (numpy): Centroids
        """

        # Range of feature space to visualise
        x_min, y_min = np.vstack((train_data, test_data)).min(axis=0) - 1
        x_max, y_max = np.vstack((train_data, test_data)).max(axis=0) + 1

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
                    name=f'Training Points'
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
                    marker_symbol="cross",
                    marker_opacity=self.opacity,
                    name=f'Testing Points'
                    )
                )

            # Plot centroids with cluster labels
            x = list(centroids[n_cluster][[0]])
            y = list(centroids[n_cluster][[1]])
            frame.append(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker_size=self.centroid_size,
                    marker_color=self.color_coding[n_cluster],
                    marker_symbol='x',
                    name=f"Centroid {n_cluster+1}"
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

    def plot_testing_3d(self, train_data, test_data, n_clusters, train_labels, test_labels, centroids):
        """Performs kmedoids testing plot 3d

        Args:
            train_data (numpy): Train data to visualise
            test_data (numpy): Test data to visualise
            n_clusters (int): Number of cluster
            train_labels (numpy): Training target labels
            test_labels (numpy): Testing target labels
            centroids (numpy): Centroids
        """

        # Range of feature space to visualise
        x_min, y_min, z_min = np.vstack((train_data, test_data)).min(axis=0) - 1
        x_max, y_max, z_max = np.vstack((train_data, test_data)).max(axis=0) + 1

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
                    name=f'Training Points'
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
                    marker_symbol="cross",
                    marker_opacity=self.opacity,
                    name=f'Testing Points'
                    )
                )


            # Plot centroids with cluster labels
            x = list(centroids[n_cluster][[0]])
            y = list(centroids[n_cluster][[1]])
            z = list(centroids[n_cluster][[2]])
            frame.append(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker_size=self.centroid_size-4,
                    marker_color=self.color_coding[n_cluster],
                    marker_symbol='x',
                    name=f"Centroid {n_cluster+1}"
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