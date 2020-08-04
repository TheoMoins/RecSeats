import plotly.graph_objects as go
import pandas as pd
import numpy as np


def figure_from_mat(mat):
    """
    Create a plotly heatmap from an array representing a room.
    :param mat: a 2D numpy array
    :return: a heatmap Plotly
    """
    scale = [[0, 'rgb(245,245,245)'],
             [0.25, 'rgb(245,245,245)'],
             [0.25, 'rgb(190,190,190)'],
             [0.50, 'rgb(190,190,190)'],
             [0.50, 'rgb(90,170,255)'],
             [0.75, 'rgb(90,170,255)'],
             [0.75, 'rgb(0,190,0)'],
             [1, 'rgb(0,190,0)']]

    tickvals = [-1, 0, 1, 2]
    ticktext = ["Occupied", "Non-existant", "Available", "Choice"]

    heatmap = go.Heatmap(z=mat,
                         colorscale=scale,
                         colorbar=dict(thickness=20,
                                       tickvals=tickvals,
                                       ticktext=ticktext),
                         zmin=-1.5,
                         zmax=2.5,
                         xgap=1.5,
                         ygap=1.5,
                         showscale=True,
                         transpose=False,
                         hovertemplate='Prob: %{z}<br>Column: %{x}<br>Row: %{y}<extra></extra>'
                         )

    return heatmap


def figure_from_mat_with_model(mat, pred):
    """
    Create a plotly heatmap from an array representing a room and the
    heatmap prediction of a model.
    :param mat: a 2D numpy array
    :param pred: a 2D numpy array
    :return: a Plotly heatmap
    """
    scale = [[0, 'rgb(245,245,245)'],
             [0.25, 'rgb(245,245,245)'],
             [0.25, 'rgb(190,190,190)'],
             [0.50, 'rgb(190,190,190)'],
             [0.50, 'rgb(49,54,149)'],
             [0.535, 'rgb(69,117,170)'],
             [0.57, 'rgb(116,136,209)'],
             [0.60, 'rgb(171,150,233)'],
             [0.62, 'rgb(224,190,248)'],
             [0.63, 'rgb(234,175,184)'],
             [0.67, 'rgb(253,154,107)'],
             [0.705, 'rgb(234,99,87)'],
             [0.74, 'rgb(215,48,59)'],
             [0.7501, 'rgb(200,35,38)'],
             [0.7501, 'rgb(0,190,0)'],
             [1, 'rgb(0,190,0)']]

    for i in range(len(pred)):
        for j in range(len(pred[0])):
            if mat[i][j] > 0:
                mat[i][j] = 0.5 + pred[i][j]

    tickvals = [-1, 0, 1, 2]
    ticktext = ["Occupied", "Non-existant", "Heatmap", "Choice"]

    # Assign an empty figure widget with two traces
    heatmap = go.Heatmap(z=mat,
                         colorscale=scale,
                         colorbar=dict(thickness=20,
                                       tickvals=tickvals,
                                       ticktext=ticktext),
                         zmin=-1.5,
                         zmax=2.5,
                         xgap=1.5,
                         ygap=1.5,
                         showscale=True,
                         transpose=False,
                         hovertemplate='Z: %{z}<br>Column: %{x}<br>Row: %{y}<extra></extra>'
                         )

    return heatmap


def print_room(fig, width=850, height=600, title='Room'):
    """
    print a heatmap of a given room.
    Blue values: occupied seats
    White values: available seats
    :param fig: a heatmap (can be obtained from figure_from_mat or figure_from_mat_with_model)
    :param width: an int indicating the width of the plot
    :param height: an int indicating the height of the plot
    :param title: Title of the plot
    :return: Nothing, just plot the heatmap
    """

    fig.update_layout(
        title=go.layout.Title(
            text=title,
            x=0.5),
        xaxis={"dtick": 1, "visible": True, "side": "top"},
        yaxis={"dtick": 1, "autorange": "reversed", "visible": True},
        showlegend=True,
        width=width, height=height,
        autosize=True)

    fig.show()


def show_feature_matrix(mat, pipeline):
    """
    Return a pandas dataframe that contains the feature matrix
    computed from a given room.
    :param mat: a room input
    :param pipeline: A pipeline object to compute the features
    :return: a pandas dataframe containing the feature matrix
    """
    feature_mat = pipeline.compute_feature_matrix(mat, output_=None)
    columns = []

    if pipeline.beta_position:
        columns.extend(['Beta_x', 'Beta_y', 'Beta_xx', 'Beta_yy', 'Beta_xy'])
    if pipeline.beta_r1:
        columns.extend(['Beta_R1'])
    if pipeline.beta_r2:
        columns.extend(['Beta_R2'])
    if pipeline.beta_r3:
        columns.extend(['Beta_R3'])
    if pipeline.beta_ps:
        columns.extend(['Beta_left', 'Beta_right', 'Beta_l+r', 'Beta_front', 'Beta_back',
                        'Beta_f_corner', 'Beta_b_corners'])

    index = []
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j] == 1:
                index.append("({},{})".format(j, i))

    return pd.DataFrame(data=feature_mat, columns=columns, index=index)


def plot_hist_accuracy_by_clients(accuracies):
    """
    Plot a histogram of the accuracy and the Top-N accuracy for customers
    :param accuracies: a tuple of size 2 that contains the Top 1 and Top N accuracy
    for all customers
    :return: Nothing, just plot the histogram
    """
    hist = go.Histogram(x=100 * accuracies[0],
                        xbins={"start": 0, "end": 100, "size": 5},
                        opacity=0.8,
                        name='Top 1',
                        marker={"line": {"width": 0.8}})

    hist2 = go.Histogram(x=100 * accuracies[1],
                         xbins={"start": 0, "end": 100, "size": 5},
                         opacity=0.8,
                         name='Top 3',
                         marker={"line": {"width": 0.8}})

    hist3 = go.Histogram(x=100 * accuracies[2],
                         xbins={"start": 0, "end": 100, "size": 5},
                         opacity=0.8,
                         name='Top 5',
                         marker={"line": {"width": 0.8}})

    layout = go.Layout(title="Accuracy Distribution",
                       xaxis={"title": "Cross Validation Accuracy (in %)", "showgrid": False},
                       yaxis={"title": "Number of customers", "showgrid": True},
                       bargap=0.2,  # gap between bars of adjacent location coordinates
                       bargroupgap=0.1)  # gap between bars of the same location coordinates

    fig = go.Figure(data=[hist, hist2, hist3], layout=layout)

    fig.show()
