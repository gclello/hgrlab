import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_radar(
    labels,
    values,
    std=None,
    ticks=None,
    range_min=None,
    range_max=None,
    width=1500,
    height=1000,
    scale=3,
    output_path=None,
):
    arg_length_val_msg = 'Arguments labels, values and std must have the same length'
    if len(labels) != len(values):
        raise Exception(arg_length_val_msg)
    
    if std is not None and len(std) != len(labels):
        raise Exception(arg_length_val_msg)
    
    if std is None:
        data1 = values
        data2 = None
    else:
        data1 = values + std
        data2 = values - std

    if range_min is None:
        range_min = np.min(values)*0.95

    if range_max is None:
        range_max = np.max(values)*1.01
    
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'polar'}]*1]*1)

    fig.add_trace(
        go.Scatterpolar(
            r=data1,
            theta=labels,
            fill='toself',
            legendgroup='data_title1',
            name='% accuracy mean + std',
            line_color='deepskyblue',
            marker_line_color="deepskyblue",
            marker_line_width=6,
        ),
        row = 1,
        col = 1,
    )

    if data2 is not None:
        fig.add_trace(
            go.Scatterpolar(
                r=data2,
                theta=labels,
                fill='toself',
                legendgroup='data_title2',
                name='% accuracy mean - std',
                line_color='darkviolet',
                marker_line_color="darkviolet",
                marker_line_width=6,
            ),
            row = 1,
            col = 1,
        )

    fig.update_layout(
        {'legend_orientation':'h'},
        legend= {'itemsizing': 'constant'},
        autosize=False,
        width=width,
        height=height,
        font=dict(
            family="Times New Roman",
            size=30,
            color="Black"
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[range_min, range_max],
                tickvals=ticks,
                linecolor='white',
                linewidth=1,
                tickcolor='red',
                gridcolor='white',
                gridwidth=3,
            ),
            angularaxis = dict(
                showline=True,
                linecolor='gainsboro',
                linewidth=3,
                tickcolor='brown',
                gridcolor='white',
                gridwidth=2,
            ),
        )
    )

    if output_path:
        fig.write_image(output_path, engine='kaleido', scale=scale, width=width)
    else:
        fig.show()
