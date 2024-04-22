import math

import numpy as np
import pandas as pd

import plotly.express as px

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import emgts

def plot_trial_channel(
        dataset_dir,
        dataset_type,
        user_id,
        trial_id,
        channel_id,
        offset=0,
        limit=None,
        preprocess=None,
        segment=None,
        show_title=False,
        output_dir=None,
        width=1000,
        scale=3,
):
    '''Plot single channel from one sEMG trial'''
    trial_set = emgts.EmgTrialSet(dataset_dir, user_id, dataset_type)
    
    sampling_rate = trial_set.sampling_rate
    trial_data = trial_set.get_trial(trial_id)
    trial_data_length = trial_set.get_trial_data_length(trial_id)
    trial_label = trial_set.get_trial_label(trial_id)
    data = trial_data[0:trial_data_length]
    
    if offset >= trial_data_length or offset < 0:
        offset = 0
    
    if not limit or limit < 0:
        limit = trial_data_length
        
    if offset + limit > trial_data_length:
        limit = trial_data_length - offset

    df = pd.DataFrame(dict(
        t = np.arange(offset, offset+limit) / sampling_rate,
    ))

    df['sEMG'] = data[offset:offset+limit,channel_id]
    y = ['sEMG']

    if preprocess:
        preprocessed_data = preprocess(data, sampling_rate)
        df['Preprocessed sEMG'] = preprocessed_data[offset:offset+limit,channel_id]
        y.append('Preprocessed sEMG')

    if show_title:
        title = 'User ID = {U}, Dataset type = {DST}, Trial ID = {T}, label = {L}'.format(
            U=user_id,
            DST=dataset_type,
            T=trial_id,
            L=trial_label,
        )
    else:
        title = None

    fig = px.line(df, x='t', y=y)
    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        legend_title=None,
        legend=dict(
            xanchor='right',
            x=0.99,
            yanchor='top',
            y=0.97,
        ),
        font=dict(
            family="Times",
            size=16,
        ),
        width=width,
        title_text=title,
        title_x=0.5,
    )

    if segment:
        start_index, end_index = segment(preprocessed_data, sampling_rate)
        fig.add_vrect(
            x0=start_index/sampling_rate,
            x1=end_index/sampling_rate,
            fillcolor='LightSeaGreen',
            opacity=0.2,
            layer='below',
            line_width=0,
        )

    if output_dir:
        fig.write_image(output_dir, engine='kaleido', scale=scale, width=width)
    else:
        fig.show()

def plot_trial(
        dataset_dir,
        dataset_type,
        user_id,
        trial_id,
        offset=0,
        limit=None,
        preprocess=None,
        segment=None,
        show_title=False,
        output_dir=None,
        width=1000,
        height=1200,
        scale=3,
):
    '''Plot all channels from one sEMG trial'''

    trial_set = emgts.EmgTrialSet(dataset_dir, user_id, dataset_type)
    
    sampling_rate = trial_set.sampling_rate
    trial_data = trial_set.get_trial(trial_id)
    trial_data_length = trial_set.get_trial_data_length(trial_id)
    trial_label = trial_set.get_trial_label(trial_id)
    data = trial_data[0:trial_data_length]

    if preprocess:
        preprocessed_data = preprocess(data, sampling_rate)
    
    if offset >= trial_data_length or offset < 0:
        offset = 0
    
    if not limit or limit < 0:
        limit = trial_data_length
        
    if offset + limit > trial_data_length:
        limit = trial_data_length - offset

    df = pd.DataFrame(dict(
        t = np.arange(offset, offset+limit) / sampling_rate,
    ))

    number_of_channels = data.shape[1]
    get_channel_name = lambda channel_id : 'Channel %d' % int(channel_id + 1)
    get_preprocessed_channel_name = lambda channel_id : '%s_preprocessed' % get_channel_name(channel_id)
    
    cols = 2
    rows = math.ceil(number_of_channels / cols)
    channels = []
    channel_id = 0
    for col in np.arange(1,cols+1):
        for row in np.arange(1,rows+1):
            ch_name = get_channel_name(channel_id)
            df[ch_name] = data[offset:offset+limit,channel_id]

            if preprocess:
                prep_ch_name = get_preprocessed_channel_name(channel_id)
                df[prep_ch_name] = preprocessed_data[offset:offset+limit,channel_id]

            channels.append({
                'id': channel_id,
                'name': ch_name,
                'preprocessed_name': prep_ch_name if preprocess else None,
                'row': row,
                'col': col,
            })

            channel_id = channel_id + 1

    titles = tuple([channel['name'] for channel in channels])

    fig = make_subplots(rows=rows, cols=cols,
        subplot_titles=titles,
        shared_xaxes=True,
        vertical_spacing=0.1,
        x_title="Time (s)",
    )
    
    for channel in channels:    
        fig.append_trace(
            go.Scatter(x=df.t, y=df[channel['name']]),
            row=channel['row'], col=channel['col'],
        )
        
        if preprocess:
            fig.append_trace(
                go.Scatter(x=df.t, y=df[channel['preprocessed_name']]),
                row=channel['row'], col=channel['col'],
            )

    if show_title:
        title = 'User ID = {U}, Dataset type = {DST}, Trial ID = {T}, label = {L}'.format(
            U=user_id,
            DST=dataset_type,
            T=trial_id,
            L=trial_label,
        )
    else:
        title = None

    fig.update_layout(
        width=width,
        height=height,
        showlegend=False,
        font=dict(
            family="Times",
            size=16,
        ),
        title_text=title,
        title_x=0.5,
    )

    if segment:
        start_index, end_index = segment(preprocessed_data, sampling_rate)
        fig.add_vrect(
            x0=start_index/sampling_rate,
            x1=end_index/sampling_rate,
            fillcolor='LightSeaGreen', #'LightSalmon', #'RoyalBlue',
            opacity=0.2,
            layer='below',
            line_width=0,
        )

    if output_dir:
        fig.write_image(output_dir, engine='kaleido', scale=scale, width=width, height=height)
    else:
        fig.show()
