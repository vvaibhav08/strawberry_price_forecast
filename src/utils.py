import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors
import numpy as np
import pandas as pd


CLIMATE_FEATURES = ["windspeed", "temp", "precip", "cloudcover", "solarradiation"]

def plot_feature_target_correlation(df, features_cols, target_col='price'):
    """
    Create a correlation heatmap between features and target variable
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input dataframe
    features_cols : list
        List of feature column names
    target_col : str
        Name of the target variable column
    """
    corr_with_target = df[features_cols].corrwith(df[target_col])
    
    corr_with_target = corr_with_target.sort_values(ascending=False)
    
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=corr_with_target.values.reshape(-1, 1),
        x=['Correlation'],
        y=corr_with_target.index,
        colorscale='Teal',  # Similar to 'GnBu'
        zmid=0,  # Center the color scale at zero
        text=np.round(corr_with_target.values.reshape(-1, 1), 2),
        texttemplate='%{text:.2f}',
        showscale=True,
        colorbar=dict(title='Correlation')
    ))
    fig.update_layout(
        title='Feature Correlation with Price Target',
        height=max(500, 20 * len(features_cols)),  # Dynamic height based on feature count
        width=400,
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(corr_with_target.index))),
            ticktext=corr_with_target.index,
            automargin=True
        )
    )
    
    return fig, corr_with_target


def plot_price_distribution(data):
    min_val = min(data['price_min'].min(), data['price'].min()) 
    max_val = max(data['price_max'].max(), data['price'].max())
    bin_size = (max_val - min_val) / 24
    bins = np.linspace(min_val, max_val, 25)
    price_counts, price_edges = np.histogram(data['price'].dropna(), bins=bins)
    price_min_counts, _ = np.histogram(data['price_min'].dropna(), bins=bins)
    price_max_counts, _ = np.histogram(data['price_max'].dropna(), bins=bins)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data['price'].dropna(),
        name='price',
        opacity=0.7,
        marker_color='rgba(0,204,150,0.8)',
        xbins=dict(
            start=min_val,
            end=max_val,
            size=bin_size
        ),
        autobinx=False
    ))

    # ROOT-style
    fig.add_trace(go.Scatter(
        x=price_edges,
        y=np.append(price_min_counts, 0),
        mode='lines',
        line=dict(color='rgba(255,127,14,1)', width=2, shape='hvh'),
        name='price_min'
    ))
    fig.add_trace(go.Scatter(
        x=price_edges,
        y=np.append(price_max_counts, 0),
        mode='lines',
        line=dict(color='rgba(255,65,54,1)', width=2, shape='hvh'),
        name='price_max'
    ))

    fig.update_layout(
        title='Price Distribution',
        xaxis_title='Price (€/kg)',
        yaxis_title='Count',
        legend=dict(
            x=0.85,
            y=0.95
        )
    )
    fig.update_yaxes(
        showgrid=False,
        ticks="inside",
        tickson="boundaries",
        ticklen=8,
        showline=True,
        linewidth=1,
        mirror=True,
        zeroline=False
    )
    fig.update_xaxes(
        showgrid=False,
        ticks="inside",
        tickson="boundaries",
        ticklen=8,
        showline=True,
        linewidth=1,
        mirror=True,
        zeroline=False
    )
    fig.show()


def plot_price_availability(data):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Price availability by week/year", "Strawberry price series"],
        column_widths=[0.35, 0.65]
    )

    min_price = data['price'].min()
    max_price = data['price'].max()
    price_data_heatmap = data.copy()
    scatter = go.Scatter(
        x=price_data_heatmap['week'], 
        y=price_data_heatmap['year'],
        mode='markers',
        marker=dict(
            size=6,
            color=price_data_heatmap['price'],
            colorscale='BuGn',
            colorbar=dict(
                title='€/kg',
                x=0.32,
                thickness=10
            ),
            cmin=min_price,
            cmax=max_price
        ),
        text='€/kg',
        showlegend=False
    )
    fig.add_trace(scatter, row=1, col=1)

    price_data_ts = data.dropna(subset=['price']).copy()
    price_data_ts['year_week'] = price_data_ts['year'].astype(str) + '-' + price_data_ts['week'].astype(str).str.zfill(2)
    price_data_ts['year_week'] = pd.to_datetime(price_data_ts['year_week'] + '-0', format='%Y-%W-%w')
    price_data_ts = price_data_ts.sort_values('year_week')
    line = go.Scatter(
        x=price_data_ts['year_week'], 
        y=price_data_ts['price'],
        mode='lines+markers',
        name='Strawberry Price',
        line=dict(color='#00b894', width=2),
        marker=dict(size=6),
    )
    fig.add_trace(line, row=1, col=2)


    fig.update_layout(
        height=500,
        width=1200,
        title_text="Strawberry price seasonality",
        margin=dict(t=100, b=50),
        showlegend=False
    )
    fig.update_xaxes(title_text="# Week", row=1, col=1)
    fig.update_yaxes(title_text="Year", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(title_text="Price", row=1, col=2)
    fig.show()


def plot_feature_target_relation(data: pd.DataFrame, colors = plotly.colors.qualitative.Plotly):
    years = range(2013, 2023)
    for feature in CLIMATE_FEATURES:
        fig = go.Figure()
        for i, year in enumerate(years):
            color = colors[i % len(colors)] 
            year_df = data[data['year'] == year].sort_values(by='week').copy()
            year_df['price_available'] = year_df['price'].notna()
            year_df['segment_id'] = (year_df['price_available'] != year_df['price_available'].shift()).cumsum()

            # for segment_id, segment_df in year_df.groupby('segment_id'):
            segments = list(year_df.groupby('segment_id'))
            for seg_index, (segment_id, segment_df) in enumerate(segments):
                x_data = segment_df['week'].values
                y_data = segment_df[feature].values

                if segment_df['price_available'].iloc[0]:
                    for j in range(len(x_data) - 1):
                        fig.add_trace(go.Scatter(x=[x_data[j], x_data[j + 1]],
                                                y=[y_data[j], y_data[j + 1]],
                                                mode='lines',
                                                line=dict(width=segment_df['price'].iloc[j],  # Scaling factor
                                                        dash='solid',
                                                        color=color),
                                                name=str(year) if segment_id == 1 and j == 0 else None,
                                                showlegend=segment_id == 1 and j == 0))
                else:
                    connect_x = []
                    connect_y = []

                    # added jargon for nicer looking plots
                    # Add the last point from the previous segment so things appear nice and connected
                    if seg_index > 0:
                        prev_segment_df = segments[seg_index - 1][1]
                        if prev_segment_df['price_available'].iloc[0]:
                            connect_x.append(prev_segment_df['week'].iloc[-1])
                            connect_y.append(prev_segment_df[feature].iloc[-1])

                    connect_x.extend(x_data)
                    connect_y.extend(y_data)
                    if seg_index < len(segments) - 1:
                        next_segment_df = segments[seg_index + 1][1]
                        if next_segment_df['price_available'].iloc[0]:
                            connect_x.append(next_segment_df['week'].iloc[0])
                            connect_y.append(next_segment_df[feature].iloc[0])

                    fig.add_trace(go.Scatter(x=connect_x,
                                            y=connect_y,
                                            mode='lines',
                                            line=dict(width=1, dash='dot', color=color),
                                            name=str(year) if seg_index == 0 else None,
                                            showlegend=False))


        fig.update_layout(title=f'{feature} | year on year | (line thickness represents price)',
                        xaxis_title='Week Number',
                        yaxis_title=feature)
        fig.show()