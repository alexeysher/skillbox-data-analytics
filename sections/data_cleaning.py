import streamlit as st
import pandas as pd
import plotly.express as px
from auxiliary import wrap_text, MegafonColors

@st.cache_resource(show_spinner='Plotting...')
def plot_csat_dist_10(data: pd.DataFrame):
    fig = px.histogram(data, x='Q1', histnorm='percent', opacity=0.5)
    fig.update_traces(texttemplate="%{y:.1f}%", hovertemplate='%{x} - %{y:.1f}%',
                      marker_color=px.colors.DEFAULT_PLOTLY_COLORS)
    fig.update_layout(font_family="Calibri", font_color=MegafonColors.scantBlue2, font_size=14,
                      # title=None, title_x=0.5, title_y=0.91, title_xanchor='center',
                      # title_font_size=20, title_font_color=MegafonColors.scantBlue2,
                      height=400,
                      bargap=0.2, margin_l=0, margin_r=0, margin_b=0, margin_t=0)
    fig.update_xaxes(title='', tickvals=data['Q1'].sort_values().unique(),
                     title_font_color=MegafonColors.brandPurple, tickfont_size=14)
    fig.update_yaxes(title='',
                     title_font_color=MegafonColors.brandPurple, tickfont_size=14)
    return fig


@st.cache_resource(show_spinner='Plotting...')
def plot_csat_dist_5(data: pd.DataFrame):
    s = (data['Q1'] + 1) // 2
    fig = px.histogram(s, x='Q1', histnorm='percent', opacity=0.5)
    fig.update_traces(texttemplate="%{y:.1f}%", hovertemplate='%{x} - %{y:.1f}%',
                      marker_color=px.colors.DEFAULT_PLOTLY_COLORS)
    fig.update_layout(font_family="Calibri", font_color=MegafonColors.content, font_size=14,
                      # title='5-бальная шкала', title_x=0.5, title_y=0.91, title_xanchor='center',
                      # title_font_size=20, title_font_color=MegafonColors.scantBlue2,
                      height=400,
                      bargap=0.2, margin_l=0, margin_r=0, margin_b=0, margin_t=0)
    fig.update_xaxes(title='', title_font_color=MegafonColors.brandPurple, tickfont_size=14)
    fig.update_yaxes(title='', title_font_color=MegafonColors.brandPurple, tickfont_size=14)
    return fig


@st.cache_resource(show_spinner='Plotting...')
def plot_reason_dist(data: pd.DataFrame):
    s = pd.Series(index=[
        "0 - Unknown", "1 - Missed calls, disconnected calls",
        "2 - Waiting time for ringtones",
        "3 - Poor connection quality in buildings, shopping centers, etc.",
        "4 - Slow mobile Internet", "5 - Slow video loading",
        "6 - Difficult to answer", "7 - Your own option"
    ], dtype=float)
    s.index = s.index.map(lambda x: wrap_text(x, 40))
    for index in range(s.size):
        s.iloc[index] = data[data['Q1'] <= 8]['Q2'].str.contains(str(index)).sum()
    s = s / s.sum() * 100

    fig = px.bar(s, orientation='h', opacity=0.5)
    fig.update_traces(texttemplate="%{x:.1f}%", hovertemplate='%{y} - %{x:.1f}%',
                      marker_color=px.colors.DEFAULT_PLOTLY_COLORS)
    fig.update_layout(font_family="Calibri", font_color=MegafonColors.brandPurple, font_size=14,
                      # title='Исходные', title_x=0.5, title_y=0.95, title_xanchor='center',
                      # title_font_size=20, title_font_color=MegafonColors.scantBlue2,
                      showlegend=False, height=450,
                      bargap=0.3, margin_l=0, margin_r=0, margin_t=0, margin_b=0)
    fig.update_xaxes(title='', title_font_color=MegafonColors.brandPurple, tickfont_size=16)
    fig.update_yaxes(title='', title_font_color=MegafonColors.brandPurple, tickfont_size=16)
    return fig


@st.cache_resource(show_spinner='Plotting...')
def plot_reason_combo_dist(data: pd.DataFrame):
    s = pd.Series(dtype=float)
    s.loc["0, 6, 7 - Unknown"] = (data['Q2'].str.contains('[067]', regex=True) & (data['Q1'] <= 8)).sum()
    s.loc["1, 2 - Voice communication"] = data['Q2'].str.contains('[12]', regex=True).sum()
    s.loc["3 - Coverage"] = data['Q2'].str.contains('3').sum()
    s.loc["4, 5 - Mobile Internet"] = data['Q2'].str.contains('[45]', regex=True).sum()
    s = s / s.sum() * 100

    fig = px.bar(s, orientation='h', opacity=0.5)
    fig.update_traces(texttemplate="%{x:.1f}%", hovertemplate='%{y} - %{x:.1f}%',
                      marker_color=px.colors.DEFAULT_PLOTLY_COLORS)
    fig.update_layout(font_family="Calibri", font_color=MegafonColors.brandPurple, font_size=16,
                      # title='Объединенные', title_x=0.5, title_y=0.95, title_xanchor='center',
                      # title_font_size=20, title_font_color=MegafonColors.scantBlue2,
                      showlegend=False, height=250,
                      bargap=0.3, margin_l=0, margin_r=0, margin_t=0, margin_b=0)
    fig.update_xaxes(title='', title_font_color=MegafonColors.brandPurple, tickfont_size=16)
    fig.update_yaxes(title='', title_font_color=MegafonColors.brandPurple, tickfont_size=16)
    return fig


st.markdown(
    '''
    # Data cleaning
    
    ## Removing unnecessary data
    
    The dataset contains a user identifier field `user_id`. 
    This information is not needed for analysis - it should be removed.
    
    ```
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3112 entries, 0 to 3111
    Data columns (total 10 columns):
     #   Column                                     Non-Null Count  Dtype  
    ---  ------                                     --------------  -----  
     0   Q1                                         3110 non-null   object 
     1   Q2                                         1315 non-null   object 
     2   Total Traffic(MB)                          3112 non-null   float64
     3   Downlink Throughput(Kbps)                  3112 non-null   float64
     4   Uplink Throughput(Kbps)                    3112 non-null   float64
     5   Downlink TCP Retransmission Rate(%)        3112 non-null   float64
     6   Video Streaming Download Throughput(Kbps)  3112 non-null   float64
     7   Video Streaming xKB Start Delay(ms)        3112 non-null   int64  
     8   Web Page Download Throughput(Kbps)         3112 non-null   float64
     9   Web Average TCP RTT(ms)                    3112 non-null   int64  
    dtypes: float64(6), int64(2), object(2)
    ```
    
    ## Cleaning answers to the 1st question
    
    To leave only valid answers for the 1st question it's need to do the following:
    1. Exclude records whose values in the `Q1` field are not a text representation of an `integer`.
    2. Convert the `Q1` field to an `integer` type.
    3. Exclude records whose values in the `Q1` field are out of the valid range from `1` to `10`.

    ```
    <class 'pandas.core.frame.DataFrame'>
    Index: 3058 entries, 0 to 3111
    Data columns (total 10 columns):
     #   Column                                     Non-Null Count  Dtype  
    ---  ------                                     --------------  -----  
     0   Q1                                         3058 non-null   int32  
     1   Q2                                         1315 non-null   object 
     2   Total Traffic(MB)                          3058 non-null   float64
     3   Downlink Throughput(Kbps)                  3058 non-null   float64
     4   Uplink Throughput(Kbps)                    3058 non-null   float64
     5   Downlink TCP Retransmission Rate(%)        3058 non-null   float64
     6   Video Streaming Download Throughput(Kbps)  3058 non-null   float64
     7   Video Streaming xKB Start Delay(ms)        3058 non-null   int64  
     8   Web Page Download Throughput(Kbps)         3058 non-null   float64
     9   Web Average TCP RTT(ms)                    3058 non-null   int64  
    dtypes: float64(6), int32(1), int64(2), object(1)
    ```
    
    After removing incorrect answers to the 1st question, there are `3058` records left in the Dataset.

    ## Cleaning answers to the 2nd question
    
    To leave only valid answers for the 2nd question it's need to do the following:
    1. Exclude records whose values in the `Q2` field are not a text representation 
    of list of comma-separated `integers`.
    2. Exclude records whose values in the `Q1` field are out of the valid range from `0` to `7`.
    3. Exclude reasons `0` ("Unknown") or `6` ("Difficult to answer") from answers 
    if customer point any other reasons `1`...`5` or `7`.
      
    ```
    <class 'pandas.core.frame.DataFrame'>
    Index: 3057 entries, 0 to 2806
    Data columns (total 10 columns):
     #   Column                                     Non-Null Count  Dtype  
    ---  ------                                     --------------  -----  
     0   Q1                                         3057 non-null   int32  
     1   Q2                                         3057 non-null   object 
     2   Total Traffic(MB)                          3057 non-null   float64
     3   Downlink Throughput(Kbps)                  3057 non-null   float64
     4   Uplink Throughput(Kbps)                    3057 non-null   float64
     5   Downlink TCP Retransmission Rate(%)        3057 non-null   float64
     6   Video Streaming Download Throughput(Kbps)  3057 non-null   float64
     7   Video Streaming xKB Start Delay(ms)        3057 non-null   int64  
     8   Web Page Download Throughput(Kbps)         3057 non-null   float64
     9   Web Average TCP RTT(ms)                    3057 non-null   int64  
    dtypes: float64(6), int32(1), int64(2), object(1)
    ```
    
    After removing incorrect answers, there are `3057` records left in the Dataset.

    ## Cleaning internet metrics
    
    It's needed to exclude values in the metric fields which are less or equal zero.
          
    ```
    <class 'pandas.core.frame.DataFrame'>
    Index: 3054 entries, 0 to 2806
    Data columns (total 10 columns):
     #   Column                                     Non-Null Count  Dtype  
    ---  ------                                     --------------  -----  
     0   Q1                                         3054 non-null   int32  
     1   Q2                                         3054 non-null   object 
     2   Total Traffic(MB)                          3054 non-null   float64
     3   Downlink Throughput(Kbps)                  3054 non-null   float64
     4   Uplink Throughput(Kbps)                    3054 non-null   float64
     5   Downlink TCP Retransmission Rate(%)        3054 non-null   float64
     6   Video Streaming Download Throughput(Kbps)  3054 non-null   float64
     7   Video Streaming xKB Start Delay(ms)        3054 non-null   int64  
     8   Web Page Download Throughput(Kbps)         3054 non-null   float64
     9   Web Average TCP RTT(ms)                    3054 non-null   int64  
    dtypes: float64(6), int32(1), int64(2), object(1)
    ```

    After removing incorrect answers, there are `3054` records left in the Dataset.
    '''
)
