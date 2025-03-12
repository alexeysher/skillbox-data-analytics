import streamlit as st


st.markdown(
    '''
    # Provided data
    ## Dataset description
    The survey data locates in the file `megafon.csv` with the following fields:
    | name | description |
    | ---- | ----------- |
    |`user_id` | user id |
    |`Q1` | answer to 1st question |
    |`Q2` | answer to 2nd question |
    |`Total Traffic(MB)` | traffic total volume <sup>1</sup> |
    |`Downlink Throughput(Kbps)` | average downlink speed <sup>2</sup> | 
    |`Uplink Throughput(Kbps)` | average uplink speed <sup>3</sup> | 
    |`Downlink TCP Retransmission Rate(%)` | frequency of downlink packets retransmission<sup>4 </sup> | 
    |`Video Streaming Download Throughput(Kbps)` | streaming video download speed <sup>5 </sup> | 
    |`Video Streaming xKB Start Delay(ms)` | delay start of video playback <sup>6 </sup> | 
    |`Web Page Download Throughput(Kbps)` | web page loading speed via browser <sup>7 </sup> | 
    |`Web Average TCP RTT(ms)` | ping when browsing web sections<sup>8 </sup> |
    
    <sup>1 </sup> — Indicates how actively the subscriber uses the mobile Internet.  
    <sup>2 </sup> — Calculates over all traffic.  
    <sup>3 </sup> — Calculates over all traffic.  
    <sup>4 </sup> — More is worse (less effective speed).  
    <sup>5 </sup> — More is better (less lag and better picture quality).  
    <sup>6 </sup> — The time between pressing the Play button and the start of video playback. The shorter this time, the faster the playback starts.  
    <sup>7 </sup> — The more the better. 
    <sup>8 </sup> — The less the better (web sections are loading faster).
    
    The first metric is given for a week before the survey. The other metrics indicates average value for a week before the survey
    
    ## Observation the dataset
    Will load the user survey data from the `megafon.csv` file into the dataset 
    and display information about the content.
    
    ```
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3112 entries, 0 to 3111
    Data columns (total 11 columns):
     #   Column                                     Non-Null Count  Dtype  
    ---  ------                                     --------------  -----  
     0   user_id                                    3112 non-null   int64  
     1   Q1                                         3110 non-null   object 
     2   Q2                                         1315 non-null   object 
     3   Total Traffic(MB)                          3112 non-null   float64
     4   Downlink Throughput(Kbps)                  3112 non-null   float64
     5   Uplink Throughput(Kbps)                    3112 non-null   float64
     6   Downlink TCP Retransmission Rate(%)        3112 non-null   float64
     7   Video Streaming Download Throughput(Kbps)  3112 non-null   float64
     8   Video Streaming xKB Start Delay(ms)        3112 non-null   int64  
     9   Web Page Download Throughput(Kbps)         3112 non-null   float64
     10  Web Average TCP RTT(ms)                    3112 non-null   int64  
    dtypes: float64(6), int64(3), object(2)
    ```
    
    From the dataset viewing the following important information can be obtained:
    - The dataset contains the results of answers and metrics of mobile Internet usage for `3112` customers.
    - The answer to the 1st question was not given by `2` customers (the `Q1` field contains `3110` non-null values).
    - Some customers, when were answering the 1st question, 
      gave answers that were not numeric (the `Q1` field has the `object` type, 
      and if all the answers were integers, the type would be `int64`).
    ''', unsafe_allow_html=True
)