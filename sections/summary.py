import streamlit as st
import plotly.express as px
from functions import display_confidence_interval

st.markdown(
    f'''
    # Summary
    
    Within this work, a research of the survey of Megafon customers was done. 
    As a result, it was found that the level of customer satisfaction `CSAT` with the Mobile Internet service 
    should be determined on a 4-point scale:
    
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[0]};opacity:0.5">
      &nbsp;&nbsp;1&nbsp;&nbsp;</span>&nbsp;&nbsp;Completely dissatisfied
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[1]};opacity:0.5">
      &nbsp;&nbsp;2&nbsp;&nbsp;</span>&nbsp;&nbsp;Partially dissatisfied
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[2]};opacity:0.5">
      &nbsp;&nbsp;3&nbsp;&nbsp;</span>&nbsp;&nbsp;Partially satisfied
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[3]};opacity:0.5">
      &nbsp;&nbsp;4&nbsp;&nbsp;</span>&nbsp;&nbsp;Completely satisfied.

    The `CSAT` is primarily affected by `Video Streaming Download Throughput(Kbps)`. The depences between `CSAT` and the statistics of this metric is clearly linear - the difference in the central values of the statistics for customers with neighboring `CSAT` is approximately `847` Kbps.
    
    For each category of customers with a certain `CSAT` confidence interval of the statistic were determined:
    ''', unsafe_allow_html=True
)

ci = st.session_state.section_9_3[3]

table = display_confidence_interval(
    ci['Video Streaming Download Throughput(Kbps)'],
    metrics=st.session_state.research_metrics.loc['Video Streaming Download Throughput(Kbps)'],
    caption='', caption_font_size=12, opacity=0.5, precision=1, index_width=30)
s = display_confidence_interval(ci['Video Streaming Download Throughput(Kbps)'],
                                metrics=st.session_state.research_metrics.loc['Video Streaming Download Throughput(Kbps)'],
                                caption='', caption_font_size=12, opacity=0.5, precision=1,
                                index_width=30, col_width=105)
st.markdown(s.to_html(table_uuid="table_pvalues_9"), unsafe_allow_html=True)
