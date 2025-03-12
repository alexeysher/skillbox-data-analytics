import streamlit as st
import pandas as pd
import plotly.express as px
from matplotlib.pyplot import ticklabel_format

from functions import (display_cat_info, plot_group_size_barchart, plot_metric_histograms,
                       plot_metric_confidence_interval, display_confidence_interval,
                       display_confidence_interval_overlapping, display_pvalues)


@st.cache_resource(show_spinner='Plotting...')
def plot_group_size_barchart_9_3():
    return plot_group_size_barchart(
        research_data,
        title=' ', title_y=0.85,
        # title='<b>Number of customers in the considered groups</b>', title_y=0.85,
        labels_font_size=14,
        axes_tickfont_size=14,
        axes_title_font_size=14,
        width=600,
        height=280
    ).update_layout(margin_t=0, margin_b=0)


@st.cache_resource(show_spinner='Plotting...')
def plot_metric_confidence_interval_9_3():
    return plot_metric_confidence_interval(
        ci, metrics=st.session_state.research_metrics,
        title='', height=263, n_cols=3,
        horizontal_spacing=0.04, vertical_spacing=0.07,
        labels_font_size=16, axes_tickfont_size=14, units_font_size=14
    ).update_layout(margin_t=60, margin_b=60)


@st.cache_resource(show_spinner='Plotting...')
def plot_metric_histograms_9_4_1(null_distributions, statistics, research_metrics, mark_statistic):
    return plot_metric_histograms(
        null_distributions, statistic=statistics, metrics=research_metrics,
        title=f' ', title_y=0.9,
        labels_font_size=16, axes_tickfont_size=14, units_font_size=14,
        height=300, n_cols=3, opacity=0.5,
        histnorm='probability density',
        add_kde=True, add_statistic=True, mark_statistic=mark_statistic,
        horizontal_spacing=0.08, vertical_spacing=0.07
    ).update_layout(margin_t=60, margin_b=60)


@st.cache_resource(show_spinner='Plotting...')
def plot_metric_histograms_9_4_2(null_distributions, statistics, research_metrics, mark_statistic):
    return plot_metric_histograms(
        null_distributions, statistic=statistics, metrics=research_metrics,
        title=f' ', title_y=0.9,
        labels_font_size=16, axes_tickfont_size=14, units_font_size=14,
        height=300, n_cols=3, opacity=0.5,
        histnorm='probability density',
        add_kde=True, add_statistic=True, mark_statistic=mark_statistic,
        horizontal_spacing=0.08, vertical_spacing=0.07
    ).update_layout(margin_t=60, margin_b=60)


@st.cache_resource(show_spinner='Plotting...')
def plot_metric_histograms_9_4_3(null_distributions, statistics, research_metrics, mark_statistic):
    return plot_metric_histograms(
        null_distributions, statistic=statistics, metrics=research_metrics,
        title=f' ', title_y=0.9,
        labels_font_size=16, axes_tickfont_size=14, units_font_size=14,
        height=300, n_cols=3, opacity=0.5,
        histnorm='probability density',
        add_kde=True, add_statistic=True, mark_statistic=mark_statistic,
        horizontal_spacing=0.08, vertical_spacing=0.07
    ).update_layout(margin_t=60, margin_b=60)


st.markdown(
    f'''
    # Research of the influence of metrics on the customer satisfaction level with Mobile Internet service

    ## Objective of the research

    As a result of the previous research, it was established that customers should be divided into 
    four categories by the level of `CSAT`.

    As part of this research, Let's check that such a division is statistically correct, 
    i.e. customers of different categories belong to different populations, the metrics 
    of which have statistically different values. Moreover, the higher the customer satisfaction category, 
    the higher the customer satisfaction index is the index of differences in metrics.

    In addition, will try to determine which of the metrics under research has the greatest impact on `CSAT`.

    To perform the research, will divide customers into groups by the value of `CSAT`. 
    Will assign a designation corresponding to the `CSAT` value and color scheme 
    for displaying graphic and tabular data to these groups:
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[0]};opacity:0.5">
      &nbsp;&nbsp;1&nbsp;&nbsp;</span> - Completely dissatisfied;
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[1]};opacity:0.5">
      &nbsp;&nbsp;2&nbsp;&nbsp;</span> - Partially dissatisfied;
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[2]};opacity:0.5">
      &nbsp;&nbsp;3&nbsp;&nbsp;</span> - Partially satisfied;
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[3]};opacity:0.5">
      &nbsp;&nbsp;4&nbsp;&nbsp;</span> - Completely satisfied.

    These groups are illustrated on the customer distribution map.

    ### Customers distribution map
    ''', unsafe_allow_html=True
)

s = display_cat_info(st.session_state.data_clean).set_properties(
    pd.IndexSlice['Very dissatisfied': 'Dissatisfied', 'Internet and Video'],
    color='white', background=px.colors.DEFAULT_PLOTLY_COLORS[0], opacity=0.5
).set_properties(
    pd.IndexSlice['Very dissatisfied': 'Dissatisfied', 'Internet': 'Video'],
    color='white', background=px.colors.DEFAULT_PLOTLY_COLORS[1], opacity=0.5
).set_properties(
    pd.IndexSlice['Neutral': 'Satisfied', 'Internet and Video'],
    color='white', background=px.colors.DEFAULT_PLOTLY_COLORS[2], opacity=0.5
).set_properties(
    pd.IndexSlice['Neutral': 'Satisfied', 'Internet': 'Video'],
    color='white', background=px.colors.DEFAULT_PLOTLY_COLORS[2], opacity=0.5
).set_properties(
    pd.IndexSlice['Very satisfied', 'No'], color='white',
    background=px.colors.DEFAULT_PLOTLY_COLORS[3], opacity=0.5
)
st.markdown(s.to_html(table_uuid="table_categories_dist_9_1"), unsafe_allow_html=True)

st.markdown(
    '''
    ## Exploratory analysis

    First, will look at the number of customers in groups utilizing a bar chart.
    '''
)

research_data, groups, group_pairs, ci, ci_overlapping, ci_center = st.session_state.section_9_3
fig = plot_group_size_barchart_9_3()
st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=False)

st.markdown(
    '''
    As can see, after changing the `CSAT` scale, the number of customers who can be classified 
    as partially satisfied customers (group `3`) became equal to the number 
    of partially dissatisfied customers (group `2`).
    
    Then will find the confidence intervals of the statistics of the metrics of these groups. 
    This information will help to estimate the central values of the statistics and check 
    for a tendency for the metrics to grow in the "larger" direction.

    Will calculate the confidence intervals using the bootstrap method.
    The results will be displayed graphically (the confidence intervals will be marked using horizontal segments,
    and their midpoints will be marked using dots).

    ### Confidence intervals of statistics
    '''
)

fig = plot_metric_confidence_interval_9_3()
st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
s = display_confidence_interval(ci, metrics=st.session_state.research_metrics,
                                caption='', caption_font_size=12,
                                opacity=0.5, precision=1, index_width=35, col_width=105)
st.markdown(s.to_html(table_uuid="table_categories_dist_9_3"), unsafe_allow_html=True)

st.markdown(
    '''
    Based on the analysis of confidence intervals, the following **conclusions** can be made:
    1. All metrics show a tendency for the central values of statistics to increase 
    with the growth of the customer satisfaction index.
    2. In accordance with the indicated tendency, the "worst" values of the metrics are observed 
    in the first group `1`, and the best - in the last group `5`.

    It is visually noticeable that only in one pair of neighboring groups: `4` and `5`, 
    the confidence intervals of the statistics of all metrics do not intersect. But, to be sure of this,
    will display information on the presence of "overlaps" of confidence intervals of the statistics
    of the groups in pairs. Will highlight ":red[negative]" results in red
    because they are the most important and informative.

    ### Overlapping confidence intervals of the statistics
    '''
)

s = display_confidence_interval_overlapping(
    ci_overlapping, metrics=st.session_state.research_metrics,
    # caption='<b>Overlapping confidence intervals of the statistics</b>',
    opacity=0.5, index_width=35, col_width=185)
st.markdown(s.to_html(table_uuid="table_confidence_interval_overlapping_9_3"), unsafe_allow_html=True)

st.markdown(
    '''
    Based on the obtained results on the presence of "overlapping" confidence intervals of statistics,
    the following conclusions can be made:
    Indeed, the confidence intervals of the statistics of all metrics **do not overlap** only 
    for the pair of groups `3` and `4`. That is, only with respect to this pair 
    of groups can a conclusion be made about a significant statistical difference in the metrics under research 
    (the metrics of group `3` are smaller than those of group `4`). 
    With respect to the remaining pairs of groups, it is impossible to make a conclusion 
    about the significance of the difference in the metrics of these groups based on exploratory analysis - 
    statistical tests must be carried out.
    
    ## Statistical tests

    Based on exploratory analysis, it has been established that there is a clear tendency for the metric values 
    to increase with increasing customer satisfaction. That is, to answer the 1st question, 
    should be performed "one-sided" tests of comparison of statistics for adjacent groups and use the test. 
    The first group in the tested pair will be the group of customers with a lower level of satisfaction, 
    so will perform "left-sided" tests.    
    '''
)

group_pair_index = 0
alternatives, pvalues, mark_statistic, null_distributions, statistics = st.session_state.section_9_4_1

st.markdown(
    f'''
    ### Groups "{groups[0]}" and "{groups[1]}"

    For all metrics, will accept as the `null hypothesis` the assumption that the values
    of the `{groups[0]}` group statistics are **not less than** the values of the `{groups[1]}` group statistics.
    And as the `alternative hypothesis` the opposite statement that the values of the `{groups[0]}`
    group statistics are still **less than** the values of the `{groups[1]}` group statistics.
    In mathematical form, the test statistics and formulated hypotheses can be written as follows:
    '''
)
st.latex(f'\Delta \hat{{TM}}=\hat{{TM}}_{{{groups[0]}}}-\hat{{TM}}_{{{groups[1]}}}'.replace(' ', ' \, '))
st.latex(r'H_0:\Delta \hat{TM}≥0')
st.latex(r'H_1:\Delta \hat{TM}<0')
st.markdown(
    f'''
    Will perform testing and visualize the results by constructing histograms
    of the null distribution of the test statistics of the metrics. On the histograms,
    will mark the observed value of the test statistic with a vertical dashed line
    and color the areas of the distributions that are used to calculate the `p-values`
    (to the left of the lines of the observed values of the test statistic).
    '''
)

st.markdown('#### Density of the null probability distribution of the test statistic')
fig = plot_metric_histograms_9_4_1(null_distributions, statistics, st.session_state.research_metrics, mark_statistic)
st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
st.markdown(
    f'''
    Will present the obtained `p-values` in a tabular form, marking in :red[red]
    those values that are below the significance level.

    #### Test p-values
    ''', unsafe_allow_html=True
)
s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                    metrics=st.session_state.research_metrics, alpha=st.session_state.alpha,
                    caption='', col_width=305)
st.markdown(s.to_html(table_uuid="table_pvalues_9_4_1"), unsafe_allow_html=True)

st.markdown(
    '''
    #### Conclusions

    1. Since the `p-values` obtained as a result of the test for all the metrics under research are **less** 
    than the significance level, we **can reject the null hypothesis** in relation to them and, accordingly, 
    will assume that the values of all the metrics under research for group `1` are **less** than for group `2`.
    2. It should be noted that the **confidence level** for the result obtained is very **high** - about `99%`.
    '''
)

group_pair_index = 1
_, pvalues, mark_statistic, null_distributions, statistics = st.session_state.section_9_4_2

st.markdown(
    f'''
    ### Groups "{groups[1]}" and "{groups[2]}"

    For all metrics, will accept as the `null hypothesis` the assumption that the values
    of the `{groups[1]}` group statistics are **not less than** the values of the `{groups[2]}` group statistics.
    And as the `alternative hypothesis` the opposite statement that the values of the `{groups[1]}`
    group statistics are still **less than** the values of the `{groups[2]}` group statistics.
    In mathematical form, the test statistics and formulated hypotheses can be written as follows:
    '''
)
st.latex(f'\Delta \hat{{TM}}=\hat{{TM}}_{{{groups[1]}}}-\hat{{TM}}_{{{groups[2]}}}'.replace(' ', ' \, '))
st.latex(r'H_0:\Delta \hat{TM}≥0')
st.latex(r'H_1:\Delta \hat{TM}<0')
st.markdown(
    f'''
    Will perform testing and visualize the results by constructing histograms
    of the null distribution of the test statistics of the metrics. On the histograms,
    will mark the observed value of the test statistic with a vertical dashed line
    and color the areas of the distributions that are used to calculate the `p-values`
    (to the left of the lines of the observed values of the test statistic).
    '''
)

st.markdown('#### Density of the null probability distribution of the test statistic')
fig = plot_metric_histograms_9_4_1(null_distributions, statistics, st.session_state.research_metrics, mark_statistic)
st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
st.markdown(
    f'''
    Will present the obtained `p-values` in a tabular form, marking in :red[red]
    those values that are below the significance level.

    #### Test p-values
    ''', unsafe_allow_html=True
)
s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                    metrics=st.session_state.research_metrics, alpha=st.session_state.alpha,
                    caption='', col_width=305)
st.markdown(s.to_html(table_uuid="table_pvalues_9_4_2"), unsafe_allow_html=True)

st.markdown(
    '''
    #### Conclusions

    1. Since the `p-value` obtained as a result of the test for the `Video Streaming Download Throughput` 
    metric is **less than** the significance level, **the null hypothesis cannot be rejected** 
    with respect to this metric and, accordingly, will assume that the values of this metric in group `2` 
    are **less than** those of group `3`.
    2. It should be noted that the `p-value` for the `Downlink Throughput` and `Web Page Download Throughput` metrics 
    is significantly greater than the significance level, which indicates that the values of these metrics 
    in group `2` are close to the values of group `3`.    
    '''
)

group_pair_index = 2
_, pvalues, mark_statistic, null_distributions, statistics = st.session_state.section_9_4_3

st.markdown(
    f'''
    ### Groups "{groups[2]}" and "{groups[3]}"

    For all metrics, will accept as the `null hypothesis` the assumption that the values
    of the `{groups[2]}` group statistics are **not less than** the values of the `{groups[3]}` group statistics.
    And as the `alternative hypothesis` the opposite statement that the values of the `{groups[2]}`
    group statistics are still **less than** the values of the `{groups[3]}` group statistics.
    In mathematical form, the test statistics and formulated hypotheses can be written as follows:
    '''
)
st.latex(f'\Delta \hat{{TM}}=\hat{{TM}}_{{{groups[2]}}}-\hat{{TM}}_{{{groups[3]}}}'.replace(' ', ' \, '))
st.latex(r'H_0:\Delta \hat{TM}≥0')
st.latex(r'H_1:\Delta \hat{TM}<0')
st.markdown(
    f'''
    Will perform testing and visualize the results by constructing histograms
    of the null distribution of the test statistics of the metrics. On the histograms,
    will mark the observed value of the test statistic with a vertical dashed line
    and color the areas of the distributions that are used to calculate the `p-values`
    (to the left of the lines of the observed values of the test statistic).
    '''
)

st.markdown('#### Density of the null probability distribution of the test statistic')
fig = plot_metric_histograms_9_4_3(null_distributions, statistics, st.session_state.research_metrics, mark_statistic)
st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
st.markdown(
    f'''
    Will present the obtained `p-values` in a tabular form, marking in :red[red]
    those values that are below the significance level.

    #### Test p-values
    ''', unsafe_allow_html=True
)
s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                    metrics=st.session_state.research_metrics, alpha=st.session_state.alpha,
                    caption='', col_width=305)
st.markdown(s.to_html(table_uuid="table_pvalues_9_4_3"), unsafe_allow_html=True)

st.markdown(
    '''
    #### Conclusions
    1. Since the `p-values` obtained as a result of the test for all the metrics under research are **less** 
    than the significance level, **the null hypothesis can be rejected** in relation to them and, accordingly, 
    will assume that the values of all the metrics under research for group `3` are **less** than for group `4`.
    2. It should be noted that the **confidence level** for the result obtained is very **high** - about `99%`.
    '''
)

st.markdown(
    '''
    ## Conclusions
    To form the conclusions of the research, it is necessary to analyze the results of all the tests performed.
    To do this, will display the obtained `p-values` of all the tests in tabular form:

    ### Tests p-values
    '''
)

s = display_pvalues(pvalues,
                    metrics=st.session_state.research_metrics, alpha=st.session_state.alpha,
                    caption='', index_width=35, col_width=305, opacity=0.5)
st.markdown(s.to_html(table_uuid="table_pvalues_9_5"), unsafe_allow_html=True)

st.markdown(
    f'''
    Based on this information, the following answers can be given to the questions posed:
    1. Since all neighboring groups have at least one customer metric with a lower `CSAT` value **less** 
    than the group with a higher `CSAT` level, we can assume that customers of all groups 
    **belong to different populations**. 
    Thus, the division of customers by the `CSAT` value is performed **correctly**.
    2. Since only the `Video Streaming Download Throughput` metric has all `p-values` **less** 
    than the significance level, this metric has the **strongest influence** on `CSAT`.
    
    Additionally, let's look at the dependence characteristics of the `Video Streaming Download Throughput` 
    metric value on `CSAT`. To do this, will construct a scatter plot of the statistics values of this metric 
    and a trend line (dashed line) describing the linear dependence of the statistics values on the `CSAT` level.
    
    ### Trend of \"{st.session_state.research_metrics.loc['Video Streaming Download Throughput(Kbps)', 'name']}"
    '''
)

df = ci_center['Video Streaming Download Throughput(Kbps)'].rename('value').to_frame()
fig = px.scatter(
    df, x=df.index, y='value',
    title=' ',
    labels={'x': '', 'value': 'kbit/s', 'index': 'CSAT'}, trendline="ols")
fig.update_layout(title_x=0.5, title_y=0.95, title_font_size=14,
                  width=500, height=350,
                  margin_t=40, margin_b=0)
fig.update_traces(hovertemplate='%{x}<br>%{y}<extra></extra>', selector={'mode': 'markers'})
fig.update_traces(line_dash='dash', selector={'mode': 'lines'})
fig.update_xaxes(tickmode='array', tickvals=[1, 2, 3, 4], tickfont_size=14)
fig.update_yaxes(tickfont_size=14)
st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=False)
results = px.get_trendline_results(fig).iloc[0, 0]

st.markdown(
    f'''
    As can see, the trend of the metrics change is described by a straight line. 
    This is confirmed by the fact that the `determination coefficient` of the linear regression 
    is very **close to 1.0** (R²=`{results.rsquared:.4f}`). The `CSAT` value increases after the growth 
    of `Video Streaming Download Throughput(Kbps)` by about `{results.params[1]:.0f}` kbps.
    '''
)

