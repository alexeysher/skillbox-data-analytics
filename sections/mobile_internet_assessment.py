import streamlit as st
import pandas as pd
import plotly.express as px
from functions import (display_cat_info, plot_group_size_barchart, plot_metric_histograms,
                       plot_metric_confidence_interval, display_confidence_interval,
                       display_confidence_interval_overlapping, display_pvalues)

st.cache_resource(show_spinner='Displaying...')
def display_confidence_interval_7_3():
    return display_confidence_interval(ci, metrics=st.session_state.research_metrics,
                                       caption='', caption_font_size=12,
                                       opacity=0.5, precision=1, index_width=141, col_width=105)


@st.cache_resource(show_spinner='Plotting...')
def plot_metric_confidence_interval_7_3():
    return plot_metric_confidence_interval(ci, metrics=st.session_state.research_metrics,
                                           title='<b>Confidence intervals of metric statistics</b>',
                                           height=230, n_cols=3,
                                           horizontal_spacing=0.04, vertical_spacing=0.07)


@st.cache_resource(show_spinner='Plotting...')
def plot_group_size_barchart_7_3():
    return plot_group_size_barchart(
        research_data,
        title=' ', title_y=0.85,
        # title='<b>Number of customers in the considered groups</b>', title_y=0.85,
        labels_font_size=14,
        axes_tickfont_size=14,
        axes_title_font_size=14,
        width=600,
        height=320
    )


@st.cache_resource(show_spinner='Displaying...')
def display_confidence_interval_overlapping_7_3():
    return display_confidence_interval_overlapping(
        ci_overlapping, metrics=st.session_state.research_metrics,
        # caption='<b>Overlapping confidence intervals of the statistics</b>',
        opacity=0.5, index_width=160, col_width=185)


@st.cache_resource(show_spinner='Plotting...')
def ci_plot_7_3():
    return plot_metric_confidence_interval(ci, metrics=st.session_state.research_metrics,
                                           title='', height=300, n_cols=3,
                                           horizontal_spacing=0.04, vertical_spacing=0.07,
                                           labels_font_size=16, axes_tickfont_size=14, units_font_size=14)


@st.cache_resource(show_spinner='Plotting...')
def plot_metric_histograms_7_4_1(null_distributions, statistics, research_metrics, mark_statistic):
    return plot_metric_histograms(
        null_distributions, statistic=statistics, metrics=research_metrics,
        # title=f'<b>Density of the null probability distribution of the test statistic</b>', title_y=0.9,
        labels_font_size=16, axes_tickfont_size=14, units_font_size=14,
        height=300, n_cols=3, opacity=0.5,
        histnorm='probability density',
        add_kde=True, add_statistic=True, mark_statistic=mark_statistic,
        horizontal_spacing=0.08, vertical_spacing=0.07)


@st.cache_resource(show_spinner='Plotting...')
def plot_metric_histograms_7_4_2(null_distributions, statistics, research_metrics, mark_statistic):
    return plot_metric_histograms(
        null_distributions, statistic=statistics, metrics=research_metrics,
        # title=f'<b>Density of the null probability distribution of the test statistic</b>', title_y=0.9,
        labels_font_size=16, axes_tickfont_size=14, units_font_size=14,
        height=300, n_cols=3, opacity=0.5,
        histnorm='probability density',
        add_kde=True, add_statistic=True, mark_statistic=mark_statistic,
        horizontal_spacing=0.08, vertical_spacing=0.07)


@st.cache_resource(show_spinner='Plotting...')
def plot_metric_histograms_7_4_3(null_distributions, statistics, research_metrics, mark_statistic):
    return plot_metric_histograms(
        null_distributions, statistic=statistics, metrics=research_metrics,
        # title=f'<b>Density of the null probability distribution of the test statistic</b>', title_y=0.9,
        labels_font_size=16, axes_tickfont_size=14, units_font_size=14,
        height=300, n_cols=3, opacity=0.5,
        histnorm='probability density',
        add_kde=True, add_statistic=True, mark_statistic=mark_statistic,
        horizontal_spacing=0.08, vertical_spacing=0.07)


@st.cache_resource(show_spinner='Plotting...')
def plot_metric_histograms_7_4_4(null_distributions, statistics, research_metrics, mark_statistic):
    return plot_metric_histograms(
        null_distributions, statistic=statistics, metrics=research_metrics,
        # title=f'<b>Density of the null probability distribution of the test statistic</b>', title_y=0.9,
        labels_font_size=16, axes_tickfont_size=14, units_font_size=14,
        height=300, n_cols=3, opacity=0.5,
        histnorm='probability density',
        add_kde=True, add_statistic=True, mark_statistic=mark_statistic,
        horizontal_spacing=0.08, vertical_spacing=0.07)


st.markdown(
    f'''
    # Research of mobile internet service quality assessments

    ## Objective of the research

    Depending on the assessment of the mobile internet service, 
    the customers were divided into five categories. 
    To research the customers belonging to the above categories, 
    will allocate the customers into groups of the same name. 
    For the convenience of analyzing the research data, 
    will assign the following color scheme to each group, which will use when displaying graphic and tabular data:
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[0]};opacity:0.5">
      &nbsp;&nbsp;Very dissatisfied&nbsp;&nbsp;</span>
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[1]};opacity:0.5">
      &nbsp;&nbsp;Dissatisfied&nbsp;&nbsp;</span>
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[2]};opacity:0.5">
      &nbsp;&nbsp;Neutral&nbsp;&nbsp;</span>
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[3]};opacity:0.5">
      &nbsp;&nbsp;Satisfied&nbsp;&nbsp;</span>
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[4]};opacity:0.5">
      &nbsp;&nbsp;Very satisfied&nbsp;&nbsp;</span>

    These groups are illustrated on the customer distribution map.

    ### Customers distribution map
    ''', unsafe_allow_html=True
)

s = display_cat_info(st.session_state.data_clean).set_properties(
    pd.IndexSlice['Very dissatisfied', 'Internet and Video': 'Video'],
    color='white', background=px.colors.DEFAULT_PLOTLY_COLORS[0], opacity=0.5
).set_properties(
    pd.IndexSlice['Dissatisfied', 'Internet and Video': 'Video'],
    color='white', background=px.colors.DEFAULT_PLOTLY_COLORS[1], opacity=0.5
).set_properties(
    pd.IndexSlice['Neutral', 'Internet and Video': 'Video'],
    color='white', background=px.colors.DEFAULT_PLOTLY_COLORS[2], opacity=0.5
).set_properties(
    pd.IndexSlice['Satisfied', 'Internet and Video': 'Video'],
    color = 'white', background = px.colors.DEFAULT_PLOTLY_COLORS[3], opacity=0.5
).set_properties(
    pd.IndexSlice['Very satisfied', 'No'],
    color = 'white', background = px.colors.DEFAULT_PLOTLY_COLORS[4], opacity=0.5
)
st.markdown(s.to_html(table_uuid="table_categories_dist_7_1"), unsafe_allow_html=True)
st.markdown(
    '''
    Perhaps this grouping is not statistically correct. For example, 
    for customers who rate the quality of mobile Internet service as `Very dissatisfied`, 
    the researched Internet metrics are close to customers who rated the mobile Internet service as `Dissatisfied`.

    In this research, will try to understand whether this is true by answering the following questions:

    - Do customers in the groups belong to different populations?
    - If customers do not belong to the same population, what metrics do they differ in?
    '''
)

st.markdown(
    '''
    ## Exploratory analysis

    First, will look at the number of customers in groups utilizing a bar chart.
    '''
)

research_data, groups, group_pairs, ci, ci_overlapping, ci_center = st.session_state.section_7_3
fig = plot_group_size_barchart_7_3()
st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=False)

st.markdown(
    '''
    As can see, the largest group is `Very satisfied` (`1084` customers). 
    This is a group of customers who are completely satisfied with the mobile Internet service. 
    The other groups are several times smaller and the difference in numbers between them is not so significant. 
    The smallest group is `Neutral` (`118` customers), which includes customers who find the quality 
    of the mobile Internet service satisfactory.

    Then will find the confidence intervals of the statistics of the metrics in the groups.
    This information will help to estimate the central values of the statistics
    and make assumptions about the presence of significant differences between the metrics
    of the customer groups and their direction (up or down).

    Will calculate the confidence intervals using the bootstrap method.
    The results will be displayed graphically (the confidence intervals will be marked using horizontal segments,
    and their midpoints will be marked using dots).
    
    ### Confidence intervals of statistics
    '''
)

fig = ci_plot_7_3()
st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)

s = display_confidence_interval_7_3()
st.markdown(s.to_html(table_uuid="table_categories_dist_7_3"), unsafe_allow_html=True)

st.markdown(
    '''
    Based on the analysis of confidence intervals, the following **conclusions** can be made:
    1. Customers of the `Internet and Video` group have significantly "worse" values of the metrics
    than the `Internet` and `Video` groups.
    2. Customers of the `Internet` and `Video` groups have similar values of the `Streaming video download speed`
    and `Web page download speed via browser` metrics.
    3. Customers of the `Video` group have a higher central value of the `Average speed "to subscriber"` metric
    statistics than the `Web page download speed via browser` metric.
    This may indicate that customers of this group consume video content to a greater extent.
    4. The confidence intervals of the metrics of the `Video` group are significantly wider than those
    of the `Internet and Video` and `Internet` groups.
    But this most likely does not indicate a significantly greater spread of the values of these metrics
    in the population to which this group belongs than in the populations to which the other groups belong.
    It is quite possible that such an effect is due to the fact that this group is significantly smaller in size.
    And, accordingly, with repeated samples from this group, the probability of obtaining
    more extreme statistical values is higher than with samples from other larger groups.

    It is visually noticeable that the confidence intervals of the statistics of the metrics under research 
    intersect in all pairs of neighboring groups. But, to be sure of this, 
    will display information on the presence of "overlaps" of confidence intervals of the statistics 
    of the groups in pairs. Will highlight ":red[negative]" results in red 
    because they are the most important and informative.
    
    ### Overlapping confidence intervals of the statistics
    '''
)

s = display_confidence_interval_overlapping_7_3()
st.markdown(s.to_html(table_uuid="table_confidence_interval_overlapping_7_3"), unsafe_allow_html=True)

st.markdown(
    '''
    Based on the obtained results on the presence of "overlapping" confidence intervals of statistics,
    the following conclusion can be made:
    since the confidence intervals of the statistics of all the metrics under research 
    in neighboring groups **do not overlap**, this does not allow us to draw a conclusion 
    about the significance of the difference in the statistics of these groups based on exploratory analysis - 
    statistical tests must be carried out.

    ## Statistical tests

    Based on exploratory analysis, it has established that there is a clear tendency for the metric values 
    to increase with the growth of the ratings given by customers to the mobile Internet service. 
    That is, to answer the 1st question, "one-sided" tests of comparison of statistics 
    of only neighboring groups should be performed. The first group in the tested pair will be the group of customers 
    with the lower rating, therefore will perform "left-sided" tests.
    '''
)

group_pair_index = 0
alternatives, pvalues, mark_statistic, null_distributions, statistics = st.session_state.section_7_4_1

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
fig = plot_metric_histograms_7_4_1(null_distributions, statistics, st.session_state.research_metrics, mark_statistic)
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
st.markdown(s.to_html(table_uuid="table_pvalues_7_4_1"), unsafe_allow_html=True)

st.markdown(
    '''
    #### Conclusions

    1. Since the `p-values` obtained as a result of the test for all metrics are **greater** 
    than the significance level, **the null hypothesis cannot be rejected** with respect 
    to all the metrics under research and, accordingly, will assume that the values of all metrics 
    in the `Very dissatisfied` group are not less than those in the `Dissatisfied` group.
    2. The groups under research have especially **close** values for the `Video Streaming Download Throughput` metric, 
    since the `p-value` value for this metric is closest to `0.5`. 
    This also confirms the conclusions of the exploratory analysis.    
    '''
)

group_pair_index = 1
_, pvalues, mark_statistic, null_distributions, statistics = st.session_state.section_7_4_2

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
fig = plot_metric_histograms_7_4_1(null_distributions, statistics, st.session_state.research_metrics, mark_statistic)
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
st.markdown(s.to_html(table_uuid="table_pvalues_7_4_2"), unsafe_allow_html=True)

st.markdown(
    '''
    #### Conclusions
    
    1. Since the `p-value` obtained as a result of the test for the `Video Streaming Download Throughput` metric 
    is **less** than the significance level, **the null hypothesis can be rejected** in relation to it and, accordingly, 
    will assume that the values of this metric for the `Dissatisfied` group are **less** than for the `Neutral` group.
    2. The values of the remaining metrics for the researched groups are quite **close**, 
    since the `p-values` for these metrics are significantly **higher** than the significance level.
    '''
)

group_pair_index = 2
_, pvalues, mark_statistic, null_distributions, statistics = st.session_state.section_7_4_3

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
fig = plot_metric_histograms_7_4_3(null_distributions, statistics, st.session_state.research_metrics, mark_statistic)
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
st.markdown(s.to_html(table_uuid="table_pvalues_7_4_3"), unsafe_allow_html=True)

st.markdown(
    '''
    #### Conclusions
    1. Since the `p-values` obtained as a result of the test for all metrics are **greater** than 
    the significance level, **the null hypothesis cannot be rejected** with respect to all the metrics under research 
    and, accordingly, will assume that the values of all metrics in the `Neutral` group 
    are **not less** than those in the `Satisfied` group.
    2. It should be noted that the values of the `Downlink Throughput` 
    and `Video Streaming Download Throughput` metrics of the researched groups have, although not significant, 
    but **significant differences**, since the `p-values` of these metrics are **close** to the significance level.
    '''
)

group_pair_index = 3
_, pvalues, mark_statistic, null_distributions, statistics = st.session_state.section_7_4_4

st.markdown(
    f'''
    ### Groups "{groups[3]}" and "{groups[4]}"

    For all metrics, will accept as the `null hypothesis` the assumption that the values 
    of the `{groups[3]}` group statistics are **not less than** the values of the `{groups[4]}` group statistics. 
    And as the `alternative hypothesis` the opposite statement that the values of the `{groups[3]}` 
    group statistics are still **less than** the values of the `{groups[4]}` group statistics. 
    In mathematical form, the test statistics and formulated hypotheses can be written as follows:
    '''
)
st.latex(f'\Delta \hat{{TM}}=\hat{{TM}}_{{{groups[3]}}}-\hat{{TM}}_{{{groups[0]}}}'.replace(' ', ' \, '))
st.latex(r'H_0:\Delta \hat{TM}=0')
st.latex(r'H_1:\Delta \hat{TM}≠0')
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
fig = plot_metric_histograms_7_4_4(null_distributions, statistics, st.session_state.research_metrics, mark_statistic)
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
st.markdown(s.to_html(table_uuid="table_pvalues_7_4_4"), unsafe_allow_html=True)

st.markdown(
    '''
    #### Conclusions
    1. Since the `p-values` of the `Video Streaming Download Throughput` and `Web Page Download Throughput` metrics 
    obtained as a result of the test are **less** than the significance level, 
    **the null hypothesis can be rejected** with respect to these metrics and, accordingly, 
    will assume that the values of these metrics of the `Satisfied` group are **less** 
    than those of the `Very satisfied` group.
    2. The values of the `Downlink Throughput` metric of the `Satisfied` group, although not significantly, 
    are **significantly** `less`, since the `p-value` value of this metric is close to the significance level.    
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
                    caption='', index_width=160, col_width=305, opacity=0.5)
st.markdown(s.to_html(table_uuid="table_pvalues_7_5"), unsafe_allow_html=True)

st.markdown(
    f'''
    Based on this information, the following answers can be given to the questions posed:
    1. Since all the metrics of the customers of the `Very dissatisfied` group are **no less** than those
    of the neighboring `Dissatisfied` group, we can assume that the customers of these groups
    **belong to the same population**.
    2. Since all the metrics of the customers of the `Neutral` group are **no less**
    than those of the neighboring `Satisfied` group, we can assume that the customers of these groups
    **belong to the same population**.
    3. The `Video Streaming Download Throughput` metric has the **strongest** influence on the division
    of customers into populations depending on the assessment of the mobile Internet service,
    since differences were found in the value of this metric between two pairs of the researched groups.
    But the `Downlink Throughput` metric has the **weakest** influence on this,
    since no pair of neighboring groups have differences in the values of this metric.

    Since it was found that the groups `Very dissatisfied` and `Dissatisfied` can be attributed to the same population,
    then in the future Let's consider the customers of these groups as belonging to the same category
    of customers with the same assessment of the quality of mobile Internet service.
    Will call this category `Dissatisfied`.

    A similar situation is with the groups `Neutral` and `Satisfied`. Let's call the combined category `Satisfied`.


    Thus, we have the following categories of customers depending on the assessment
    of the quality of mobile Internet service:
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[0]};opacity:0.5">
      &nbsp;Dissatisfied&nbsp;</span>
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[1]};opacity:0.5">
      &nbsp;Satisfied&nbsp;</span>
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[2]};opacity:0.5">
      &nbsp;Very satisfied&nbsp;</span>.

    Will illustrate these categories on a customer distribution map.

    ### Customers distribution map
    ''', unsafe_allow_html=True
)

s = display_cat_info(st.session_state.data_clean).set_properties(
    pd.IndexSlice['Very dissatisfied': 'Dissatisfied', 'Internet and Video':'Video'],
    color='white', background=px.colors.DEFAULT_PLOTLY_COLORS[0], opacity=0.5
).set_properties(
    pd.IndexSlice['Neutral': 'Satisfied', 'Internet and Video':'Video'],
    color='white', background=px.colors.DEFAULT_PLOTLY_COLORS[1], opacity=0.5
).set_properties(
    pd.IndexSlice['Very satisfied', 'No'],
    color='white', background=px.colors.DEFAULT_PLOTLY_COLORS[2], opacity=0.5
)
st.markdown(s.to_html(table_uuid="table_categories_dist_7_5"), unsafe_allow_html=True)
