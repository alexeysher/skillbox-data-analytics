import streamlit as st
import pandas as pd
import plotly.express as px
from functions import (display_cat_info, plot_group_size_barchart, plot_metric_histograms,
                       plot_metric_confidence_interval, display_confidence_interval,
                       display_confidence_interval_overlapping, display_pvalues)


@st.cache_resource(show_spinner='Plotting...')
def plot_metric_confidence_interval_8_3():
    return plot_metric_confidence_interval(ci, metrics=st.session_state.research_metrics,
                                           title='<b>Confidence intervals of metric statistics</b>',
                                           height=230, n_cols=3,
                                           horizontal_spacing=0.04, vertical_spacing=0.07)


@st.cache_resource(show_spinner='Plotting...')
def plot_group_size_barchart_8_3():
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
def display_confidence_interval_overlapping_8_3():
    return display_confidence_interval_overlapping(
        ci_overlapping, metrics=st.session_state.research_metrics,
        # caption='<b>Overlapping confidence intervals of the statistics</b>',
        opacity=0.5, index_width=160, col_width=185)


@st.cache_resource(show_spinner='Plotting...')
def ci_plot_8_3():
    return plot_metric_confidence_interval(ci, metrics=st.session_state.research_metrics,
                                           title='', height=300, n_cols=3,
                                           horizontal_spacing=0.04, vertical_spacing=0.07,
                                           labels_font_size=16, axes_tickfont_size=14, units_font_size=14)


@st.cache_resource(show_spinner='Plotting...')
def plot_metric_histograms_8_4_1(null_distributions, statistics, research_metrics, mark_statistic):
    return plot_metric_histograms(
        null_distributions, statistic=statistics, metrics=research_metrics,
        # title=f'<b>Density of the null probability distribution of the test statistic</b>', title_y=0.9,
        labels_font_size=16, axes_tickfont_size=14, units_font_size=14,
        height=300, n_cols=3, opacity=0.5,
        histnorm='probability density',
        add_kde=True, add_statistic=True, mark_statistic=mark_statistic,
        horizontal_spacing=0.08, vertical_spacing=0.07)


@st.cache_resource(show_spinner='Plotting...')
def plot_metric_histograms_8_4_2(null_distributions, statistics, research_metrics, mark_statistic):
    return plot_metric_histograms(
        null_distributions, statistic=statistics, metrics=research_metrics,
        # title=f'<b>Density of the null probability distribution of the test statistic</b>', title_y=0.9,
        labels_font_size=16, axes_tickfont_size=14, units_font_size=14,
        height=300, n_cols=3, opacity=0.5,
        histnorm='probability density',
        add_kde=True, add_statistic=True, mark_statistic=mark_statistic,
        horizontal_spacing=0.08, vertical_spacing=0.07)


@st.cache_resource(show_spinner='Plotting...')
def plot_metric_histograms_8_4_3(null_distributions, statistics, research_metrics, mark_statistic):
    return plot_metric_histograms(
        null_distributions, statistic=statistics, metrics=research_metrics,
        # title=f'<b>Density of the null probability distribution of the test statistic</b>', title_y=0.9,
        labels_font_size=16, axes_tickfont_size=14, units_font_size=14,
        height=300, n_cols=3, opacity=0.5,
        histnorm='probability density',
        add_kde=True, add_statistic=True, mark_statistic=mark_statistic,
        horizontal_spacing=0.08, vertical_spacing=0.07)


@st.cache_resource(show_spinner='Plotting...')
def plot_metric_histograms_8_4_4(null_distributions, statistics, research_metrics, mark_statistic):
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
    # Research of satisfaction levels with mobile internet service

    ## Objective of the research

    The performed research of dividing customers into categories of assessments and reasons 
    for dissatisfaction with the mobile internet service obtained as a result of the survey. 
    Based on the results of these studies, all customers can be divided into five categories 
    depending on the degree of satisfaction with the quality of the mobile internet service. 
    The degree of customer satisfaction is usually called `CSAT` from the English `Customer Satisfaction Score`. 
    This indicator is usually indicated as a serial number (starting with 1) 
    of the degree of satisfaction (in the ascending direction). 
    Will list the categories of customers in ascending order of `CSAT`:
    
    1. Customers who negatively assessed the quality of the mobile internet service (category `dissatisfactory`) 
    and are dissatisfied with both the speed of the mobile internet and the speed 
    of downloading video (category `Internet and Video`), can be distinguish in the category of customers 
    who are `Completely dissatisfied` with the quality of the mobile internet service. 
    2. Customers who rated the quality of mobile internet service negatively (category `dissatisfactory`), 
    but indicated only slow mobile internet or only slow video loading as the root cause (category `Internet or video`), 
    can be classified into the category of customers who are `Partially dissatisfied` 
    with the quality of mobile internet service.
    3. Customers who rated the quality of mobile internet service positively overall (category `Satisfactory`), 
    but who still have complaints about the speed of mobile internet and the speed of video loading 
    (category `Internet and Video`), can be classified into the category of customers who are 
    `Neither satisfied nor disappointed` with the quality of mobile internet service.
    4. Customers who generally rated the quality of mobile Internet service positively (category `Satisfactory`), 
    but who still have complaints about the speed of mobile Internet or the speed 
    of downloading video (category `Internet or video`), 
    can be classified into the category of customers who are `Partially satisfied` 
    with the quality of mobile Internet service.
    5. The remaining customers who rated the service without having complaints about the mobile Internet service 
    (category `Very satisfied`), will be classified into the category of `Completely satisfied` customers.
    
    Perhaps such a division of customers by satisfaction levels is not statistically correct, 
    i.e. some groups of customers with different `CSAT` do not have statistically significant differences in metrics.
    
    In this research, will try to understand whether this is true by answering the **following questions**:
    
    1. Do customers with different CSAT belong to different populations?
    2. If customers with different CSAT do not belong to the same population, 
    then by which metrics do they differ especially strongly?
    
    For this purpose, in this research will divide customers into groups by the `CSAT` value. 
    Will assign a designation corresponding to the `CSAT` value and color scheme to these groups when displaying graphic and tabular data:
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[0]};opacity:0.5">
      &nbsp;&nbsp;1&nbsp;&nbsp;</span> - Completely dissatisfied
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[1]};opacity:0.5">
      &nbsp;&nbsp;2&nbsp;&nbsp;</span> - Partly dissatisfied
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[2]};opacity:0.5">
      &nbsp;&nbsp;3&nbsp;&nbsp;</span> - Neutral
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[3]};opacity:0.5">
      &nbsp;&nbsp;4&nbsp;&nbsp;</span> - Partly satisfied
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[4]};opacity:0.5">
      &nbsp;&nbsp;5&nbsp;&nbsp;</span> - Completely satisfied

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
    color='white', background=px.colors.DEFAULT_PLOTLY_COLORS[3], opacity=0.5
).set_properties(
    pd.IndexSlice['Very satisfied', 'No'],
    color='white', background=px.colors.DEFAULT_PLOTLY_COLORS[4], opacity=0.5
)
st.markdown(s.to_html(table_uuid="table_categories_dist_8_1"), unsafe_allow_html=True)

st.markdown(
    '''
    ## Exploratory analysis

    First, will look at the number of customers in groups utilizing a bar chart.
    '''
)

research_data, groups, group_pairs, ci, ci_overlapping, ci_center = st.session_state.section_8_3
fig = plot_group_size_barchart_8_3()
st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=False)

st.markdown(
    '''
    As expected, the largest group of companies with a satisfaction index of "5" (Customer "1084") 
    is the group whose customers are completely satisfied with the local Internet service. 
    The other groups are several times smaller. In the opposite direction, 
    a group of neutral customers with an index of `3` is displayed - there are only `54` customers in it.
    
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

fig = ci_plot_8_3()
st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
s = display_confidence_interval(ci, metrics=st.session_state.research_metrics,
                                caption='', caption_font_size=12,
                                opacity=0.5, precision=1, index_width=55, col_width=105)
st.markdown(s.to_html(table_uuid="table_categories_dist_8_3"), unsafe_allow_html=True)

st.markdown(
    '''
    Based on the analysis of confidence intervals, the following **conclusions** can be made:
    1. All metrics show a tendency for the central values of statistics to increase with the growth 
    of the customer satisfaction index. Therefore, statistically significant differences 
    may be absent primarily between neighboring groups.
    2. Group `3` stands out from the general picture. For this group, the metrics `Downlink Throughput` 
    and `Web Page Download Throughput` are **worse** than for group `2`. 
    And this difference looks **significant**, especially for the metric `Downlink Throughput`. 
    The value of this metric for group `3` has almost the same value as for group `1`.
    3. In accordance with the indicated tendency, the worst metric values are observed for the first group `1`, 
    and the best ones are for group `5`.
    4. The confidence intervals of group `5` are significantly **smaller** than for the other groups. 
    But this most likely does not indicate a significantly smaller spread of metric values 
    in the population to which this group belongs than in the populations to which the other groups belong. 
    It is quite possible that such an effect is due to the fact that this group is significantly larger in size, 
    and therefore, with repeated samples from it, there is a **lower probability** 
    of obtaining more extreme statistical values than in larger groups.
    5. But the **larger** size of the confidence intervals of the metrics in group `4` than 
    in groups comparable in size: `1`, `2` and `3`, may indicate a **larger** range of metrics in this group.

    It is visually noticeable that the confidence intervals of the statistics of the metrics under research
    intersect in all pairs of neighboring groups. But, to be sure of this,
    will display information on the presence of "overlaps" of confidence intervals of the statistics
    of the groups in pairs. Will highlight ":red[negative]" results in red
    because they are the most important and informative.

    ### Overlapping confidence intervals of the statistics
    '''
)

s = display_confidence_interval_overlapping_8_3()
st.markdown(s.to_html(table_uuid="table_confidence_interval_overlapping_8_3"), unsafe_allow_html=True)

st.markdown(
    '''
    Based on the obtained results on the presence of "overlapping" confidence intervals of statistics,
    the following conclusions can be made:
    Indeed, the confidence intervals of the statistics of all metrics **do not overlap** only for the pair 
    of groups `4` and `5`. 
    That is, only with respect to this pair of groups can a conclusion be made 
    about a significant statistical difference in the metrics under research 
    (the metrics of group `4` are smaller than those of group `5`). 
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
alternatives, pvalues, mark_statistic, null_distributions, statistics = st.session_state.section_8_4_1

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
fig = plot_metric_histograms_8_4_1(null_distributions, statistics, st.session_state.research_metrics, mark_statistic)
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
st.markdown(s.to_html(table_uuid="table_pvalues_8_4_1"), unsafe_allow_html=True)

st.markdown(
    '''
    #### Conclusions

    1. Since the `p-values` obtained as a result of the test for all the metrics under research are **less** 
    than the significance level, **the null hypothesis can reject** in relation to them and, accordingly, 
    will assume that the values of all the metrics under research for group `1` are **less** than for group `2`.
    2. It should be noted that the **confidence level** for the result obtained is very **high** - about `99%`.
    '''
)

group_pair_index = 1
_, pvalues, mark_statistic, null_distributions, statistics = st.session_state.section_8_4_2

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
fig = plot_metric_histograms_8_4_1(null_distributions, statistics, st.session_state.research_metrics, mark_statistic)
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
st.markdown(s.to_html(table_uuid="table_pvalues_8_4_2"), unsafe_allow_html=True)

st.markdown(
    '''
    #### Conclusions

    1. Since the `p-values` obtained as a result of the test for all the metrics under research are **greater** 
    than the significance level, **the null hypothesis cannot be rejected** in their regard and, accordingly, 
    will assume that the values of all the metrics under research for group `2` 
    are **not less** than those of group `3`.
    2. It should be noted that the `p-value` value for the `Downlink Throughput` 
    and `Web Page Download Throughput` metrics is significantly greater than `0.5`, 
    which indicates that the values of these metrics for group `2` 
    are significantly **"better"** than those of group `3`.
    '''
)

group_pair_index = 2
_, pvalues, mark_statistic, null_distributions, statistics = st.session_state.section_8_4_3

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
fig = plot_metric_histograms_8_4_3(null_distributions, statistics, st.session_state.research_metrics, mark_statistic)
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
st.markdown(s.to_html(table_uuid="table_pvalues_8_4_3"), unsafe_allow_html=True)

st.markdown(
    '''
    #### Conclusions
    1. Since the `p-values` of the `Downlink Throughput` metric obtained as a result of the test are **less than** 
    the significance level, **the null hypothesis can be rejected** with respect to this metric and, 
    accordingly, will assume that the values of this metric of group `3` are **less than** those of group `4`.
    2. It should be noted that the `p-value` value of the `Web Page Download Throughput` metric 
    is **slightly higher** than the significance level, which indicates **significant**, 
    although statistically **not significant**, differences in the values of these metrics between the test groups.
    '''
)

group_pair_index = 3
_, pvalues, mark_statistic, null_distributions, statistics = st.session_state.section_8_4_4

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
fig = plot_metric_histograms_8_4_4(null_distributions, statistics, st.session_state.research_metrics, mark_statistic)
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
st.markdown(s.to_html(table_uuid="table_pvalues_8_4_4"), unsafe_allow_html=True)

st.markdown(
    '''
    #### Conclusions
    1. Since the `p-values` obtained as a result of the test for all the metrics under research are **less** 
    than the significance level, we **the null hypothesis can be rejected** in relation to them and, accordingly, 
    will assume that the values of all the metrics under research for group `4` are **less** than for group `5`.
    2. It should be noted that the **confidence level** for the result obtained is very **high** - more than `99%`.
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
st.markdown(s.to_html(table_uuid="table_pvalues_8_5"), unsafe_allow_html=True)

st.markdown(
    f'''
    Based on this information, the following answers can be given to the questions posed:
    1. Since all the metrics of customers in group `1` are **less** than those of the neighboring group `2`, 
    it can be assumed that the customers of these groups **belong to different populations**. 
    On the same basis, customers of neighboring groups `4` and `5` should be assigned to different populations.
    2. Since the `Downlink Throughput` metric of customers in group `3` is **less** 
    than those of the neighboring group `4`, it can be assumed that the customers of these groups 
    **belong to different populations**.
    3. Since all the metrics of customers in group `2` are **not less** than those 
    of the neighboring group `3`, it can assumed that the customers of these groups 
    **belong to the same population**. 
    4. The metric `Video Streaming Download Throughput` has the **strongest** impact on dividing customers 
    into populations depending on their mobile internet service assessment, 
    since the average `p-value` of this metric (`0.1472`) is significantly lower 
    than that of the other metrics: `Downlink Throughput` (`0.2066`) and `Web Page Download Throughput` (`0.2088`).

    Since it was established that customers of groups `2` and `3` should be classified as belonging 
    to the same population, will further consider customers of these groups as belonging 
    to the same category of customers with the same level of satisfaction with the quality of mobile internet service. 

    Thus, the CSAT scale is narrowed to 4, i.e. the following customer categories remain 
    depending on the CSAT of the customers:
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[0]};opacity:0.5">
      &nbsp;&nbsp;1&nbsp;&nbsp;</span> - Completely dissatisfied;
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[1]};opacity:0.5">
      &nbsp;&nbsp;2&nbsp;&nbsp;</span> - Partially dissatisfied;
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[2]};opacity:0.5">
      &nbsp;&nbsp;3&nbsp;&nbsp;</span> - Partially satisfied;
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[3]};opacity:0.5">
      &nbsp;&nbsp;4&nbsp;&nbsp;</span> - Completely satisfied.

    Will illustrate these categories on a customer distribution map.

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
    color='white', background=px.colors.DEFAULT_PLOTLY_COLORS[1], opacity=0.5
).set_properties(
    pd.IndexSlice['Neutral': 'Satisfied', 'Internet': 'Video'],
    color='white', background=px.colors.DEFAULT_PLOTLY_COLORS[2], opacity=0.5
).set_properties(
    pd.IndexSlice['Very satisfied', 'No'], color='white',
    background=px.colors.DEFAULT_PLOTLY_COLORS[3], opacity=0.5
)
st.markdown(s.to_html(table_uuid="table_categories_dist_8_5"), unsafe_allow_html=True)
