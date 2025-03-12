import streamlit as st
import pandas as pd
import plotly.express as px
from functions import (display_cat_info, plot_group_size_barchart, plot_metric_histograms,
                       plot_metric_confidence_interval, display_confidence_interval,
                       display_confidence_interval_overlapping, display_pvalues)


@st.cache_resource(show_spinner='Plotting...')
def plot_group_size_barchart_6_3():
    return plot_group_size_barchart(
        research_data,
        title=' ', title_y=0.85,
        labels_font_size=14,
        axes_tickfont_size=14,
        axes_title_font_size=14,
        width=600,
        height=200
    ).update_layout(margin_t=0, margin_b=0)


@st.cache_resource(show_spinner='Plotting...')
def plot_metric_confidence_interval_6_3():
    return plot_metric_confidence_interval(
        ci, metrics=st.session_state.research_metrics,
        title='', height=223, n_cols=3,
        horizontal_spacing=0.04, vertical_spacing=0.07,
        labels_font_size=16, axes_tickfont_size=14, units_font_size=14
    ).update_layout(margin_t=60, margin_b=60)


@st.cache_resource(show_spinner='Plotting...')
def plot_metric_histograms_6_4_1(null_distributions, statistics, research_metrics, mark_statistic):
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
def plot_metric_histograms_6_4_2(null_distributions, statistics, research_metrics, mark_statistic):
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
def plot_metric_histograms_6_4_3(null_distributions, statistics, research_metrics, mark_statistic):
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
    # Research of reasons for dissatisfaction with mobile Internet service
    
    ## Objective of the research
    
    Depending on the reasons for dissatisfaction with the mobile Internet service, 
    the customers were divided into three categories. To research customers belonging to the above categories, 
    will allocate the selected customers into groups of the same name. 
    For the convenience of analyzing the research data, 
    will assign the following color scheme to each group, which will use when displaying graphic and tabular data:
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[0]};opacity:0.5">
      &nbsp;Internet and Video&nbsp;</span> - dissatisfied with the speed of Mobile Internet and Video loading;
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[1]};opacity:0.5">
      &nbsp;Internet&nbsp;</span> - dissatisfied primarily with the speed of Mobile Internet;
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[2]};opacity:0.5">
      &nbsp;Video&nbsp;</span> - dissatisfied primarily with the speed of Video loading.
    
    These groups are illustrated on the customer distribution map.
    
    ### Customers distribution map
    ''', unsafe_allow_html=True
)

s = display_cat_info(st.session_state.data_clean).set_properties(
    pd.IndexSlice['Very dissatisfied':'Satisfied', 'Internet and Video'],
    color='white', background=px.colors.DEFAULT_PLOTLY_COLORS[0], opacity=0.5
).set_properties(
    pd.IndexSlice['Very dissatisfied':'Satisfied', 'Internet'],
    color='white', background=px.colors.DEFAULT_PLOTLY_COLORS[1], opacity=0.5
).set_properties(
    pd.IndexSlice['Very dissatisfied':'Satisfied', 'Video'],
    color='white', background=px.colors.DEFAULT_PLOTLY_COLORS[2], opacity=0.5
)
st.markdown(s.to_html(table_uuid="table_categories_dist_6_1"), unsafe_allow_html=True)
st.markdown(
    '''
    Perhaps this grouping is not statistically correct. For example, 
    customers who are dissatisfied with video loading may simply consume more video content, 
    which is why they only reported slow video loading in their responses, 
    although their page loading speed is also low.

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

research_data, groups, group_pairs, ci, ci_overlapping, ci_center = st.session_state.section_6_3
fig = plot_group_size_barchart_6_3()
st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=False)

st.markdown(
    '''
    As can see, the largest group `Internet` (`440` customers) is the group of customers dissatisfied primarily 
    with the speed of Mobile Internet. There are significantly fewer customers in the group `Internet and video` 
    who are dissatisfied with both the speed of Mobile Internet and the speed of Video loading (`185` customers). 
    And the group of customers dissatisfied primarily with the speed of video downloading `Video` 
    is very small (`37` customers) - several times smaller than the number of customers in other groups.

    Then will find the confidence intervals of the statistics of the metrics in the groups. 
    This information will help to estimate the central values of the statistics 
    and make assumptions about the presence of significant differences between the metrics 
    of the customer groups and their direction (up or down).

    Will calculate the confidence intervals using the bootstrap method. 
    The results will be displayed graphically (the confidence intervals will be marked using horizontal segments, 
    and their midpoints will be marked using dots).

    For clarity, will also present the obtained results in a tabular form 
    (the "worst" and "best" values of the confidence interval centers 
    will be highlighted in ":red[red]" and ":green[green]" colors, respectively).

    ### Confidence intervals of statistics
    '''
)

fig = plot_metric_confidence_interval_6_3()
st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
s = display_confidence_interval(ci, metrics=st.session_state.research_metrics,
                                caption='', caption_font_size=12,
                                opacity=0.5, precision=1, index_width=150, col_width=105)
st.markdown(s.to_html(table_uuid="table_categories_dist_6_3"), unsafe_allow_html=True)

st.markdown(
    '''
    Based on the analysis of confidence intervals, the following conclusions can be made:
    1. All metrics show a tendency for the central values of statistics to increase, 
    i.e. to change for the "better" with an increase in the level of satisfaction. 
    That is, we can state that the dynamics of metric values is consistent with the level of assessment 
    of the quality of mobile Internet service by customers. 
    Therefore, statistically significant differences may be absent primarily between neighboring groups.
    2. The metric `Video Streaming Download Throughput` stands out from the overall picture. 
    Customers of the `Very dissatisfied` group have a slightly higher average statistical value 
    than the more satisfied `Dissatisfied` group. But this difference does not seem to be significant.
    3. The confidence intervals of the `Very satisfied` group are significantly smaller than those of the other groups. 
    But this does not indicate a significantly smaller spread of metric values in the population 
    to which this group belongs than in the populations to which the other groups belong. 
    It is quite possible that such an effect is due to the fact that this group 
    is significantly larger in size, and therefore, with repeated samples from it, 
    the probability of obtaining more extreme statistical values is lower than in smaller groups.
    4. But the larger size of the confidence interval of the `Satisfied` group than that of groups comparable in size: 
    `Very dissatisfied`, `Dissatisfied` and `Neutral`, may indicate a greater dispersion of metrics in this group.
    
    For clarity, will also present the obtained results in a tabular form 
    (the "worst" and "best" values of the confidence interval centers 
    will be highlighted in ":red[red]" and ":green[green]" colors, respectively).

    ### Overlapping confidence intervals of the statistics
    '''
)

s = display_confidence_interval_overlapping(
    ci_overlapping, metrics=st.session_state.research_metrics,
    # caption='<b>Overlapping confidence intervals of the statistics</b>',
    opacity=0.5, index_width=150, col_width=185)
st.markdown(s.to_html(table_uuid="table_confidence_interval_overlapping_6_3"), unsafe_allow_html=True)

st.markdown(
    '''
    Based on the obtained results on the presence of "overlapping" confidence intervals of statistics, 
    the following conclusions can be made:
    1. Confidence intervals of the `Video Streaming Download Throughput` and `Web Page Download Throughput` metrics 
    of the `Internet and Video` and `Internet` groups **do not overlap**, 
    which indicates a **significant difference** in the statistics of these metrics of this pair of groups. 
    Thus, it is possible not to perform a test for this pair of groups, 
    but to immediately conclude that these groups belong to different populations.
    2. Confidence intervals of the `Downlink Throughput` metric statistics of the `Internet and Video` 
    and `Video` groups also are not overlapped. 
    This indicates a significant difference in the statistics of this metric of this pair of groups. 
    Thus, it is possible not to perform a test for this pair of groups, 
    but to immediately conclude that these groups belong to different populations.
    3. Confidence intervals of the statistics of all metrics of the `Internet` and `Video` groups "overlapped". 
    Therefore, a conclusion about the significance of the difference in the statistics 
    of the researched metrics of these groups based on exploratory analysis cannot be drawn - a statistical test 
    must be performed.    

    ## Statistical tests

    Based on exploratory analysis, it has been established that the `Internet and Video` group 
    has significant differences in metrics relative to the `Internet` and `Video` groups. 
    That is, without performing testing, it can already be conclude that the `Internet and Video` group belongs 
    to a separate population.
    
    To answer the 1st question, it is enough to perform a test only for the `Internet` and `Video` groups. 
    But it's still needed to find out which metrics influence the difference between the groups most strongly, 
    so it's needed to perform tests for all pairs of groups.
    
    In this case, it does not matter to us in which direction (larger or smaller) this difference is directed. 
    Therefore, it is possible to perform the so-called `two-sided` test.
    
    But the `p-value` for a two-sided `Permutation test` is twice the minimum `p-value` 
    for a left-sided or right-sided test. Therefore, to compare with the selected significance level, 
    it is necessary to first halve the obtained `p-values`.    
    '''
)

group_pair_index = 0
alternatives, pvalues, mark_statistic, null_distributions, statistics = st.session_state.section_6_4_1

st.markdown(
    f'''
    ### Groups "{groups[0]}" and "{groups[1]}"
    
    For all metrics, will accept the assumption that there are **no significant differences** 
    between the values of the test group statistics as the `null hypothesis`. 
    And the opposite statement that **there are differences** as the `alternative hypothesis`. 
    In mathematical form, the test statistics and formulated hypotheses can be written as follows:
    '''
)
st.latex(f'\Delta \hat{{TM}}=\hat{{TM}}_{{{groups[0]}}}-\hat{{TM}}_{{{groups[1]}}}'.replace(' ', ' \, '))
st.latex(r'H_0:\Delta \hat{TM}=0')
st.latex(r'H_1:\Delta \hat{TM}≠0')
st.markdown(
    f'''
    Will perform testing and visualize the results by constructing histograms 
    of the null distribution of the test statistics of the metrics. On the histograms, 
    will mark the observed value of the test statistic with a vertical dashed line 
    and color the areas of the distributions that are used to calculate the `p-values` 
    (to the right or left of the lines of the observed values of the test statistic).
    '''
)

st.markdown('#### Density of the null probability distribution of the test statistic')
fig = plot_metric_histograms_6_4_1(null_distributions, statistics, st.session_state.research_metrics, mark_statistic)
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
st.markdown(s.to_html(table_uuid="table_pvalues_6_4_1"), unsafe_allow_html=True)

st.markdown(
    '''
    #### Conclusions
    
    1. Since the `p-values` obtained as a result of the test for all metrics are **less than** the significance level, 
    **the null hypothesis can be rejected** and, accordingly,
    will assume that there are **significant differences** between all metrics of these groups.
    2. Since the areas of the null distribution of the test statistics for all metrics **to the left** 
    of the observed test statistics are **less** than **to the right**, 
    this indicates that the values of all metrics of the `Internet and Video` group are **less** 
    than those of the `Internet` group, which confirms the preliminary conclusions of the exploratory analysis.
    '''
)

group_pair_index = 1
_, pvalues, mark_statistic, null_distributions, statistics = st.session_state.section_6_4_2

st.markdown(
    f'''
    ### Groups "{groups[1]}" and "{groups[2]}"

    For all metrics, will accept the assumption that there are **no significant differences** 
    between the values of the test group statistics as the `null hypothesis`. 
    And the opposite statement that **there are differences** as the `alternative hypothesis`. 
    In mathematical form, the test statistics and formulated hypotheses can be written as follows:
    '''
)
st.latex(f'\Delta \hat{{TM}}=\hat{{TM}}_{{{groups[1]}}}-\hat{{TM}}_{{{groups[2]}}}'.replace(' ', ' \, '))
st.latex(r'H_0:\Delta \hat{TM}=0')
st.latex(r'H_1:\Delta \hat{TM}≠0')
st.markdown(
    f'''
    Will perform testing and visualize the results by constructing histograms 
    of the null distribution of the test statistics of the metrics. On the histograms, 
    will mark the observed value of the test statistic with a vertical dashed line 
    and color the areas of the distributions that are used to calculate the `p-values` 
    (to the right or left of the lines of the observed values of the test statistic).
    '''
)

st.markdown('#### Density of the null probability distribution of the test statistic')
fig = plot_metric_histograms_6_4_1(null_distributions, statistics, st.session_state.research_metrics, mark_statistic)
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
st.markdown(s.to_html(table_uuid="table_pvalues_6_4_2"), unsafe_allow_html=True)

st.markdown(
    '''
    #### Conclusions
    
    1. Since the `p-values` obtained as a result of the test for all metrics are 
    **greater** than the significance level, **the null hypothesis cannot be rejected** and, accordingly, 
    will assume that there are **no significant differences** between all metrics of these groups.
    2. Since the areas of the zero distribution of the test statistics for the `Downlink Throughput` 
    and `Web Page Download Throughput` metrics **to the left** of the observed test statistics 
    are **smaller** than **to the right**, this suggests that the values of these metrics 
    for the `Internet` group are **smaller** than for the `Video` group, 
    which confirms the preliminary conclusions of the exploratory analysis.
    3. Since the null distribution region of the test statistic for the `Video Streaming Download Throughput` metric 
    is **larger** to the left** of the observed test statistic than to the **right**, 
    this suggests that the values of this metric for the `Internet` group 
    are **larger** than for the `Video` group. 
    However, this difference is not significant, since the `p-value` is close to `0.5`. 
    This also confirms the conclusions of the exploratory analysis.
    '''
)

group_pair_index = 2
_, pvalues, mark_statistic, null_distributions, statistics = st.session_state.section_6_4_3

st.markdown(
    f'''
    ### Groups "{groups[2]}" and "{groups[0]}"

    For all metrics, will accept the assumption that there are **no significant differences** 
    between the values of the test group statistics as the `null hypothesis`. 
    And the opposite statement that **there are differences** as the `alternative hypothesis`. 
    In mathematical form, the test statistics and formulated hypotheses can be written as follows:
    '''
)
st.latex(f'\Delta \hat{{TM}}=\hat{{TM}}_{{{groups[2]}}}-\hat{{TM}}_{{{groups[0]}}}'.replace(' ', ' \, '))
st.latex(r'H_0:\Delta \hat{TM}=0')
st.latex(r'H_1:\Delta \hat{TM}≠0')
st.markdown(
    f'''
    Will perform testing and visualize the results by constructing histograms 
    of the null distribution of the test statistics of the metrics. On the histograms, 
    will mark the observed value of the test statistic with a vertical dashed line 
    and color the areas of the distributions that are used to calculate the `p-values` 
    (to the right or left of the lines of the observed values of the test statistic).
    '''
)

st.markdown('#### Density of the null probability distribution of the test statistic')
fig = plot_metric_histograms_6_4_3(null_distributions, statistics, st.session_state.research_metrics, mark_statistic)
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
st.markdown(s.to_html(table_uuid="table_pvalues_6_4_3"), unsafe_allow_html=True)

st.markdown(
    '''
    #### Conclusions
    1. Since the `p-values` obtained as a result of the test for all metrics are 
    **greater** than the significance level, **the null hypothesis cannot be rejected** and, accordingly, 
    will assume that there are **no significant differences** between all metrics of these groups.
    2. Since the areas of the zero distribution of the test statistics for the `Downlink Throughput` 
    and `Web Page Download Throughput` metrics **to the left** of the observed test statistics 
    are **smaller** than **to the right**, this suggests that the values of these metrics 
    for the `Internet` group are **smaller** than for the `Video` group, 
    which confirms the preliminary conclusions of the exploratory analysis.
    3. Since the null distribution region of the test statistic for the `Video Streaming Download Throughput` metric 
    is **larger** to the left** of the observed test statistic than to the **right**, 
    this suggests that the values of this metric for the `Internet` group 
    are **larger** than for the `Video` group. 
    However, this difference is not significant, since the `p-value` is close to `0.5`. 
    This also confirms the conclusions of the exploratory analysis.
    '''
)

st.markdown(
    '''
    ## Conclusion
    To form the conclusions of the research, it is necessary to analyze the results of all the tests performed. 
    To do this, will display the obtained `p-values` of all the tests in tabular form:
    
    ### Tests p-values  
    '''
)

s = display_pvalues(pvalues,
                    metrics=st.session_state.research_metrics, alpha=st.session_state.alpha,
                    caption='', index_width=150, col_width=305, opacity=0.5)
st.markdown(s.to_html(table_uuid="table_pvalues_6_5"), unsafe_allow_html=True)

st.markdown(
    f'''
    Based on this information, the following answers can be given to the questions posed:
    1. customers who expressed dissatisfaction primarily with the Internet speed (the `Internet` group) 
    or primarily with video downloading (the `Video` group) do not have statistically significant differences 
    in the metrics under research and belong to the same population. 
    A noticeable difference is observed only for the `Downlink Throughput` metric, 
    since the `p-value` of this metric is only slightly higher than the significance level.
    2. The researched metrics of the `Internet and Video` group have statistically significant differences 
    with the other groups. That is, this group belongs to a separate population. 
    The strongest differences were found in the `Downlink Throughput` metric.
    
    Since it was established that customers of the `Internet` and `Video` groups 
    can be attributed to the same population, then in the future will consider customers 
    of these groups as belonging to the same category of customers of reasons for dissatisfaction 
    with the mobile Internet service. Will call this category `Internet or video`. 
    Thus, two categories of customers are left based on their dissatisfaction with the mobile Internet service:
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[0]};opacity:0.5">
      &nbsp;Internet and Video&nbsp;</span> - Dissatisfied with the speed of mobile Internet and video loading
    - <span style="color:white;background-color:{px.colors.DEFAULT_PLOTLY_COLORS[1]};opacity:0.5">
      &nbsp;Internet or Video&nbsp;</span> - Dissatisfied with the speed of mobile Internet or video loading
    
    Will illustrate these categories on a customer distribution map.
    
    ### Customers distribution map
    ''', unsafe_allow_html=True
)

s = display_cat_info(st.session_state.data_clean).set_properties(
    pd.IndexSlice['Very dissatisfied':'Satisfied', 'Internet and Video'],
    color='white',
    background=px.colors.DEFAULT_PLOTLY_COLORS[0],
    opacity=0.5).set_properties(
    pd.IndexSlice['Very dissatisfied':'Satisfied', 'Internet': 'Video'], color='white',
    background=px.colors.DEFAULT_PLOTLY_COLORS[1], opacity=0.5).set_properties(
    opacity=0.5)
st.markdown(s.to_html(table_uuid="table_categories_dist_6_5"), unsafe_allow_html=True)
