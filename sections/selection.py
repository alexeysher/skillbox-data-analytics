import streamlit as st
import pandas as pd
from functions import plot_metric_histograms


@st.cache_resource(show_spinner='Displaying...')
def show_important_metric_table(metrics: pd.DataFrame):
    df = metrics.loc[
        ['Downlink Throughput(Kbps)', 'Video Streaming Download Throughput(Kbps)',
         'Web Page Download Throughput(Kbps)'], ['name']
    ]
    return df.style


@st.cache_resource(show_spinner='Plotting...')
def plot_metric_histograms_5_1():
    return plot_metric_histograms(st.session_state.statistic_distributions,
                                  statistic=st.session_state.statistic_distributions.median(),
                                  metrics=st.session_state.research_metrics,
                                  title='<b>Плотность распределения вероятностей статистики</b>', title_y=0.9,
                                  labels_font_size=16,
                                  units_font_size=16,
                                  axes_tickfont_size=14,
                                  height=300, n_cols=3, opacity=0.5,
                                  histnorm='probability density',
                                  add_kde=True, add_statistic=True, mark_confidence_interval=True,
                                  horizontal_spacing=0.06, vertical_spacing=0.07)


st.markdown(
    '''
    # Selection of metrics, statistics and criteria
    
    ## Selecting key metrics
    
    There is information on 8 metrics of mobile Internet service. 
    All of these metrics, except `Data transfer traffic volume`, 
    affect the convenience of using this service to one degree or another. 
    It\'s needed to select from these metrics those that have the greatest impact 
    on the reasons for dissatisfaction indicated by users:
    - Slow mobile Internet
    - Slow video loading.
    
    Both of these reasons are related to the assessment of data transfer speed. 
    Will turn to the opinion of experts and see which metrics are used specifically for Internet speed.
    One of such experts is the company [Ookla](https://www.ookla.com/), 
    which is engaged in assessing the quality of services of mobile and fixed Internet operators around the world. 
    This company publishes its methods for forming some assessments. Will use this information.
    
    The company\'s experts evaluate the speed of the Internet provided by the operator based on two metrics: 
    `Average speed "to the subscriber"` and `Average speed "from the subscriber"`. 
    Moreover, the ratio of the influence of these metrics is estimated as `9:1`: 
    *"...Speed Score which incorporates a measure of each provider’s download and upload speed 
    to rank network speed performance (90% of the final Speed Score is attributed to download speed 
    and the remaining 10% to upload speed)"*.
    
    Therefore, let\'s consider the metrics characterize data transfer speed:
    - `Downlink Throughput`
    - `Video Streaming Download Throughput`
    - `Web Page Download Throughput`.
    
    ## Statistic selection
    
    As was found out earlier, using the "average" to estimate the central position of the metric distribution 
    is not very correct due to the asymmetry of the metric distributions 
    and a large number of values with suspected anomalies. 
    Accordingly, it is necessary to select statistics that will be more applicable in these conditions. 
    There are the following standard versions of such statistics:
    - median;
    - trimmed mean;
    - trimmer (weighted average of median, 10th and fourth quartiles, Trimean (TM)).
    
    In practice, various modifications of these metrics are also used. 
    For example, the company Ookla uses a modified version of the trimmer in the methodology 
    for assessing the speed of Internet connections of providers. 
    The company's experts estimate the central value of speed as a weighted average 
    of the 10th, 50th (median) and 90th percentiles in the ratio 1:8:1, which is described by the following formula:
    '''
)
st.latex(r'\hat{TM}={P_{10}+8\cdot P_{50}+P_{90} \over {10}}')
st.markdown(
    '''
    Will use the developments of experts in this field and will estimate 
    the central position of the metrics under research in the same way.

    To see how the one works in this case will build the distribution of this statistic for the metrics under research.
    To build the distribution, will use the *bootstrap* method. 
    Will visualize the distributions using histograms, additionally marking the 95% confidence intervals, 
    coloring the distribution areas outside the confidence intervals, 
    and the median values using dashed lines.
    
    ### Probability density function of statistics    
    '''
)

fig = plot_metric_histograms_5_1()
st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
st.markdown(
    '''
    As can see, the distributions of statistic of all metrics are almost symmetrical, 
    therefore the median values of statistic match with the modes and 
    are approximately in the middle of the confidence intervals. 
    This circumstance allows, knowing the confidence interval, 
    to *estimate the central position of statistics* as its middle.
    
    ## Selecting a statistical criteria for testing hypotheses
    
    Will select a statistical test based on the following points:
    1. The metrics being studied are quantitative, so the criteria must be suitable for comparing quantitative data.
    2. As was established earlier, the distribution of the metrics being studied is far from "Neutral", 
    so parametric criteria are not very suitable - a non-parametric criteria must be used .
    3. Since the difference between groups that include different customers will be investigating 
    it's possible to assume that the groups are "independent". 
    Therefore, a criteria that is suitable for independent samples must be selected.
    4. In addition, the criteria must allow to compare groups using the statistic have been chosen - 
    a [modified trimmer](#statistic-selection).
    
    Based on the above conditions, tests based on the use of repeated samples can help to verify statistical hypotheses:
    - [Bootstrap test](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))
    - [Permutation test](https://en.wikipedia.org/wiki/Permutation_test).
    
    According to many experts, the Permutation Test is more suitable for testing hypotheses 
    about the belonging of groups to the same population. 
    But the Bootstrap Test is better used to test a certain difference between group statistics. 
    Since the task is to identify statistically distinguishable classes among customers, 
    *the Permutation Test should be used as a criteria*. 
    And will use *the Bootstrap to find confidence intervals and central positions 
    of metric statistics* in the groups under research.
    
    As a test statistic for the Permutation test will use *the difference in the statistics* of the test groups:
    '''
)
st.latex('\Delta \hat{TM} = \hat{TM}_1 - \hat{TM}_2,')

st.markdown(
    '''
    where $\hat{TM}_1$ and $\hat{TM}_2$ are the values of the statistics 
    of the first and second test groups, respectively.
    
    ## Selecting the significance level for testing statistical hypotheses and the confidence level for interval estimation of statistics
    
    The cost of error in the research is not as high as, for example, in a research of drug effectiveness, 
    so it's possible to choose a standard significance level $\\alpha$ equal to *0.05*. 
    The confidence level $\\beta$, accordingly, will be chosen equal to *0.95* (1-$\\alpha$).

    ## Decision rule for customer groups belonging to the same or different populations
    
    Will make a decision on the belonging of two test groups based on the results of a statistical test, 
    using the following rule:
    - if the *p-value for all metrics is below the significance level*, 
    i.e. if the null hypothesis can be regected for all metrics, 
    then *will consider that the customers of the test groups belong to the same population*
    - otherwise, will consider that the customers of the test groups belong to different populations.
    '''
)
