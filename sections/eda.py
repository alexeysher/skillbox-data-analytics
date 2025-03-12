import streamlit as st
import pandas as pd
from scipy import stats
import plotly.express as px
from auxiliary import wrap_text, set_text_style, MegafonColors
from functions import plot_metric_histograms

@st.cache_resource(show_spinner='Plotting...')
def plot_csat_dist_10(data: pd.DataFrame):
    fig = px.histogram(data, x='Q1', histnorm='percent', opacity=0.5)
    fig.update_traces(texttemplate="%{y:.1f}%", hovertemplate='%{x} - %{y:.1f}%',
                      marker_color=px.colors.DEFAULT_PLOTLY_COLORS)
    fig.update_layout(font_color=MegafonColors.scantBlue2, font_size=14,
                      height=400, bargap=0.2)
    fig.update_xaxes(title='', tickvals=data['Q1'].sort_values().unique(),
                     tickfont_size=14)
    fig.update_yaxes(title='',
                     tickfont_size=14)
    return fig


@st.cache_resource(show_spinner='Plotting...')
def plot_csat_dist_5(data: pd.DataFrame):
    s = (data['Q1'] + 1) // 2
    fig = px.histogram(s, x='Q1', histnorm='percent', opacity=0.5)
    fig.update_traces(texttemplate="%{y:.1f}%", hovertemplate='%{x} - %{y:.1f}%',
                      marker_color=px.colors.DEFAULT_PLOTLY_COLORS)
    fig.update_layout(font_color=MegafonColors.content, font_size=14,
                      height=400,
                      bargap=0.2)
    fig.update_xaxes(title='', tickfont_size=14)
    fig.update_yaxes(title='', tickfont_size=14)
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
    s.index = s.index.map(lambda x: wrap_text(x, 50))
    for index in range(s.size):
        s.iloc[index] = data[data['Q1'] <= 8]['Q2'].str.contains(str(index)).sum()
    s = s / s.sum() * 100

    fig = px.bar(s, orientation='h', opacity=0.5)
    fig.update_traces(texttemplate="%{x:.1f}%", hovertemplate='%{y} - %{x:.1f}%',
                      marker_color=px.colors.DEFAULT_PLOTLY_COLORS)
    fig.update_layout(
        height = 500,
        showlegend=False,
        bargap=0.3, font_size=14
    )
    fig.update_xaxes(title='', tickfont_size=14)
    fig.update_yaxes(title='', tickfont_size=14)
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
    fig.update_layout(
        font_size = 14,
        height = 330,
        showlegend=False,
        bargap=0.3)
    fig.update_xaxes(title='', tickfont_size=14)
    fig.update_yaxes(title='', tickfont_size=14)
    return fig


@st.cache_resource(show_spinner='Displaying...')
def show_metric_impact_table(metrics: pd.DataFrame):
    metrics = metrics.reset_index()
    # metrics.index.rename('Метрика', inplace=True)
    # style = metrics.style.set_caption('ddd').set_properties({'font-size': '20px'})
    header = f''
    table = \
        f'| {set_text_style("Metric", tag="span", text_align="center")} ' \
        f'| {set_text_style("Impact", tag="span", text_align="center")} |\n' \
        f'|---------|:-------:|\n'
    for _, (metric, impact) \
            in metrics[['metric', 'impact']].iterrows():
        if impact == '+':
            impact = set_text_style('▲', tag='span', color=MegafonColors.brandGreen)
        elif impact == '-':
            impact = set_text_style('▼', tag='span', color='red')
        else:
            impact = '─'
        table += f'|{metric}|{impact}|\n'

    return table


@st.cache_resource(show_spinner='Displaying...')
def show_metric_impact_legend():
    legend = '''
    - ─ - Neutral
    - :green[▲] -  The higher, the better
    - :red[▼] -  The lower, the better
    '''
    return legend


@st.cache_resource(show_spinner='Plotting...')
def plot_all_metric_histograms(data: pd.DataFrame, metrics: pd.DataFrame):
    fig = plot_metric_histograms(
        data, metrics,
        title=' ',
        labels_font_size=14,
        height=700, boxplot_height_fraq=0.15, n_cols=4, opacity=0.5,
        histnorm='probability density',
        add_boxplot=True, add_kde=True, add_mean=True,
        horizontal_spacing=0.07, vertical_spacing=0.15).update_layout(margin_t=60, margin_b=0)
    return fig


st.markdown(
    '''
    # Exploratory data analysis
    
    ## Analyzing answers to the 1st question

    To analyze the structure of the answers to the 1st question will build a histogram 
    of the percentage distribution of answers to this question.

    ### Distribution of customers by quality satisfaction scores
    '''
)

with st.columns([60, 20, 20])[0]:
    fig = plot_csat_dist_10(st.session_state.data)
    st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)

st.markdown(
    '''
    The histogram shows that more than a quarter of customers (`27.7%`) 
    are completely satisfied with the quality of communication. 
    But it is also worth noting that a significant part (`17.4%`) gave an extremely negative assessment.

    The main thing to pay attention to is that the percentage of related assessments among customers 
    often differs several times. For example, the share of customers who gave a score of `6` is only `3.3%`, 
    but the share of customers who gave a score of `5` and `7` is approximately 2 times greater. 
    This circumstance may indicate that it was difficult for customers who participated in the survey 
    to give a score on a 10-point scale. Indeed, it is quite difficult to determine 
    the difference between a scores `6` and `7`. Even the survey organizers themselves, as if anticipating this, 
    did not ask the 2nd question to customers who gave a score `9` as well as a score `10`. 
    To verify it will convert the `10`-point scale to `5`-point and look at the distribution again..
    
    ### Distribution of customers by quality satisfaction score on a 5-point scale
    '''
)

with st.columns([32, 30, 38])[0]:
    fig = plot_csat_dist_5(st.session_state.data)
    st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)

st.markdown(
    '''
    This distribution looks **more uniform** and **probable**. 
    Will take this circumstance into account in further work.
    '''
)

st.markdown(
    '''
    ## Analyzing answers to the 2nd question
    
    To analyze the structure of the answers to the 1st question will build a histogram of the percentage distribution 
    of answers to this question.
    
    ### Distribution of reasons for quality dissatisfaction
    '''
)

with st.columns([60, 20, 20])[0]:
    fig = plot_reason_dist(st.session_state.data)
    st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)

st.markdown(
    '''
    The diagram shows that almost 1/5 (`19.9%`) of customers are dissatisfied with mobile Internet. 
    A smaller, but still significant (`7.1%`) part of the responders are dissatisfied with the video loading speed.
    
    Will look at the structure of the responses, grouping them by services. 
    So, responses `1` and `2` refer to the voice communication service, `4` and `5` to the mobile Internet service. 
    Responses `6` and `7`, as well as the absence of responses, 
    do not give us an idea of the reason for the customer's lower scoring. So they can also be grouped. 
    Rating `3` shows that the customer is dissatisfied with the quality of coverage.
    
    For analyze purpose will build a diagram that shows the structure 
    of the responses in the context of the above groups.
    
    ### Distribution of reasons for quality dissatisfaction grouped by services
    '''
)

with st.columns([60, 20, 20])[0]:
    fig = plot_reason_combo_dist(st.session_state.data)
    st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)

st.markdown(
    '''
    As can see, the answers are distributed into approximately equal parts.
    About `1/4` of the customers are dissatisfied with voice communication, mobile Internet or coverage.
    
    ## Analyzing mobile internet metrics
    
    First, will describe the metrics that are present in the dataset. This description will be useful later.
    Will collect information about the names (`name`) and units of measurement (`units`) of the metrics. 
    This information will be needed when outputting the reporting text and graphic information.
    
    For each metric, will indicate what influence (`impact`) it has on the quality 
    of the mobile internet service and, accordingly, on customer satisfaction. 
    This information will be useful to us in further research.
    
    Metrics with a "positive" (:green[▲]) impact include metrics whose value is "the higher, the better":
    - `Downlink Throughput(Kbps)`;
    - `Uplink Throughput(Kbps)`;
    - `Video Streaming Download Throughput(Kbps)`;
    - `Web Page Download Throughput(Kbps)`.
    
    Metrics with "negative" (:red[▼]) influence include metrics whose value is "the lower, the better":
    - `Downlink TCP Retransmission Rate(%)`;
    - `Video Streaming xKB Start Delay(ms)`;
    - `Web Average TCP RTT(ms)`.
    
    Since the `'Total Traffic(MB)'` metric is only an indicator of the intensity 
    of mobile Internet usage by the customer, will indicate its influence as absent (─).
    
    ### Impact of the metrics to the quality of mobile Internet service
    '''
)

metrics_table = show_metric_impact_table(st.session_state.metrics)
st.markdown(metrics_table, unsafe_allow_html=True)
legend = show_metric_impact_legend()
st.markdown(legend, unsafe_allow_html=True)
st.markdown(
    '''
    Next, will analyze the distributions of the metric values. 
    First of all, this must be done in order to correctly select statistics for assessing 
    the central position of the distributions and the criteria that can be used to test statistical hypotheses. 
    To assess the distribution, will use standard tools - a histogram and a "box with whiskers". 
    Additionally, for the convenience of assessing the shape of the distribution, 
    will overlay the curves of the kernel density estimate (KDE) on the histograms.

    ### Probability density distribution and box plot   
    '''
)

fig = plot_all_metric_histograms(st.session_state.data[st.session_state.metrics.index], st.session_state.metrics)
st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
st.markdown(
    '''
    Based on the histogram analysis, can make the following observations:
    - The distributions of all metrics except `Total traffic` 
    are strongly skewed to the right and have a very long thin "tail"
    - The distribution of all metrics is far from "normal"
    - A lot of values on the right are located far from the rest, 
    which raises questions about their reliability (these may be the so-called "outliers").
    
    As noted, there is a lot of data in the sample that are suspected of being "outliers". 
    They can have a negative impact on the reliability of the assessment of the central position of the distributions. 
    That is, the mean value will reflect a far different metric value from what an ordinary user "sees". 
    Will check this by deriving the percentile values corresponding to the mean values of the metrics.    
    '''
)

df = st.session_state.metrics['name'].to_frame().merge(
    st.session_state.data[st.session_state.metrics.index].apply(
        lambda s: stats.percentileofscore(s, s.mean())
    ).to_frame(),
    left_index=True, right_index=True
).rename(columns={'name': 'Metric', 0: 'Percentile'}).style.hide(axis='index').format(precision=0)
st.dataframe(df, hide_index=True)

st.markdown(
    '''
    There are two options for further action in this situation:
    - either try to clean the data from outliers;
    - or use robust (less susceptible) statistics to "outliers" to estimate the central positions of the distributions.
    
    There are no universal and reliable methods for detecting "outliers". 
    Since the sample distributions are very skewed to the right and are far from normal, 
    traditional "simple" methods such as the "3-sigma rule" or determining outliers by 
    the border of the "whiskers" boxpolot are not suitable. 
    Of course, more "advanced" methods such as 
    [Local Outlier Factor (LOF)](https://en.wikipedia.org/wiki/Local_outlier_factor) or 
    [Isolation Forest](https://en.wikipedia.org/wiki/Isolation_forest) can be applied, 
    but they all require fine-tuning. 
    Therefore, will use the second option for solving the problem. 
    **Will not remove "outliers", but will use statistics that are robust to them.**    
    '''
)
