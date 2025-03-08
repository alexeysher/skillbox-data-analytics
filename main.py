import streamlit as st
from streamlit_option_menu import option_menu
from streamlit import config
import pandas as pd
import plotly.express as px
from auxiliary import hide_menu_button, remove_blank_space, set_text_style, set_widget_style, MegafonColors, wrap_text
from functions import plot_metric_histograms, display_cat_info, trimean_mod, my_bootstrap, display_confidence_interval, \
    trimean_mod_diff, confidence_interval_info, plot_metric_confidence_interval, display_pvalues
from pickle import load


@st.cache_data
def load_data() -> [pd.DataFrame]:
    data = pd.read_csv('megafon.csv')
    data.drop(columns='user_id', inplace=True)
    data = data[data.Q1.notna()]
    data = data[data.Q1.str.isdecimal()]
    data.Q1 = data.Q1.astype(int)
    data = data[(data.Q1 >= 1) & (data.Q1 <= 10)]
    # Заполняем нулями отсутствующие ответы на второй вопрос
    data['Q2'] = data['Q2'].fillna('0')
    # Преобразуем строки с овтетами на второй вопрос в список оценок
    data['Q2'] = data['Q2'].str.split(', ')
    # Разворачиваем списки с оценками из ответа на второй вопрос.
    # В итоге у нас получается датасет, в котором для каджой оценки из ответа на второй вопрос создается отдельная строка
    data = data.explode('Q2')
    # Оставляем только те записи, в которых ответы на второй вопрос являются числами
    data = data[data['Q2'].astype(str).str.isdecimal()]
    # Теперь ответы на второй вопрос приводим к целочисленному типу
    data['Q2'] = data['Q2'].astype(int)
    # Оставляем только те записи, в которых ответы на второй вопрос являются числами от 0 до 7
    data = data[(data['Q2'] >= 0) & (data['Q2'] <= 7)]
    # Данные сортируем по номеру ответа на второй вопрос
    data.sort_values('Q2', inplace=True)
    # Обратно сворачиваем оценки данные на второй вопрос в строку с разделителем в виде запятой и пробела
    data['Q2'] = data['Q2'].astype(str)
    data['Q2'] = data['Q2'].groupby(level=0).apply(', '.join)
    # Удаляем дубликаты
    data.drop_duplicates(inplace=True)
    # Удаляем ответы "0" или "6", если есть ещё другие ответы
    data['Q2'] = data['Q2'].str.replace('([06], )|(, [06])', '', regex=True)
    # Оставляем только исследуемые данные
    data_clean = data.loc[(data['Q1'] >= 9) | data['Q2'].str.contains('[45]', regex=True)].copy()
    # Добавляем информацию об оценке сервиса мобильного интернета
    data_clean['Internet score'] = data_clean['Q1'].apply(
        lambda q1: 'Very unsatisfied' if q1 <= 2 else ('Unsatisfied' if q1 <= 4 else (
            'Neutral' if q1 <= 6 else ('Satisfied' if q1 <= 8 else 'Very satisfied'))))
    # Добавляем информацию о причине неудовлетворенности сервисом мобильного интернета
    data_clean['Dissatisfaction reasons'] = data_clean['Q2'].apply(
        lambda q2: 'Internet and Video' if q2.find('4, 5') >= 0 else (
            'Internet' if q2.find('4') >= 0 else (
                'Video' if q2.find('5') >= 0 else 'No')))
    # Сортируем данные так, чтобы причины неудовлетворенности располагались в обозначенном поряке
    data_clean.sort_values(['Q1', 'Q2'],
                           key=lambda x: x if x.name == 'Q1' else (x.str.contains('5') - x.str.contains('4, 5') * 2),
                           inplace=True)
    metrics = pd.DataFrame(
        columns=['name', 'units', 'impact'],
        index=pd.Index(data.columns.drop(['Q1', 'Q2']), name='metric')
    )

    for metric in metrics.index:
        name, units = metric[:-1].split('(')
        label = '<b>' + wrap_text(name, 50) + '</b>'
        metrics.loc[metric, ['name', 'units', 'label']] = name, units, label

    metrics.loc['Total Traffic(MB)', 'impact'
    ] = '0'

    metrics.loc[
        ['Downlink Throughput(Kbps)', 'Uplink Throughput(Kbps)',
         'Video Streaming Download Throughput(Kbps)',
         'Web Page Download Throughput(Kbps)'], 'impact'
    ] = '+'

    metrics.loc[
        ['Downlink TCP Retransmission Rate(%)',
         'Video Streaming xKB Start Delay(ms)',
         'Web Average TCP RTT(ms)'], 'impact'
    ] = '-'
    return data, data_clean, metrics


@st.cache_resource
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


# @st.cache_resource
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


@st.cache_resource
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


@st.cache_resource
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


@st.cache_resource
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


@st.cache_resource
def show_metric_impact_legend():
    # legend = '|    |    |\n'
    legend = '| Marker | Description |\n'
    legend += '|:--:|----|\n'
    legend += '| ─ | Neurtal |\n'
    legend += f'|{set_text_style('▲', tag='span', color=MegafonColors.brandGreen)}| The higher, the better|\n'
    legend += f'|{set_text_style('▼', tag='span', color='red')}| The lower, the better |'
    return legend


config.dataFrameSerialization = "arrow"

st.set_page_config(page_title='Research of MegaFon (large mobile and telecom operator) customer success survey',
                   page_icon='bar-chart', layout='wide')


data, data_clean, metrics = load_data()
research_metrics = metrics.loc[[
    'Downlink Throughput(Kbps)',
    'Video Streaming Download Throughput(Kbps)',
    'Web Page Download Throughput(Kbps)'
]]
research_metrics['statistic'] = trimean_mod
research_metrics['test statistic'] = trimean_mod_diff

alpha = 0.05
betta = 1 - alpha

with st.sidebar:
    choice = option_menu(
        '',
        options=[
            "Title",
            "Introduction",
            "Data preparation and exploratory analysis",
            "Setting the objectives",
            "Selection of metrics, statistics and criteria",
            "Reasons for dissatisfaction with mobile Internet service",
            "Mobile Internet service quality assessments",
            "CSAT of mobile Internet service",
            "Influence of the metrics on the CSAT of mobile Internet service",
            "---",
            "Summary"
        ],
        icons=[
            "",
            "map",
            "eyeglasses",
            "filter",
            "hand-thumbs-down",
            "badge-4k",
            "sort-numeric-down-alt",
            "speedometer2",
            'book'
        ],
        orientation='vertical',
        # styles={"container": {"padding": "5!important", "background-color": "#fafafa"}},
        key='main_menu'
    )

match choice:
    case "Title":
        plain_text = "Research of MegaFon customer success survey"
        font_formatted_text = f'**{set_text_style(plain_text, font_size=80, color=MegafonColors.brandPurple)}**'
        st.markdown(font_formatted_text, unsafe_allow_html=True)
        font_formatted_text = set_text_style('&nbsp;', font_size=24)
        st.markdown(font_formatted_text, unsafe_allow_html=True)
        author = set_text_style('Author: ', font_size=48, color=MegafonColors.brandGreen, tag='span') + \
                 set_text_style('**Alexey Sherstobitov**', font_size=48, color=MegafonColors.brandGreen, tag='span')
        st.markdown(author, unsafe_allow_html=True)
    case "Introduction":
        plain_text = "<b>Problem statement</b>"
        font_formatted_text = f'**{set_text_style(
            plain_text, font_size=24, 
            color=MegafonColors.brandPurple
        )}**'
        st.markdown(font_formatted_text, unsafe_allow_html=True)
        plain_text = """
        <a href="https://en.wikipedia.org/wiki/MegaFon">Megafon</a> is a large mobile phone and telecom operator.
        Like any business, this company wants to increase customer satisfaction with its service quality.
        """
        font_formatted_text = set_text_style(plain_text, tag='p', font_size=20)
        st.html(font_formatted_text)
        plain_text = """
        For this reason, the company have managed a short customers survey.  
        First of all, the customers were asked to score their satisfaction for communication service quality 
        on 10-point scale (10 - excellent, 1 - terrible)."""
        font_formatted_text = set_text_style(plain_text, tag='p', font_size=20)
        st.html(font_formatted_text)
        plain_text = """        
        If the customer score the quality of communication at 9 or 10 points, the survey ended. 
        If the customer score it below 9, a second question was asked about the reasons for dissatisfaction.
        For the second question the numbered answer options were provided:"""
        font_formatted_text = set_text_style(plain_text, tag='p', font_size=20)
        st.html(font_formatted_text)
        plain_text = """        
        <li>Unknown</li>
        <li>Missed calls, disconnected calls</li>
        <li>Waiting time for ringtones</li>
        <li>Poor connection quality in buildings, shopping centers, etc.</li>
        <li>Slow mobile Internet</li>
        <li>Slow video loading</li>
        <li>Difficult to answer</li>
        <li>Your own option</li>
        """
        font_formatted_text = set_text_style(plain_text, tag='ol start="0"', font_size=20)
        st.html(font_formatted_text)
        plain_text = """        
        The answer could be given in a free format or by listing the answer numbers separated by commas."""
        font_formatted_text = set_text_style(plain_text, tag='p', font_size=20)
        st.html(font_formatted_text)
    case "Data preparation and exploratory analysis":
        tabs = ['Answers for the 1st question', 'Answers for the 2nd question', 'Metric values']
        tab1, tab2, tab3 = st.tabs(tabs)
        for tab in tabs:
            set_widget_style(tab, font_size=24)
        with tab1:
            st.markdown(
                set_text_style(
                    '<b>Distribution of customers by quality satisfaction scores</b>',
                    font_size=24, text_align='center', color=MegafonColors.brandPurple
                ),
                unsafe_allow_html=True
            )
            c1, c2, c3 = st.columns([54, 16, 30], gap='medium')
            with c1:
                col_title = set_text_style('<b>✘</b> ', tag='span', color='red') + 'Initial (10-score scale)'
                col_title = set_text_style(col_title, font_size=20, text_align='center')
                st.markdown(col_title, unsafe_allow_html=True)
                fig = plot_csat_dist_10(data)
                st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
                # conclusion = set_text_style(
                #     """The distribution doesn't look smooth and balanced.
                #     It indicates that customers probably had a difficulty with the scoring on this scale.""",
                #     font_size=20
                # )
                # st.markdown(conclusion, unsafe_allow_html=True)
            with c2:
                conclusion = set_text_style(
                    """<br><br><br><br>Converting scores to 5-point scale""",
                    font_size=20
                )
                st.markdown(conclusion, unsafe_allow_html=True)
                col_title = set_text_style('<b>►</b> ', tag='span')
                col_title = set_text_style(col_title, font_size=64, text_align='center')
                st.markdown(col_title, unsafe_allow_html=True)
            with c3:
                col_title = set_text_style('<b>✔</b> ', tag='span', color=MegafonColors.brandGreen) + \
                            'Modified (5-score scale)'
                col_title = set_text_style(col_title, font_size=20, text_align='center')
                st.markdown(col_title, unsafe_allow_html=True)
                fig = plot_csat_dist_5(data)
                st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
                # conclusion = set_text_style(
                #     """The distribution looks much more probable""",
                #     font_size=20
                # )
                # st.markdown(conclusion, unsafe_allow_html=True)

            st.markdown(
                set_text_style(
                    '<b>Observation</b>',
                    font_size=24, text_align='center', color=MegafonColors.orangeDark
                ),
                unsafe_allow_html=True
            )
            st.markdown(
                set_text_style(
                    'The scores distribution on 5-point scale looks much more probable.',
                    font_size=20
                ),
                unsafe_allow_html=True
            )
        with tab2:
            st.markdown(
                set_text_style('<b>Distribution of reasons for quality dissatisfaction</b>',
                               color=MegafonColors.brandPurple,
                               font_size=24, text_align='center'),
                unsafe_allow_html=True
            )
            c1, c2, c3 = st.columns([45, 16, 39], gap='medium')
            with c1:
                col_title = set_text_style('<b>✘</b> ', tag='span', color='red') + 'Initial'
                col_title = set_text_style(col_title, font_size=20, text_align='center')
                st.markdown(col_title, unsafe_allow_html=True)
                fig = plot_reason_dist(data)
                st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            with c2:
                conclusion = set_text_style(
                    """<br><br><br><br>Grouping by services""",
                    font_size=20
                )
                st.markdown(conclusion, unsafe_allow_html=True)
                col_title = set_text_style('<b>►</b> ', tag='span')
                col_title = set_text_style(col_title, font_size=64, text_align='center')
                st.markdown(col_title, unsafe_allow_html=True)
            with c3:
                col_title = set_text_style('<b>✔</b> ', tag='span', color=MegafonColors.brandGreen) + 'Combined'
                col_title = set_text_style(col_title, font_size=20, text_align='center')
                st.markdown(col_title, unsafe_allow_html=True)
                fig = plot_reason_combo_dist(data)
                st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            st.markdown(
                set_text_style(
                    '<b>Observation</b>',
                    font_size=24, text_align='center', color=MegafonColors.orangeDark
                ),
                unsafe_allow_html=True
            )
            st.markdown(
                set_text_style(
                    'Answers are distributed into approximately equal parts. '
                    'About 1/4 of the customers are unsatisfied with voice communication, '
                    'mobile Internet or coverage.',
                    tag='ul',
                    font_size=20
                ),
                unsafe_allow_html=True
            )
        with tab3:
            # st.markdown(
            #     set_text_style('<b>Impact of the metrics to the quality of mobile Internet service</b>',
            #                    font_size=24, text_align='center', color=MegafonColors.brandPurple),
            #     unsafe_allow_html=True)
            # metrics_table = show_metric_impact_table(metrics)
            # c1, c2 = st.columns([53, 47], gap='medium')
            # with c1:
            #     st.markdown(metrics_table, unsafe_allow_html=True)
            # with c2:
            #     legend = show_metric_impact_legend()
            #     st.markdown(legend, unsafe_allow_html=True)
            # st.markdown('---')
            st.markdown(
                set_text_style('<b>Probability density distribution and box plot</b>',
                               color=MegafonColors.brandPurple,
                               font_size=24, text_align='center'),
                unsafe_allow_html=True
            )
            fig = plot_metric_histograms(
                data[metrics.index], metrics,
                # title='<b>Плотность распределения вероятностей метрик в наблюдаемой выборке</b>',
                # title_y=0.95, title_font_size=24,
                labels_font_size=20,
                axes_tickfont_size=14,
                height=900, boxplot_height_fraq=0.15, n_cols=3, opacity=0.5,
                histnorm='probability density',
                add_boxplot=True, add_kde=True, add_mean=True,
                horizontal_spacing=0.07, vertical_spacing=0.15)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            st.markdown(
                set_text_style(
                    '<b>Observation</b>',
                    font_size=24, text_align='center', color=MegafonColors.orangeDark
                ),
                unsafe_allow_html=True
            )
            st.markdown(
                set_text_style(
                    '''
                        <li>The distributions of all metrics except Total traffic are strongly skewed to the right and have a very long thin "tail"</li>
                        <li>The distributions of all metrics are far from "normal"</li>
                        <li>A lot of values on the right are located far from the rest, which raises questions about their reliability (these may be the so-called "outliers")</li>
                    ''',
                    tag='ul',
                    font_size=20
                ),
                unsafe_allow_html=True
            )
    case "Setting the objectives":
        tabs = ['Objectives of the research',
                'Initial customers classification']
        tab1, tab2 = st.tabs(tabs)
        for tab in tabs:
            set_widget_style(tab, font_size=24)
        with tab1:
            st.markdown(set_text_style('<b>Objectives</b>', font_size=24, text_align='center',
                                       color=MegafonColors.brandPurple), unsafe_allow_html=True)
            st.markdown(
                set_text_style(
                    '''
                    <li>Classify customers according to their assessment of the quality of mobile Internet service.</li>
                    <li>Determine witch metrics have the strongest influence to the customers assessment.</li> 
                    ''',
                    tag='ul', font_size=20),
                unsafe_allow_html=True)
            st.markdown('&nbsp;')
            df = pd.DataFrame(
                index=[
                    "0 - Unknown", "1 - Missed calls, disconnected calls",
                    "2 - Waiting time for ringtones",
                    "3 - Poor connection quality in buildings, shopping centers, etc.",
                    "4 - Slow mobile Internet", "5 - Slow video loading",
                    "6 - Difficult to answer", "7 - Your own option"
                ],
                columns=pd.RangeIndex(1, 11)
            )
            df.loc[("4 - Slow mobile Internet", "5 - Slow video loading"), :] = '+'
            df.loc[:, (9, 10)] = '+'
            df.fillna('-', inplace=True)
            s = df.style
            s.set_table_styles([
                # {'selector': 'th, td', 'props': 'border: 1px solid lightgray;'},
                {'selector': 'th:not(.blank)',
                 'props': f'font-size: 20px; color: white; background-color: {MegafonColors.brandPurple80}; font-weight: normal;'},
                {'selector': 'th.col_heading', 'props': 'text-align: center; width: 50px;'},
                {'selector': 'td', 'props': 'text-align: center; font-size: 20px; font-weight: normal;'},
                {'selector': 'th.blank', 'props': 'border-style: none'}
            ], overwrite=False)
            s = s.map(lambda v:
                      f'color: white; background-color: {MegafonColors.brandGreen};' if v == '+'
                      else f'color: {MegafonColors.spbSky2}; background-color: {MegafonColors.spbSky1};')
            st.markdown(s.to_html(table_uuid="table_client_choosing"), unsafe_allow_html=True)
        with tab2:
            c1, c2 = st.columns(2, gap='large')
            with c1:
                st.markdown(set_text_style('<b>Categories of CSAT according to Q1 answer</b>', font_size=24,
                                           text_align='center', color=MegafonColors.brandPurple),
                            unsafe_allow_html=True)
                df = pd.DataFrame(
                    index=["Very unsatisfied", "Unsatisfied", "Neutral", "Satisfied", "Very satisfied"],
                    columns=pd.RangeIndex(1, 11)
                )
                df.loc["Very unsatisfied", (1, 2)] = '+'
                df.loc["Unsatisfied", (3, 4)] = '+'
                df.loc["Neutral", (5, 6)] = '+'
                df.loc["Satisfied", (7, 8)] = '+'
                df.loc["Very satisfied", (9, 10)] = '+'
                df.fillna('-', inplace=True)
                s = df.style
                s.set_table_styles([
                    # {'selector': 'th, td', 'props': 'border: 1px solid lightgray;'},
                    {'selector': 'th:not(.blank)',
                     'props': f'font-size: 20px; color: white; background-color: {MegafonColors.brandPurple80}; font-weight: normal;'},
                    {'selector': 'th.col_heading', 'props': 'text-align: center; width: 50px;'},
                    {'selector': 'td', 'props': 'text-align: center; font-size: 20px; font-weight: normal;'},
                    {'selector': 'th.blank', 'props': 'border-style: none'}
                ], overwrite=False)
                s = s.map(lambda v:
                          f'color: white; background-color: {MegafonColors.brandGreen};' if v == '+'
                          else f'color: {MegafonColors.spbSky2}; background-color: {MegafonColors.spbSky1};')
                st.markdown(s.to_html(table_uuid="table_categories_1"), unsafe_allow_html=True)
            with c2:
                st.markdown(
                    set_text_style(
                        '<b>Categories of reasons for dissatisfaction with the quality of mobile Internet service '
                        'according to Q2 aswers</b>',
                        font_size=24, text_align='center', color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
                df = pd.DataFrame(
                    index=["Internet and Video", "Internet", "Video", "No"],
                    columns=[
                        "4 - Slow mobile Internet", "5 - Slow video loading",
                    ]
                )
                df.loc["Internet and Video", ("4 - Slow mobile Internet", "5 - Slow video loading")] = '+'
                df.loc["Internet", "4 - Slow mobile Internet"] = '+'
                df.loc["Video", "5 - Slow video loading"] = '+'
                df.fillna('-', inplace=True)
                s = df.style
                s.set_table_styles([
                    # {'selector': 'th, td', 'props': 'border: 1px solid lightgray;'},
                    {'selector': 'th:not(.blank)',
                     'props': f'font-size: 20px; color: white; background-color: {MegafonColors.brandPurple80}; font-weight: normal;'},
                    {'selector': 'th.col_heading', 'props': 'text-align: center; width: 240px;'},
                    {'selector': 'td', 'props': 'text-align: center; font-size: 20px; font-weight: normal;'},
                    {'selector': 'th.blank', 'props': 'border-style: none'}
                ], overwrite=False)
                s = s.map(lambda v:
                          f'color: white; background-color: {MegafonColors.brandGreen};' if v == '+'
                          else f'color: {MegafonColors.spbSky2}; background-color: {MegafonColors.spbSky1};')
                st.markdown(s.to_html(table_uuid="table_categories_2"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            st.markdown(set_text_style('<b>Distribution of customers</b>', font_size=24,
                                       text_align='center', color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            s = display_cat_info(data_clean)
            st.markdown(s.to_html(table_uuid="table_categories_dist"), unsafe_allow_html=True)
    case "Selection of metrics, statistics and criteria":

        @st.cache_resource
        def show_important_metric_table(metrics: pd.DataFrame):
            df = metrics.loc[
                ['Downlink Throughput(Kbps)', 'Video Streaming Download Throughput(Kbps)',
                'Web Page Download Throughput(Kbps)'], ['name']
            ]
            return df.style


        @st.cache_resource
        def plot_metric_histograms_4_1():
            return plot_metric_histograms(statistic_distributions, statistic=statistic_distributions.median(),
                                          metrics=research_metrics,
                                          title='<b>Плотность распределения вероятностей статистики</b>', title_y=0.9,
                                          labels_font_size=16,
                                          units_font_size=16,
                                          axes_tickfont_size=14,
                                          height=300, n_cols=3, opacity=0.5,
                                          histnorm='probability density',
                                          add_kde=True, add_statistic=True, mark_confidence_interval=True,
                                          horizontal_spacing=0.06, vertical_spacing=0.07)


        @st.cache_data
        def get_stat_dist():
            return data[research_metrics.index] \
                .apply(lambda s: my_bootstrap(s, research_metrics.loc[s.name, 'statistic'], n_resamples=9999))


        tabs = ['Metrics and statistics',
                'Criteria and significance level']
        tab1, tab2 = st.tabs(tabs)
        for tab in tabs:
            set_widget_style(tab, font_size=24)
        with tab1:
            st.markdown(set_text_style('<b>Metrics for Internet speed assessment</b>', font_size=24,
                                       color=MegafonColors.brandPurple, text_align='center'),
                        unsafe_allow_html=True)
           # st.markdown(set_text_style('<b>Important metrics:</b>', font_size=24, color=MegafonColors.orangeDark),
            #             unsafe_allow_html=True)
            st.markdown(
                set_text_style(
                    'According to [Ookla](https://www.ookla.com/) experts 90% of the user assessment of Internet '
                    'is attributed to download speed and the remaining 10% to upload speed. '
                    'Therefore, in the further research only the following metrics will be focused on:', tag='span',
                    font_size=20),
                unsafe_allow_html=True
            )
            st.markdown('''       
            <ul>
            <li><span style="font-size:20px">Downlink Throughput</span></li>
            <li><span style="font-size:20px">Video Streaming Download Throughput</span></li>
            <li><span style="font-size:20px">Web Page Download Throughput</span></li>
            </ul>
            ''', unsafe_allow_html=True)
            st.markdown('---')
            st.markdown(set_text_style('<b>Statistic for assessing of the average value of the metrics</b>',
                                       font_size=24,
                                       color=MegafonColors.brandPurple, text_align='center'),
                        unsafe_allow_html=True)
            st.markdown(
                set_text_style('''[Ookla](https://www.ookla.com/) uses modified trimmer in the methodology 
                for assessing the average speed of Internet connections of providers:''', tag='span',
                               font_size=20),
                unsafe_allow_html=True
            )
            st.latex(r'\hat{TM}={P_{10}+8\cdot P_{50}+P_{90} \over {10}}')
            st.markdown('---')
            st.markdown(set_text_style('<b>Probability density function of statistics</b>',
                                       font_size=24,
                                       color=MegafonColors.brandPurple, text_align='center'),
                        unsafe_allow_html=True)
            statistic_distributions = get_stat_dist()
            fig = plot_metric_histograms_4_1()
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            st.markdown(set_text_style('The distributions of statistics are almost symmetrical. '
                                       'Therefore, the central position of statistic can be estimated as a middle of '
                                       'confidence interval.',
                                       font_size=20),
                        unsafe_allow_html=True)
        with tab2:
            st.markdown('')
            # c1, c2 = st.columns(2)
            # with c1:
            st.markdown(set_text_style('<b>Statistical criteria (test)</b>', font_size=24,
                                       color=MegafonColors.brandPurple,
                                       text_align='center'),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('Permutation test', font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            # with c2:
            st.markdown(set_text_style('<b>Estimating the confidence interval</b>', font_size=24, color=MegafonColors.brandPurple,
                                       text_align='center'),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('Bootstrapping', font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            st.markdown('---')
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(set_text_style('<b>Significance level</b>', font_size=24, color=MegafonColors.brandPurple,
                                           text_align='center'),
                            unsafe_allow_html=True)
                st.latex(r'\large \\alpha=0.05')
            with c2:
                st.markdown(set_text_style('<b>Confidence level</b>', font_size=24, color=MegafonColors.brandPurple,
                                           text_align='center'),
                            unsafe_allow_html=True)
                st.latex(r'\large \\beta=0.95\\(1-alpha)')
            st.markdown('---')
            st.markdown(set_text_style('<b>Decision rule</b>', font_size=24, color=MegafonColors.brandPurple,
                                       text_align='center'),
                        unsafe_allow_html=True)
            rule_text = '''
            <ul>
            <li><span style="font-size:24px">If the <b>p-value is less then significance level for all metrics</b>, 
            i.e. if the null hypothesis can be rejected for all metrics, 
            then it's considered that <b>the customers of the test groups belong to the same population</b><span></li>
            <li><span style="font-size:24px"><b>Otherwise</b>, it's considered that <b>the customers of the test groups 
            belong to different populations</b></li>
            </ul>
            '''
            st.markdown(rule_text, unsafe_allow_html=True)
    case "Reasons for dissatisfaction with mobile Internet service":
        tabs = ['Objective of research',
                'Exploratory analysis',
                'Statistical tests',
                'Conclusion']
        tab1, tab2, tab3, tab4 = st.tabs(tabs)
        for tab in tabs:
            set_widget_style(tab, font_size=24)
        with tab1:
            c1, c2 = st.columns(2, gap='medium')
            with c1:
                st.markdown('&nbsp;')
                s = display_cat_info(data_clean).set_properties(pd.IndexSlice['Very unsatisfied':'Satisfied', 'Internet and Video'],
                                                                color='white',
                                                                background=px.colors.DEFAULT_PLOTLY_COLORS[0],
                                                                opacity=0.5).set_properties(
                    pd.IndexSlice['Very unsatisfied':'Satisfied', 'Internet'], color='white',
                    background=px.colors.DEFAULT_PLOTLY_COLORS[1], opacity=0.5).set_properties(
                    pd.IndexSlice['Very unsatisfied':'Satisfied', 'Video'], color='white',
                    background=px.colors.DEFAULT_PLOTLY_COLORS[2],
                    opacity=0.5)
                st.markdown(s.to_html(table_uuid="table_categories_dist"), unsafe_allow_html=True)
                categories = '''
            <br>
            <ul>
            <li><span style="color:white;background-color:rgb(31, 119, 180);opacity:0.5;font-size:20px">&nbsp;Internet and Video&nbsp;</span> - <span style="font-size:20px">unsatisfied with the speed of mobile Internet and Video loading</span></li>
            <li><span style="color:white;background-color:rgb(255, 127, 14);opacity:0.5;font-size:20px">&nbsp;Internet&nbsp;</span> - <span style="font-size:20px">unsatisfied primarily with the speed of mobile Internet</span></li>
            <li><span style="color:white;background-color:rgb(44, 160, 44);opacity:0.5;font-size:20px">&nbsp;Video&nbsp;</span> - <span style="font-size:20px">unsatisfied primarily with the speed of Video loading</span></li>
            </ul>
                            '''
                st.markdown(categories, unsafe_allow_html=True)
            with c2:
                st.markdown('&nbsp;')
                goal_text = set_text_style('<b>Objective</b><br>', tag='p', color=MegafonColors.brandPurple, font_size=24,
                                           text_align='center')
                goal_text += set_text_style('Determine whether the division is statistically correct?',
                                            tag='span', font_size=20)
                st.markdown(goal_text, unsafe_allow_html=True)
                st.markdown('&nbsp;')
                task_text = set_text_style('<b>Questions</b><br>', tag='p',
                                           color=MegafonColors.brandPurple, font_size=24, text_align='center')
                task_text += set_text_style(f"""
                <ul>
                <li>{set_text_style('Do customers in the groups belong to different populations?',
                                    font_size=20)}</li>
                <li>{set_text_style('If customers do not belong to the same population, '
                                    'what metrics do they differ in?', font_size=20)}</li>
                </ul>
                """, tag='span', font_size=20)
                st.markdown(task_text, unsafe_allow_html=True)
        with tab2:
            @st.cache_resource
            def load_5_3():
                with open('data/5_3.dmp', 'rb') as fp:
                    data = load(fp)
                return data


            @st.cache_resource
            def ci_plot_5_3():
                return plot_metric_confidence_interval(ci, metrics=research_metrics,
                                                       title='', height=250, n_cols=3,
                                                       horizontal_spacing=0.04, vertical_spacing=0.07,
                                                       labels_font_size=16, axes_tickfont_size=14, units_font_size=14)


            research_data, groups, group_pairs, ci, ci_overlapping, ci_center = load_5_3()

            # st.markdown('&nbsp;')
            st.markdown(set_text_style('<b>Confidence intervals of statistics</b>', text_align='center',
                                       font_size=24, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            fig = ci_plot_5_3()
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            s = display_confidence_interval(ci, metrics=research_metrics,
                                            caption='', caption_font_size=12,
                                            opacity=0.5, precision=1)
            st.markdown(s.to_html(table_uuid="table_categories_dist_5_3"), unsafe_allow_html=True)
        with tab3:
            @st.cache_resource
            def load_5_4_1():
                with open('data/5_4_1.dmp', 'rb') as fp:
                    data = load(fp)
                return data


            @st.cache_resource
            def plot_metric_histograms_5_4_1(null_distributions, statistics, research_metrics, mark_statistic):
                return plot_metric_histograms(
                    null_distributions, statistic=statistics, metrics=research_metrics,
                    # title=f'<b>Density of the null probability distribution of the test statistic</b>', title_y=0.9,
                    labels_font_size=16, axes_tickfont_size=14, units_font_size=14,
                    height=300, n_cols=3, opacity=0.5,
                    histnorm='probability density',
                    add_kde=True, add_statistic=True, mark_statistic=mark_statistic,
                    horizontal_spacing=0.08, vertical_spacing=0.07)


            @st.cache_resource
            def load_5_4_2():
                with open('data/5_4_2.dmp', 'rb') as fp:
                    data = load(fp)
                return data


            @st.cache_resource
            def plot_metric_histograms_5_4_2(null_distributions, statistics, research_metrics, mark_statistic):
                return plot_metric_histograms(
                    null_distributions, statistic=statistics, metrics=research_metrics,
                    # title=f'<b>Density of the null probability distribution of the test statistic</b>', title_y=0.9,
                    labels_font_size=16, axes_tickfont_size=14, units_font_size=14,
                    height=300, n_cols=3, opacity=0.5,
                    histnorm='probability density',
                    add_kde=True, add_statistic=True, mark_statistic=mark_statistic,
                    horizontal_spacing=0.08, vertical_spacing=0.07)


            @st.cache_resource
            def load_5_4_3():
                with open('data/5_4_3.dmp', 'rb') as fp:
                    data = load(fp)
                return data


            @st.cache_resource
            def plot_metric_histograms_5_4_3(null_distributions, statistics, research_metrics, mark_statistic):
                return plot_metric_histograms(
                    null_distributions, statistic=statistics, metrics=research_metrics,
                    # title=f'<b>Density of the null probability distribution of the test statistic</b>', title_y=0.9,
                    labels_font_size=16, axes_tickfont_size=14, units_font_size=14,
                    height=300, n_cols=3, opacity=0.5,
                    histnorm='probability density',
                    add_kde=True, add_statistic=True, mark_statistic=mark_statistic,
                    horizontal_spacing=0.08, vertical_spacing=0.07)


            alternatives, pvalues, mark_statistic, null_distributions, statistics = load_5_4_1()

            st.markdown(set_text_style(f'<b>Groups "{groups[0]}" и "{groups[1]}"</b>', text_align='center',
                                       font_size=24, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('<b>Hypothesis</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([40, 30, 30])
            st.markdown('&nbsp;')
            c1.latex(r'\Delta \hat{TM}=\hat{TM}_{Internet \, и \, видео}-\hat{TM}_{Internet}')
            c2.latex(r'H_0:\Delta \hat{TM}=0')
            c3.latex(r'H_1:\Delta \hat{TM}≠0')
            st.markdown(set_text_style('<b>Density of the null probability distribution of the test statistic</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            fig = plot_metric_histograms_5_4_1(null_distributions, statistics, research_metrics, mark_statistic)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            group_pair_index = 0
            st.markdown(set_text_style(f'<b>Statistical test p-values</b>', font_size=24,
                                       color=MegafonColors.brandGreenDarken10, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                                metrics=research_metrics, alpha=alpha,
                                caption='', col_width=400)
            st.markdown(s.to_html(table_uuid="table_pvalues_5_4_1"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Outcomes</b>', tag='p',
                                             color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'есть значимые различия</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)

            _, pvalues, mark_statistic, null_distributions, statistics = load_5_4_2()

            st.markdown('---')
            st.markdown(set_text_style(f'<b>Groups "{groups[1]}" и "{groups[2]}"</b>', text_align='center',
                                       font_size=24, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('<b>Hypothesis</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([40, 30, 30])
            st.markdown('&nbsp;')
            c1.latex(r'\Delta \hat{TM}=\hat{TM}_{Internet \, и \, видео}-\hat{TM}_{Video}')
            c2.latex(r'H_0:\Delta \hat{TM}=0')
            c3.latex(r'H_1:\Delta \hat{TM}≠0')
            st.markdown(set_text_style('<b>Density of the null probability distribution of the test statistic</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            fig = plot_metric_histograms_5_4_2(null_distributions, statistics, research_metrics, mark_statistic)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            group_pair_index = 1
            st.markdown(set_text_style(f'<b>Statistical test p-values</b>', font_size=24,
                                       color=MegafonColors.brandGreenDarken10, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                                metrics=research_metrics, alpha=alpha,
                                caption='', col_width=400)
            st.markdown(s.to_html(table_uuid="table_pvalues_5_4_2"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Outcomes</b><br>', tag='p', font_size=24,
                                             color=MegafonColors.brandGreenDarken10, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'нет значимых различий</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)

            _, pvalues, mark_statistic, null_distributions, statistics = load_5_4_3()

            st.markdown('---')
            st.markdown(set_text_style(f'<b>Groups "{groups[2]}" и "{groups[0]}"</b>', text_align='center',
                                       font_size=24, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('<b>Hypothesis</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([40, 30, 30])
            st.markdown('&nbsp;')
            c1.latex(r'\Delta \hat{TM}=\hat{TM}_{Video}-\hat{TM}_{Internet \, и \, видео}')
            c2.latex(r'H_0:\Delta \hat{TM}=0')
            c3.latex(r'H_1:\Delta \hat{TM}≠0')
            st.markdown(set_text_style('<b>Density of the null probability distribution of the test statistic</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            fig = plot_metric_histograms_5_4_3(null_distributions, statistics, research_metrics, mark_statistic)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            group_pair_index = 2
            st.markdown(set_text_style(f'<b>Statistical test p-values</b>', font_size=24,
                                       color=MegafonColors.brandGreenDarken10, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                                metrics=research_metrics, alpha=alpha,
                                caption='', col_width=400)
            st.markdown(s.to_html(table_uuid="table_pvalues_5_4_3"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Outcomes</b>', tag='p',
                                             color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'есть значимые различия</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)
        with tab4:
            st.markdown(set_text_style(f'<b>Statistical tests p-values</b>', font_size=24,
                                       color=MegafonColors.brandPurple, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues, metrics=research_metrics, alpha=alpha,
                                caption='', opacity=0.5, index_width=180, col_width=300)
            st.markdown(s.to_html(table_uuid="table_pvalues_5_5"), unsafe_allow_html=True)
            st.markdown('---')
            # st.markdown('---')
            # st.markdown(set_text_style(f'<b>Категории по причинам недовольства сервисом мобильного интернета</b>', font_size=24,
            #                            color=MegafonColors.brandPurple, text_align='center'),
            #             unsafe_allow_html=True)
            # st.markdown('&nbsp;')
            c1, c2 = st.columns(2, gap='medium')
            with c1:
                s = display_cat_info(data_clean) \
                    .set_properties(pd.IndexSlice['Very unsatisfied':'Satisfied', 'Internet and Video'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[0], opacity=0.5) \
                    .set_properties(pd.IndexSlice['Very unsatisfied':'Satisfied', 'Internet': 'Video'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[1], opacity=0.5)
                st.markdown(s.to_html(table_uuid="table_categories_5_5"), unsafe_allow_html=True)
                categories = '''
                <br>
                <ul>
                <li><span style="color:white;background-color:rgb(31, 119, 180);opacity:0.5;font-size:20px">&nbsp;
                Internet and Video&nbsp;</span>&nbsp- Dissatisfied with the speed of mobile Internet and video loading</li>
                <li><span style="color:white;background-color:rgb(255, 127, 14);opacity:0.5;font-size:20px">&nbsp;
                Internet or Video&nbsp;</span>&nbsp- Dissatisfied with the speed of mobile Internet or video loading</li>
                </ul>
                '''
                st.markdown(categories, unsafe_allow_html=True)
            with c2:
                st.markdown(set_text_style(f'<b>Outcomes</b>', font_size=24,
                                           color=MegafonColors.orangeDark, text_align='center'),
                            unsafe_allow_html=True)
                conclusion_text = '''
                <ul>
                <li><span style="font-size:24px">Customers of groups "<b>Internet</b>" and "<b>Video</b>" 
                belong to <b>the same population</b>.</span></li>
                <li><span style="font-size:24px">Customers of group "<b>Internet and Video</b>" 
                belong to <b>separate population</b>. The strongest differences is in the <b>"Downlink Throughput"</b> 
                metric.</span></li>
                </ul>
                '''
                st.markdown(conclusion_text, unsafe_allow_html=True)
    case "Mobile Internet service quality assessments":
        tabs = ['Objective of research',
                'Exploratary analysis',
                'Statistical tests',
                'Conclusion']
        tab1, tab2, tab3, tab4 = st.tabs(tabs)
        for tab in tabs:
            set_widget_style(tab, font_size=24)
        with tab1:
            c1, c2 = st.columns(2, gap='medium')
            with c1:
                st.markdown('&nbsp;')
                s = display_cat_info(data_clean)\
                    .set_properties(pd.IndexSlice['Very unsatisfied', 'Internet and Video': 'Video'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[0], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Unsatisfied', 'Internet and Video': 'Video'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[1], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Neutral', 'Internet and Video': 'Video'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[2], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Satisfied', 'Internet and Video': 'Video'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[3], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Very satisfied', 'No'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[4], opacity=0.5)
                st.markdown(s.to_html(table_uuid="table_categories_dist"), unsafe_allow_html=True)
                categories = '''
            <br>
            <ul>
            <li><span style="color:white;background-color:rgb(31, 119, 180);opacity:0.5;font-size:20px">&nbsp;Very unsatisfied&nbsp;</span></li>
            <li><span style="color:white;background-color:rgb(255, 127, 14);opacity:0.5;font-size:20px">&nbsp;Unsatisfied&nbsp;</span></li>
            <li><span style="color:white;background-color:rgb(44, 160, 44);opacity:0.5;font-size:20px">&nbsp;Neutral&nbsp;</span></li>
            <li><span style="color:white;background-color:rgb(214, 39, 40);opacity:0.5;font-size:20px">&nbsp;Satisfied&nbsp;</span></li>
            <li><span style="color:white;background-color:rgb(148, 103, 189);opacity:0.5;font-size:20px">&nbsp;Very satisfied&nbsp;</span></li>
            </ul>
                            '''
                st.markdown(categories, unsafe_allow_html=True)
            with c2:
                st.markdown('&nbsp;')
                goal_text = set_text_style('<b>Objective</b><br>', tag='p', color=MegafonColors.orangeDark, font_size=24,
                                           text_align='center')
                goal_text += set_text_style('Determine whether the division is statistically correct?',
                                            tag='span', font_size=24)
                st.markdown(goal_text, unsafe_allow_html=True)
                st.markdown('&nbsp;')
                task_text = set_text_style('<b>Questions</b><br>', tag='p',
                                           color=MegafonColors.orangeDark, font_size=24, text_align='center')
                task_text += set_text_style(f"""
                <ul>
                <li>{set_text_style('Do customers in the groups belong to different populations?',
                                    font_size=24)}</li>
                <li>{set_text_style('If customers do not belong to the same population, '
                                    'what metrics do they differ in?', font_size=24)}</li>
                </ul>
                """, tag='span', font_size=24)
                st.markdown(task_text, unsafe_allow_html=True)
        with tab2:
            @st.cache_resource
            def load_6_3():
                with open('data/6_3.dmp', 'rb') as fp:
                    data = load(fp)
                return data


            @st.cache_resource
            def ci_plot_6_3():
                return plot_metric_confidence_interval(ci, metrics=research_metrics,
                                                       title='', height=300, n_cols=3,
                                                       horizontal_spacing=0.04, vertical_spacing=0.07,
                                                       labels_font_size=16, axes_tickfont_size=14, units_font_size=14)


            research_data, groups, group_pairs, ci, ci_overlapping, ci_center = load_6_3()

            # st.markdown('&nbsp;')
            st.markdown(set_text_style('<b>Confidence intervals of statistics</b>', text_align='center',
                                       font_size=24, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            fig = ci_plot_6_3()
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            s = display_confidence_interval(ci, metrics=research_metrics,
                                            caption='', caption_font_size=12,
                                            opacity=0.5, precision=1)
            st.markdown(s.to_html(table_uuid="table_categories_dist_6_3"), unsafe_allow_html=True)
        with tab3:
            @st.cache_resource
            def load_6_4_1():
                with open('data/6_4_1.dmp', 'rb') as fp:
                    data = load(fp)
                return data


            @st.cache_resource
            def plot_metric_histograms_6_4_1(null_distributions, statistics, research_metrics, mark_statistic):
                return plot_metric_histograms(
                    null_distributions, statistic=statistics, metrics=research_metrics,
                    # title=f'<b>Density of the null probability distribution of the test statistic</b>', title_y=0.9,
                    labels_font_size=16, axes_tickfont_size=14, units_font_size=14,
                    height=300, n_cols=3, opacity=0.5,
                    histnorm='probability density',
                    add_kde=True, add_statistic=True, mark_statistic=mark_statistic,
                    horizontal_spacing=0.08, vertical_spacing=0.07)


            @st.cache_resource
            def load_6_4_2():
                with open('data/6_4_2.dmp', 'rb') as fp:
                    data = load(fp)
                return data


            @st.cache_resource
            def plot_metric_histograms_6_4_2(null_distributions, statistics, research_metrics, mark_statistic):
                return plot_metric_histograms(
                    null_distributions, statistic=statistics, metrics=research_metrics,
                    # title=f'<b>Density of the null probability distribution of the test statistic</b>', title_y=0.9,
                    labels_font_size=16, axes_tickfont_size=14, units_font_size=14,
                    height=300, n_cols=3, opacity=0.5,
                    histnorm='probability density',
                    add_kde=True, add_statistic=True, mark_statistic=mark_statistic,
                    horizontal_spacing=0.08, vertical_spacing=0.07)


            @st.cache_resource
            def load_6_4_3():
                with open('data/6_4_3.dmp', 'rb') as fp:
                    data = load(fp)
                return data


            @st.cache_resource
            def plot_metric_histograms_6_4_3(null_distributions, statistics, research_metrics, mark_statistic):
                return plot_metric_histograms(
                    null_distributions, statistic=statistics, metrics=research_metrics,
                    # title=f'<b>Density of the null probability distribution of the test statistic</b>', title_y=0.9,
                    labels_font_size=16, axes_tickfont_size=14, units_font_size=14,
                    height=300, n_cols=3, opacity=0.5,
                    histnorm='probability density',
                    add_kde=True, add_statistic=True, mark_statistic=mark_statistic,
                    horizontal_spacing=0.08, vertical_spacing=0.07)


            @st.cache_resource
            def load_6_4_4():
                with open('data/6_4_4.dmp', 'rb') as fp:
                    data = load(fp)
                return data


            @st.cache_resource
            def plot_metric_histograms_6_4_4(null_distributions, statistics, research_metrics, mark_statistic):
                return plot_metric_histograms(
                    null_distributions, statistic=statistics, metrics=research_metrics,
                    # title=f'<b>Density of the null probability distribution of the test statistic</b>', title_y=0.9,
                    labels_font_size=16, axes_tickfont_size=14, units_font_size=14,
                    height=300, n_cols=3, opacity=0.5,
                    histnorm='probability density',
                    add_kde=True, add_statistic=True, mark_statistic=mark_statistic,
                    horizontal_spacing=0.08, vertical_spacing=0.07)


            alternatives, pvalues, mark_statistic, null_distributions, statistics = load_6_4_1()

            st.markdown(set_text_style(f'<b>Groups "{groups[0]}" и "{groups[1]}"</b>', text_align='center',
                                       font_size=24, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('<b>Hypothesis</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([40, 30, 30])
            st.markdown('&nbsp;')
            c1.latex(r'\Delta \hat{TM}=\hat{TM}_{Very unsatisfied}-\hat{TM}_{Unsatisfied}')
            c2.latex(r'H_0:\Delta \hat{TM}≥0')
            c3.latex(r'H_1:\Delta \hat{TM}<0')
            st.markdown(set_text_style('<b>Density of the null probability distribution of the test statistic</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            fig = plot_metric_histograms_6_4_1(null_distributions, statistics, research_metrics, mark_statistic)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            group_pair_index = 0
            st.markdown(set_text_style(f'<b>Statistical test p-values</b>', font_size=24,
                                       color=MegafonColors.brandGreenDarken10, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                                metrics=research_metrics, alpha=alpha,
                                caption='', col_width=400)
            st.markdown(s.to_html(table_uuid="table_pvalues_6_4_1"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Outcomes</b>', tag='p',
                                             color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'есть значимые различия</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)

            _, pvalues, mark_statistic, null_distributions, statistics = load_6_4_2()

            st.markdown('---')
            st.markdown(set_text_style(f'<b>Groups "{groups[1]}" и "{groups[2]}"</b>', text_align='center',
                                       font_size=24, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('<b>Hypothesis</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([40, 30, 30])
            st.markdown('&nbsp;')
            c1.latex(r'\Delta \hat{TM}=\hat{TM}_{Unsatisfied}-\hat{TM}_{Neutral}')
            c2.latex(r'H_0:\Delta \hat{TM}≥0')
            c3.latex(r'H_1:\Delta \hat{TM}<0')
            st.markdown(set_text_style('<b>Density of the null probability distribution of the test statistic</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            fig = plot_metric_histograms_6_4_2(null_distributions, statistics, research_metrics, mark_statistic)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            group_pair_index = 1
            st.markdown(set_text_style(f'<b>Statistical test p-values</b>', font_size=24,
                                       color=MegafonColors.brandGreenDarken10, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                                metrics=research_metrics, alpha=alpha,
                                caption='', col_width=400)
            st.markdown(s.to_html(table_uuid="table_pvalues_6_4_2"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Outcomes</b><br>', tag='p', font_size=24,
                                             color=MegafonColors.brandGreenDarken10, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'нет значимых различий</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)

            _, pvalues, mark_statistic, null_distributions, statistics = load_6_4_3()

            st.markdown('---')
            st.markdown(set_text_style(f'<b>Groups "{groups[2]}" и "{groups[3]}"</b>', text_align='center',
                                       font_size=24, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('<b>Hypothesis</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([40, 30, 30])
            st.markdown('&nbsp;')
            c1.latex(r'\Delta \hat{TM}=\hat{TM}_{Neutral}-\hat{TM}_{Satisfied}')
            c2.latex(r'H_0:\Delta \hat{TM}≥0')
            c3.latex(r'H_1:\Delta \hat{TM}<0')
            st.markdown(set_text_style('<b>Density of the null probability distribution of the test statistic</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            fig = plot_metric_histograms_6_4_3(null_distributions, statistics, research_metrics, mark_statistic)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            group_pair_index = 2
            st.markdown(set_text_style(f'<b>Statistical test p-values</b>', font_size=24,
                                       color=MegafonColors.brandGreenDarken10, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                                metrics=research_metrics, alpha=alpha,
                                caption='', col_width=400)
            st.markdown(s.to_html(table_uuid="table_pvalues_6_4_3"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Outcomes</b>', tag='p',
                                             color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'есть значимые различия</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)

            _, pvalues, mark_statistic, null_distributions, statistics = load_6_4_4()

            st.markdown('---')
            st.markdown(set_text_style(f'<b>Groups "{groups[3]}" и "{groups[0]}"</b>', text_align='center',
                                       font_size=24, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('<b>Hypothesis</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([40, 30, 30])
            st.markdown('&nbsp;')
            c1.latex(r'\Delta \hat{TM}=\hat{TM}_{Satisfied}-\hat{TM}_{Very satisfied}')
            c2.latex(r'H_0:\Delta \hat{TM}≥0')
            c3.latex(r'H_1:\Delta \hat{TM}<0')
            st.markdown(set_text_style('<b>Density of the null probability distribution of the test statistic</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            fig = plot_metric_histograms_6_4_4(null_distributions, statistics, research_metrics, mark_statistic)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            group_pair_index = 2
            st.markdown(set_text_style(f'<b>Statistical test p-values</b>', font_size=24,
                                       color=MegafonColors.brandGreenDarken10, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                                metrics=research_metrics, alpha=alpha,
                                caption='', col_width=400)
            st.markdown(s.to_html(table_uuid="table_pvalues_6_4_4"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Outcomes</b>', tag='p',
                                             color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'есть значимые различия</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)
        with tab4:
            st.markdown(set_text_style(f'<b>Statistical tests p-values</b>', font_size=24,
                                       color=MegafonColors.brandPurple, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues, metrics=research_metrics, alpha=alpha,
                                caption='', opacity=0.5, index_width=180, col_width=300)
            st.markdown(s.to_html(table_uuid="table_pvalues_6_5"), unsafe_allow_html=True)
            st.markdown('---')
            # st.markdown('---')
            # st.markdown(set_text_style(f'<b>Категории по причинам недовольства сервисом мобильного интернета</b>', font_size=24,
            #                            color=MegafonColors.brandPurple, text_align='center'),
            #             unsafe_allow_html=True)
            # st.markdown('&nbsp;')
            c1, c2 = st.columns(2, gap='medium')
            with c1:
                s = display_cat_info(data_clean)\
                    .set_properties(pd.IndexSlice['Very unsatisfied': 'Unsatisfied', 'Internet and Video':'Video'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[0], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Neutral': 'Satisfied', 'Internet and Video':'Video'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[1], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Very satisfied', 'No'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[2], opacity=0.5)
                st.markdown(s.to_html(table_uuid="table_categories_6_5"), unsafe_allow_html=True)
                categories = '''
                <br>
                <ul>
                <li><span style="color:white;background-color:rgb(31, 119, 180);opacity:0.5;font-size:20px">&nbsp;Неудовлетворительно&nbsp;</span>&nbsp</li>
                <li><span style="color:white;background-color:rgb(255, 127, 14);opacity:0.5;font-size:20px">&nbsp;Удовлетворительно&nbsp;</span>&nbsp</li>
                <li><span style="color:white;background-color:rgb(44, 160, 44);opacity:0.5;font-size:20px">&nbsp;Very satisfied&nbsp;</span>&nbsp</li>
                </ul>
                '''
                st.markdown(categories, unsafe_allow_html=True)
            with c2:
                st.markdown(set_text_style(f'<b>Outcomes</b>', font_size=24,
                                           color=MegafonColors.orangeDark, text_align='center'),
                            unsafe_allow_html=True)
                conclusion_text = '''
                <ul>
                <li><span style="font-size:24px">Customers of groups "<b>Very unsatisfied</b>" и "<b>Unsatisfied</b>" 
                belong to <b>the same population</b>.</span></li>
                <li><span style="font-size:24px">Customers of groups "<b>Neutral</b>" и "<b>Satisfied</b>" 
                belong to <b>the same population</b>.</span></li>
                <li><span style="font-size:24px">The "<b>Video Streaming Download Throughput</b>" has the strongest influence, 
                and the <b>Downlink Throughput</b> has the weakest one.</span></li>
                </ul>
                '''
                st.markdown(conclusion_text, unsafe_allow_html=True)
    case "CSAT of mobile Internet service":
        tabs = ['Objective of research',
                'Exploratory analysis',
                'Statistical tests',
                'Conclusion']
        tab1, tab2, tab3, tab4 = st.tabs(tabs)
        for tab in tabs:
            set_widget_style(tab, font_size=24)
        with tab1:
            c1, c2 = st.columns(2, gap='medium')
            with c1:
                st.markdown('&nbsp;')
                s = display_cat_info(data_clean)\
                    .set_properties(pd.IndexSlice['Very unsatisfied': 'Unsatisfied', 'Internet and Video'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[0], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Very unsatisfied': 'Unsatisfied', 'Internet': 'Video'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[1], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Neutral': 'Satisfied', 'Internet and Video'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[2], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Neutral': 'Satisfied', 'Internet': 'Video'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[3], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Very satisfied', 'No'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[4], opacity=0.5)
                st.markdown(s.to_html(table_uuid="table_categories_dist"), unsafe_allow_html=True)
                categories = '''
            <br>
            <ul>
            <li><span style="color:white;background-color:rgb(31, 119, 180);opacity:0.5;font-size:20px">&nbsp;1&nbsp;</span> - Very unsatisfied</li>
            <li><span style="color:white;background-color:rgb(255, 127, 14);opacity:0.5;font-size:20px">&nbsp;2&nbsp;</span> - Partially unsatisfied</li>
            <li><span style="color:white;background-color:rgb(44, 160, 44);opacity:0.5;font-size:20px">&nbsp;3&nbsp;</span> - Neutral</li>
            <li><span style="color:white;background-color:rgb(214, 39, 40);opacity:0.5;font-size:20px">&nbsp;4&nbsp;</span> - Partially satisfied</li>
            <li><span style="color:white;background-color:rgb(148, 103, 189);opacity:0.5;font-size:20px">&nbsp;5&nbsp;</span> - Very satisfied</li>
            </ul>
                            '''
                st.markdown(categories, unsafe_allow_html=True)
            with c2:
                st.markdown('&nbsp;')
                goal_text = set_text_style('<b>Objective</b><br>', tag='p', color=MegafonColors.orangeDark, font_size=24,
                                           text_align='center')
                goal_text += set_text_style('Determine whether the division is statistically correct?',
                                            tag='span', font_size=24)
                st.markdown(goal_text, unsafe_allow_html=True)
                st.markdown('&nbsp;')
                task_text = set_text_style('<b>Questions</b><br>', tag='p',
                                           color=MegafonColors.orangeDark, font_size=24, text_align='center')
                task_text += set_text_style(f"""
                <ul>
                <li>{set_text_style('Do customers in the groups belong to different populations?',
                                    font_size=24)}</li>
                <li>{set_text_style('If customers do not belong to the same population, '
                                    'what metrics do they differ in?', font_size=24)}</li>
                </ul>
                """, tag='span', font_size=24)
                st.markdown(task_text, unsafe_allow_html=True)
        with tab2:
            @st.cache_resource
            def load_7_3():
                with open('data/7_3.dmp', 'rb') as fp:
                    data = load(fp)
                return data


            @st.cache_resource
            def ci_plot_7_3():
                return plot_metric_confidence_interval(ci, metrics=research_metrics,
                                                       title='', height=300, n_cols=3,
                                                       horizontal_spacing=0.04, vertical_spacing=0.07,
                                                       labels_font_size=16, axes_tickfont_size=14, units_font_size=14)


            research_data, groups, group_pairs, ci, ci_overlapping, ci_center = load_7_3()

            # st.markdown('&nbsp;')
            st.markdown(set_text_style('<b>Confidence intervals of statistics</b>', text_align='center',
                                       font_size=24, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            fig = ci_plot_7_3()
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            s = display_confidence_interval(ci, metrics=research_metrics,
                                            caption='', caption_font_size=12,
                                            opacity=0.5, precision=1)
            st.markdown(s.to_html(table_uuid="table_categories_dist_7_3"), unsafe_allow_html=True)
        with tab3:
            @st.cache_resource
            def load_7_4_1():
                with open('data/7_4_1.dmp', 'rb') as fp:
                    data = load(fp)
                return data


            @st.cache_resource
            def plot_metric_histograms_7_4_1(null_distributions, statistics, research_metrics, mark_statistic):
                return plot_metric_histograms(
                    null_distributions, statistic=statistics, metrics=research_metrics,
                    # title=f'<b>Density of the null probability distribution of the test statistic</b>', title_y=0.9,
                    labels_font_size=16, axes_tickfont_size=14, units_font_size=14,
                    height=300, n_cols=3, opacity=0.5,
                    histnorm='probability density',
                    add_kde=True, add_statistic=True, mark_statistic=mark_statistic,
                    horizontal_spacing=0.08, vertical_spacing=0.07)


            @st.cache_resource
            def load_7_4_2():
                with open('data/7_4_2.dmp', 'rb') as fp:
                    data = load(fp)
                return data


            @st.cache_resource
            def plot_metric_histograms_7_4_2(null_distributions, statistics, research_metrics, mark_statistic):
                return plot_metric_histograms(
                    null_distributions, statistic=statistics, metrics=research_metrics,
                    # title=f'<b>Density of the null probability distribution of the test statistic</b>', title_y=0.9,
                    labels_font_size=16, axes_tickfont_size=14, units_font_size=14,
                    height=300, n_cols=3, opacity=0.5,
                    histnorm='probability density',
                    add_kde=True, add_statistic=True, mark_statistic=mark_statistic,
                    horizontal_spacing=0.08, vertical_spacing=0.07)


            @st.cache_resource
            def load_7_4_3():
                with open('data/7_4_3.dmp', 'rb') as fp:
                    data = load(fp)
                return data


            @st.cache_resource
            def plot_metric_histograms_7_4_3(null_distributions, statistics, research_metrics, mark_statistic):
                return plot_metric_histograms(
                    null_distributions, statistic=statistics, metrics=research_metrics,
                    # title=f'<b>Density of the null probability distribution of the test statistic</b>', title_y=0.9,
                    labels_font_size=16, axes_tickfont_size=14, units_font_size=14,
                    height=300, n_cols=3, opacity=0.5,
                    histnorm='probability density',
                    add_kde=True, add_statistic=True, mark_statistic=mark_statistic,
                    horizontal_spacing=0.08, vertical_spacing=0.07)


            @st.cache_resource
            def load_7_4_4():
                with open('data/7_4_4.dmp', 'rb') as fp:
                    data = load(fp)
                return data


            @st.cache_resource
            def plot_metric_histograms_7_4_4(null_distributions, statistics, research_metrics, mark_statistic):
                return plot_metric_histograms(
                    null_distributions, statistic=statistics, metrics=research_metrics,
                    # title=f'<b>Density of the null probability distribution of the test statistic</b>', title_y=0.9,
                    labels_font_size=16, axes_tickfont_size=14, units_font_size=14,
                    height=300, n_cols=3, opacity=0.5,
                    histnorm='probability density',
                    add_kde=True, add_statistic=True, mark_statistic=mark_statistic,
                    horizontal_spacing=0.08, vertical_spacing=0.07)


            alternatives, pvalues, mark_statistic, null_distributions, statistics = load_7_4_1()

            st.markdown(set_text_style(f'<b>Groups "{groups[0]}" и "{groups[1]}"</b>', text_align='center',
                                       font_size=24, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('<b>Hypothesis</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([40, 30, 30])
            st.markdown('&nbsp;')
            c1.latex(r'\Delta \hat{TM}=\hat{TM}_{1}-\hat{TM}_{2}')
            c2.latex(r'H_0:\Delta \hat{TM}≥0')
            c3.latex(r'H_1:\Delta \hat{TM}<0')
            st.markdown(set_text_style('<b>Density of the null probability distribution of the test statistic</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            fig = plot_metric_histograms_7_4_1(null_distributions, statistics, research_metrics, mark_statistic)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            group_pair_index = 0
            st.markdown(set_text_style(f'<b>Statistical test p-values</b>', font_size=24,
                                       color=MegafonColors.brandGreenDarken10, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                                metrics=research_metrics, alpha=alpha,
                                caption='', col_width=400)
            st.markdown(s.to_html(table_uuid="table_pvalues_7_4_1"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Outcomes</b>', tag='p',
                                             color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'есть значимые различия</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)

            _, pvalues, mark_statistic, null_distributions, statistics = load_7_4_2()

            st.markdown('---')
            st.markdown(set_text_style(f'<b>Groups "{groups[1]}" и "{groups[2]}"</b>', text_align='center',
                                       font_size=24, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('<b>Hypothesis</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([40, 30, 30])
            st.markdown('&nbsp;')
            c1.latex(r'\Delta \hat{TM}=\hat{TM}_{2}-\hat{TM}_{3}')
            c2.latex(r'H_0:\Delta \hat{TM}≥0')
            c3.latex(r'H_1:\Delta \hat{TM}<0')
            st.markdown(set_text_style('<b>Density of the null probability distribution of the test statistic</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            fig = plot_metric_histograms_7_4_2(null_distributions, statistics, research_metrics, mark_statistic)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            group_pair_index = 1
            st.markdown(set_text_style(f'<b>Statistical test p-values</b>', font_size=24,
                                       color=MegafonColors.brandGreenDarken10, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                                metrics=research_metrics, alpha=alpha,
                                caption='', col_width=400)
            st.markdown(s.to_html(table_uuid="table_pvalues_7_4_2"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Outcomes</b><br>', tag='p', font_size=24,
                                             color=MegafonColors.brandGreenDarken10, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'нет значимых различий</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)

            _, pvalues, mark_statistic, null_distributions, statistics = load_7_4_3()

            st.markdown('---')
            st.markdown(set_text_style(f'<b>Groups "{groups[2]}" и "{groups[3]}"</b>', text_align='center',
                                       font_size=24, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('<b>Hypothesis</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([40, 30, 30])
            st.markdown('&nbsp;')
            c1.latex(r'\Delta \hat{TM}=\hat{TM}_{3}-\hat{TM}_{4}')
            c2.latex(r'H_0:\Delta \hat{TM}≥0')
            c3.latex(r'H_1:\Delta \hat{TM}<0')
            st.markdown(set_text_style('<b>Density of the null probability distribution of the test statistic</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            fig = plot_metric_histograms_7_4_3(null_distributions, statistics, research_metrics, mark_statistic)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            group_pair_index = 2
            st.markdown(set_text_style(f'<b>Statistical test p-values</b>', font_size=24,
                                       color=MegafonColors.brandGreenDarken10, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                                metrics=research_metrics, alpha=alpha,
                                caption='', col_width=400)
            st.markdown(s.to_html(table_uuid="table_pvalues_7_4_3"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Outcomes</b>', tag='p',
                                             color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'есть значимые различия</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)

            _, pvalues, mark_statistic, null_distributions, statistics = load_7_4_4()

            st.markdown('---')
            st.markdown(set_text_style(f'<b>Groups "{groups[3]}" и "{groups[0]}"</b>', text_align='center',
                                       font_size=24, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('<b>Hypothesis</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([40, 30, 30])
            st.markdown('&nbsp;')
            c1.latex(r'\Delta \hat{TM}=\hat{TM}_{4}-\hat{TM}_{5}')
            c2.latex(r'H_0:\Delta \hat{TM}≥0')
            c3.latex(r'H_1:\Delta \hat{TM}<0')
            st.markdown(set_text_style('<b>Density of the null probability distribution of the test statistic</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            fig = plot_metric_histograms_7_4_4(null_distributions, statistics, research_metrics, mark_statistic)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            group_pair_index = 2
            st.markdown(set_text_style(f'<b>Statistical test p-values</b>', font_size=24,
                                       color=MegafonColors.brandGreenDarken10, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                                metrics=research_metrics, alpha=alpha,
                                caption='', col_width=400)
            st.markdown(s.to_html(table_uuid="table_pvalues_7_4_4"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Outcomes</b>', tag='p',
                                             color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'есть значимые различия</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)
        with tab4:
            st.markdown(set_text_style(f'<b>Statistical tests p-values</b>', font_size=24,
                                       color=MegafonColors.brandPurple, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues, metrics=research_metrics, alpha=alpha,
                                caption='', opacity=0.5, index_width=30, col_width=300)
            st.markdown(s.to_html(table_uuid="table_pvalues_7_5"), unsafe_allow_html=True)
            st.markdown('---')
            # st.markdown('---')
            # st.markdown(set_text_style(f'<b>Категории по причинам недовольства сервисом мобильного интернета</b>', font_size=24,
            #                            color=MegafonColors.brandPurple, text_align='center'),
            #             unsafe_allow_html=True)
            # st.markdown('&nbsp;')
            c1, c2 = st.columns(2, gap='medium')
            with c1:
                s = display_cat_info(data_clean)\
                    .set_properties(pd.IndexSlice['Very unsatisfied': 'Unsatisfied', 'Internet and Video'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[0], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Very unsatisfied': 'Unsatisfied', 'Internet': 'Video'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[1], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Neutral': 'Satisfied', 'Internet and Video'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[2], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Neutral': 'Satisfied', 'Internet': 'Video'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[2], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Very satisfied', 'No'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[3], opacity=0.5)
                st.markdown(s.to_html(table_uuid="table_categories_7_5"), unsafe_allow_html=True)
                categories = '''
                <br>
                <ul>
                <li><span style="color:white;background-color:rgb(31, 119, 180);opacity:0.5;font-size:20px">&nbsp;1&nbsp;</span>&nbsp- Very unsatisfied</li>
                <li><span style="color:white;background-color:rgb(255, 127, 14);opacity:0.5;font-size:20px">&nbsp;2&nbsp;</span>&nbsp- Partially unsatisfied</li>
                <li><span style="color:white;background-color:rgb(44, 160, 44);opacity:0.5;font-size:20px">&nbsp;3&nbsp;</span> - Partially satisfied</li>
                <li><span style="color:white;background-color:rgb(214, 39, 40);opacity:0.5;font-size:20px">&nbsp;4&nbsp;</span>&nbsp- Very satisfied</li>
                </ul>
                '''
                st.markdown(categories, unsafe_allow_html=True)
            with c2:
                st.markdown(set_text_style(f'<b>Outcomes</b>', font_size=24,
                                           color=MegafonColors.orangeDark, text_align='center'),
                            unsafe_allow_html=True)
                conclusion_text = '''
                <ul>
                <li><span style="font-size:24px">Customers of groups "<b>1</b>" и "<b>2</b>" and also of groups"<b>4</b>" и "<b>5</b>" belong to <b>the same population</b>.</span></li>
                <li><span style="font-size:24px">Customers of groups "<b>2</b>" и "<b>3</b>" belong to <b>the same population</b>.</span></li>
                <li><span style="font-size:24px">Самое сильное влияние у метрики "<b>Video Streaming Download Throughput</b>", а самое слабое у метрики <b>Downlink Throughput</b>.</span></li>
                </ul>
                '''
                st.markdown(conclusion_text, unsafe_allow_html=True)
    case "Influence of the metrics on the CSAT of mobile Internet service":
        tabs = ['Objective of research',
                'Exploratory analysis',
                'Statistical tests',
                'Conclusion']
        tab1, tab2, tab3, tab4 = st.tabs(tabs)
        for tab in tabs:
            set_widget_style(tab, font_size=24)
        with tab1:
            c1, c2 = st.columns(2, gap='medium')
            with c1:
                st.markdown('&nbsp;')
                s = display_cat_info(data_clean)\
                    .set_properties(pd.IndexSlice['Very unsatisfied': 'Unsatisfied', 'Internet and Video'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[0], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Very unsatisfied': 'Unsatisfied', 'Internet': 'Video'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[1], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Neutral': 'Satisfied', 'Internet and Video'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[2], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Neutral': 'Satisfied', 'Internet': 'Video'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[2], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Very satisfied', 'No'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[3], opacity=0.5)
                st.markdown(s.to_html(table_uuid="table_categories_dist"), unsafe_allow_html=True)
                categories = '''
            <br>
            <ul>
            <li><span style="color:white;background-color:rgb(31, 119, 180);opacity:0.5;font-size:20px">&nbsp;1&nbsp;</span> - Very unsatisfied</li>
            <li><span style="color:white;background-color:rgb(255, 127, 14);opacity:0.5;font-size:20px">&nbsp;2&nbsp;</span> - Partially unsatisfied</li>
            <li><span style="color:white;background-color:rgb(44, 160, 44);opacity:0.5;font-size:20px">&nbsp;3&nbsp;</span> - Partially satisfied</li>
            <li><span style="color:white;background-color:rgb(214, 39, 40);opacity:0.5;font-size:20px">&nbsp;4&nbsp;</span> - Very satisfied</li>
            </ul>
                            '''
                st.markdown(categories, unsafe_allow_html=True)
            with c2:
                st.markdown('&nbsp;')
                goal_text = set_text_style('<b>Objective</b><br>', tag='p', color=MegafonColors.orangeDark, font_size=24,
                                           text_align='center')
                goal_text += set_text_style('Determine whether the division is statistically correct?',
                                            tag='span', font_size=24)
                st.markdown(goal_text, unsafe_allow_html=True)
                st.markdown('&nbsp;')
                task_text = set_text_style('<b>Questions</b><br>', tag='p',
                                           color=MegafonColors.orangeDark, font_size=24, text_align='center')
                task_text += set_text_style(f"""
                <ul>
                <li>{set_text_style('Do customers in the groups belong to different populations?',
                                    font_size=24)}</li>
                <li>{set_text_style('If customers do not belong to the same population, '
                                    'what metrics do they differ in?', font_size=24)}</li>
                </ul>
                """, tag='span', font_size=24)
                st.markdown(task_text, unsafe_allow_html=True)
        with tab2:
            @st.cache_resource
            def load_8_3():
                with open('data/8_3.dmp', 'rb') as fp:
                    data = load(fp)
                return data


            @st.cache_resource
            def ci_plot_8_3():
                return plot_metric_confidence_interval(ci, metrics=research_metrics,
                                                       title='', height=300, n_cols=3,
                                                       horizontal_spacing=0.04, vertical_spacing=0.07,
                                                       labels_font_size=16, axes_tickfont_size=14, units_font_size=14)


            research_data, groups, group_pairs, ci, ci_overlapping, ci_center = load_8_3()

            # st.markdown('&nbsp;')
            st.markdown(set_text_style('<b>Confidence intervals of statistics</b>', text_align='center',
                                       font_size=24, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            fig = ci_plot_8_3()
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            s = display_confidence_interval(ci, metrics=research_metrics,
                                            caption='', caption_font_size=12,
                                            opacity=0.5, precision=1)
            st.markdown(s.to_html(table_uuid="table_categories_dist_8_3"), unsafe_allow_html=True)
        with tab3:
            @st.cache_resource
            def load_8_4_1():
                with open('data/8_4_1.dmp', 'rb') as fp:
                    data = load(fp)
                return data


            @st.cache_resource
            def plot_metric_histograms_8_4_1(null_distributions, statistics, research_metrics, mark_statistic):
                return plot_metric_histograms(
                    null_distributions, statistic=statistics, metrics=research_metrics,
                    # title=f'<b>Density of the null probability distribution of the test statistic</b>', title_y=0.9,
                    labels_font_size=16, axes_tickfont_size=14, units_font_size=14,
                    height=300, n_cols=3, opacity=0.5,
                    histnorm='probability density',
                    add_kde=True, add_statistic=True, mark_statistic=mark_statistic,
                    horizontal_spacing=0.08, vertical_spacing=0.07)


            @st.cache_resource
            def load_8_4_2():
                with open('data/8_4_2.dmp', 'rb') as fp:
                    data = load(fp)
                return data


            @st.cache_resource
            def plot_metric_histograms_8_4_2(null_distributions, statistics, research_metrics, mark_statistic):
                return plot_metric_histograms(
                    null_distributions, statistic=statistics, metrics=research_metrics,
                    # title=f'<b>Density of the null probability distribution of the test statistic</b>', title_y=0.9,
                    labels_font_size=16, axes_tickfont_size=14, units_font_size=14,
                    height=300, n_cols=3, opacity=0.5,
                    histnorm='probability density',
                    add_kde=True, add_statistic=True, mark_statistic=mark_statistic,
                    horizontal_spacing=0.08, vertical_spacing=0.07)


            @st.cache_resource
            def load_8_4_3():
                with open('data/8_4_3.dmp', 'rb') as fp:
                    data = load(fp)
                return data


            @st.cache_resource
            def plot_metric_histograms_8_4_3(null_distributions, statistics, research_metrics, mark_statistic):
                return plot_metric_histograms(
                    null_distributions, statistic=statistics, metrics=research_metrics,
                    # title=f'<b>Density of the null probability distribution of the test statistic</b>', title_y=0.9,
                    labels_font_size=16, axes_tickfont_size=14, units_font_size=14,
                    height=300, n_cols=3, opacity=0.5,
                    histnorm='probability density',
                    add_kde=True, add_statistic=True, mark_statistic=mark_statistic,
                    horizontal_spacing=0.08, vertical_spacing=0.07)


            alternatives, pvalues, mark_statistic, null_distributions, statistics = load_8_4_1()

            st.markdown(set_text_style(f'<b>Groups "{groups[0]}" и "{groups[1]}"</b>', text_align='center',
                                       font_size=24, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('<b>Hypothesis</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([40, 30, 30])
            st.markdown('&nbsp;')
            c1.latex(r'\Delta \hat{TM}=\hat{TM}_{1}-\hat{TM}_{2}')
            c2.latex(r'H_0:\Delta \hat{TM}≥0')
            c3.latex(r'H_1:\Delta \hat{TM}<0')
            st.markdown(set_text_style('<b>Density of the null probability distribution of the test statistic</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            fig = plot_metric_histograms_8_4_1(null_distributions, statistics, research_metrics, mark_statistic)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            group_pair_index = 0
            st.markdown(set_text_style(f'<b>Statistical test p-values</b>', font_size=24,
                                       color=MegafonColors.brandGreenDarken10, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                                metrics=research_metrics, alpha=alpha,
                                caption='', col_width=400)
            st.markdown(s.to_html(table_uuid="table_pvalues_8_4_1"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Outcomes</b>', tag='p',
                                             color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'есть значимые различия</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)

            _, pvalues, mark_statistic, null_distributions, statistics = load_8_4_2()

            st.markdown('---')
            st.markdown(set_text_style(f'<b>Groups "{groups[1]}" и "{groups[2]}"</b>', text_align='center',
                                       font_size=24, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('<b>Hypothesis</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([40, 30, 30])
            st.markdown('&nbsp;')
            c1.latex(r'\Delta \hat{TM}=\hat{TM}_{2}-\hat{TM}_{3}')
            c2.latex(r'H_0:\Delta \hat{TM}≥0')
            c3.latex(r'H_1:\Delta \hat{TM}<0')
            st.markdown(set_text_style('<b>Density of the null probability distribution of the test statistic</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            fig = plot_metric_histograms_8_4_2(null_distributions, statistics, research_metrics, mark_statistic)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            group_pair_index = 1
            st.markdown(set_text_style(f'<b>Statistical test p-values</b>', font_size=24,
                                       color=MegafonColors.brandGreenDarken10, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                                metrics=research_metrics, alpha=alpha,
                                caption='', col_width=400)
            st.markdown(s.to_html(table_uuid="table_pvalues_8_4_2"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Outcomes</b><br>', tag='p', font_size=24,
                                             color=MegafonColors.brandGreenDarken10, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'нет значимых различий</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)

            _, pvalues, mark_statistic, null_distributions, statistics = load_8_4_3()

            st.markdown('---')
            st.markdown(set_text_style(f'<b>Groups "{groups[2]}" и "{groups[3]}"</b>', text_align='center',
                                       font_size=24, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('<b>Hypothesis</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([40, 30, 30])
            st.markdown('&nbsp;')
            c1.latex(r'\Delta \hat{TM}=\hat{TM}_{3}-\hat{TM}_{4}')
            c2.latex(r'H_0:\Delta \hat{TM}≥0')
            c3.latex(r'H_1:\Delta \hat{TM}<0')
            st.markdown(set_text_style('<b>Density of the null probability distribution of the test statistic</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            fig = plot_metric_histograms_8_4_3(null_distributions, statistics, research_metrics, mark_statistic)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            group_pair_index = 2
            st.markdown(set_text_style(f'<b>Statistical test p-values</b>', font_size=24,
                                       color=MegafonColors.brandGreenDarken10, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                                metrics=research_metrics, alpha=alpha,
                                caption='', col_width=400)
            st.markdown(s.to_html(table_uuid="table_pvalues_8_4_3"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Outcomes</b>', tag='p',
                                             color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'есть значимые различия</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Outcomes</b>', tag='p',
                                             color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'есть значимые различия</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)
        with tab4:
            st.markdown(set_text_style(f'<b>Statistical tests p-values</b>', font_size=24,
                                       color=MegafonColors.brandPurple, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues, metrics=research_metrics, alpha=alpha,
                                caption = '', opacity=0.5, col_width=180, index_width=30)
            st.markdown(s.to_html(table_uuid="table_pvalues_8_5"), unsafe_allow_html=True)
            st.markdown('---')
            # st.markdown(set_text_style(f'<b>Категории по причинам недовольства сервисом мобильного интернета</b>', font_size=24,
            #                            color=MegafonColors.brandPurple, text_align='center'),
            #             unsafe_allow_html=True)
            # st.markdown('&nbsp;')
            c1, c2 = st.columns(2, gap='medium')
            with c1:
                st.markdown(set_text_style(f'<b>Trend of <br>"'
                                           f'{research_metrics.loc["Video Streaming Download Throughput(Kbps)", "name"]}"</b>',
                                           font_size=24, text_align='center', color=MegafonColors.brandPurple),
                            unsafe_allow_html=True)
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
                fig.update_xaxes(tickmode='array', tickvals=[1, 2, 3, 4])
                st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=False)
                results = px.get_trendline_results(fig).iloc[0, 0]
            with c2:
                st.markdown(set_text_style(f'<b>Outcomes</b>', font_size=24,
                                           color=MegafonColors.orangeDark, text_align='center'),
                            unsafe_allow_html=True)
                conclusion_text = '''
                <ul>
                <li><span style="font-size:24px">Клиенты всех групп принадлежат к <b>разным популяциям</b>.</span></li>
                <li><span style="font-size:24px">Самое сильное влияние у метрики "<b>Video Streaming Download Throughput</b>".</span></li>
                </ul>
                '''
                st.markdown(conclusion_text, unsafe_allow_html=True)
    case "Summary":
        @st.cache_resource
        def load_9():
            with open('data/8_3.dmp', 'rb') as fp:
                data = load(fp)
            return data

        research_data, groups, group_pairs, ci, ci_overlapping, ci_center = load_9()
        st.markdown(set_text_style(f'<b>Customer satisfaction scores (CSAT)<br>'
                                   f'with the mobile Internet service</b>', font_size=24, text_align='center',
                                   color=MegafonColors.brandPurple),
                    unsafe_allow_html=True)
        text = f"""
        <ul>
            <li><span style="font-size:24px"><span style="color:white;background-color:rgb(31, 119, 180);opacity:0.5">&nbsp;&nbsp;1&nbsp;&nbsp;</span>&nbsp;&nbsp;Полностью неудовлетворены</span></li>
            <li><span style="font-size:24px"><span style="color:white;background-color:rgb(255, 127, 14);opacity:0.5">&nbsp;&nbsp;2&nbsp;&nbsp;</span>&nbsp;&nbsp;Частично неудовлетворены</span></li>
            <li><span style="font-size:24px"><span style="color:white;background-color:rgb(44, 160, 44);opacity:0.5">&nbsp;&nbsp;3&nbsp;&nbsp;</span>&nbsp;&nbsp;Частично удовлетворены</span></li>
            <li><span style="font-size:24px"><span style="color:white;background-color:rgb(214, 39, 40);opacity:0.5">&nbsp;&nbsp;4&nbsp;&nbsp;</span>&nbsp;&nbsp;Полностью удовлетворены</span></li>
        </ul>
"""
        st.markdown(text, unsafe_allow_html=True)
        st.markdown('---')
        st.markdown(set_text_style(f'<b>Confidence intervals of metric<br>"'
                                   f'{research_metrics.loc["Video Streaming Download Throughput(Kbps)", "name"]}'
                                   f'"</b>', font_size=24, text_align='center', color=MegafonColors.brandPurple),
                    unsafe_allow_html=True)
        table = display_confidence_interval(
            ci['Video Streaming Download Throughput(Kbps)'],
            metrics=research_metrics.loc['Video Streaming Download Throughput(Kbps)'],
            caption='', caption_font_size=12, opacity=0.5, precision=1, index_width=30)
        s = display_confidence_interval(ci['Video Streaming Download Throughput(Kbps)'],
                                        metrics=research_metrics.loc['Video Streaming Download Throughput(Kbps)'],
                                        caption='', caption_font_size=12, opacity=0.5, precision=1, index_width=30)
        st.markdown(s.to_html(table_uuid="table_pvalues_9"), unsafe_allow_html=True)
