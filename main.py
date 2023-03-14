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
    data = data[data.Q1.str.isdecimal().fillna(False)]
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
        lambda q1: 'Ужасно' if q1 <= 2 else ('Плохо' if q1 <= 4 else (
            'Нормально' if q1 <= 6 else ('Хорошо' if q1 <= 8 else 'Отлично'))))
    # Добавляем информацию о причине неудовлетворенности сервисом мобильного интернета
    data_clean['Dissatisfaction reasons'] = data_clean['Q2'].apply(
        lambda q2: 'Интернет и видео' if q2.find('4, 5') >= 0 else (
            'Интернет' if q2.find('4') >= 0 else (
                'Видео' if q2.find('5') >= 0 else 'Нет')))
    # Сортируем данные так, чтобы причины неудовлетворенности располагались в обозначенном поряке
    data_clean.sort_values(['Q1', 'Q2'],
                           key=lambda x: x if x.name == 'Q1' else (x.str.contains('5') - x.str.contains('4, 5') * 2),
                           inplace=True)
    metrics = pd.DataFrame({
        'description': ["Объем трафика передачи данных",
                        "Средняя скорость «к абоненту»",
                        "Средняя скорость «от абонента»",
                        "Частота переотправок пакетов «к абоненту»",
                        "Скорость загрузки потокового видео",
                        "Задержка старта воспроизведения видео",
                        "Скорость загрузки web-страниц через браузер",
                        "Пинг при просмотре web-страниц"],
        'units': ["МБ", "кбит/с", "кбит/с", "%", "кбит/с", "мс", "кбит/с", "мс"],
        'impact': ['0', '+', '-', '+', '+', '-', '+', '-']
    }, index=pd.Index(data.columns.drop(['Q1', 'Q2']), name='metric'))
    metrics['label'] = metrics['description'].apply(lambda d: '<b>' + wrap_text(d, 30) + '</b>')
    return data, data_clean, metrics


@st.cache_resource
def plot_csat_dist_10(data: pd.DataFrame):
    fig = px.histogram(data, x='Q1', histnorm='percent', opacity=0.5)
    fig.update_traces(texttemplate="%{y:.1f}%", hovertemplate='%{x} - %{y:.1f}%',
                      marker_color=px.colors.DEFAULT_PLOTLY_COLORS)
    fig.update_layout(font_family="Calibri", font_color=MegafonColors.scantBlue2, font_size=14,
                      # title=None, title_x=0.5, title_y=0.91, title_xanchor='center',
                      # title_font_size=18, title_font_color=MegafonColors.scantBlue2,
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
                      # title_font_size=18, title_font_color=MegafonColors.scantBlue2,
                      height=400,
                      bargap=0.2, margin_l=0, margin_r=0, margin_b=0, margin_t=0)
    fig.update_xaxes(title='', title_font_color=MegafonColors.brandPurple, tickfont_size=14)
    fig.update_yaxes(title='', title_font_color=MegafonColors.brandPurple, tickfont_size=14)
    return fig


@st.cache_resource
def plot_reason_dist(data: pd.DataFrame):
    s = pd.Series(index=[
        "0 - Неизвестно", "1 - Недозвоны, обрывы при звонках",
        "2 - Время ожидания гудков при звонке",
        "3 - Плохое качество связи в зданиях, торговых центрах и т.п.",
        "4 - Медленный мобильный Интернет", "5 - Медленная загрузка видео",
        "6 - Затрудняюсь ответить", "7 - Свой вариант"
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
                      # title_font_size=18, title_font_color=MegafonColors.scantBlue2,
                      showlegend=False, height=450,
                      bargap=0.3, margin_l=0, margin_r=0, margin_t=0, margin_b=0)
    fig.update_xaxes(title='', title_font_color=MegafonColors.brandPurple, tickfont_size=16)
    fig.update_yaxes(title='', title_font_color=MegafonColors.brandPurple, tickfont_size=16)
    return fig


@st.cache_resource
def plot_reason_combo_dist(data: pd.DataFrame):
    s = pd.Series(dtype=float)
    s.loc["0, 6, 7 - Неизвестно"] = (data['Q2'].str.contains('[067]', regex=True) & (data['Q1'] <= 8)).sum()
    s.loc["1, 2 - Голосовая связь"] = data['Q2'].str.contains('[12]', regex=True).sum()
    s.loc["3 - Покрытие"] = data['Q2'].str.contains('3').sum()
    s.loc["4, 5 - Мобильный интернет"] = data['Q2'].str.contains('[45]', regex=True).sum()
    s = s / s.sum() * 100

    fig = px.bar(s, orientation='h', opacity=0.5)
    fig.update_traces(texttemplate="%{x:.1f}%", hovertemplate='%{y} - %{x:.1f}%',
                      marker_color=px.colors.DEFAULT_PLOTLY_COLORS)
    fig.update_layout(font_family="Calibri", font_color=MegafonColors.brandPurple, font_size=16,
                      # title='Объединенные', title_x=0.5, title_y=0.95, title_xanchor='center',
                      # title_font_size=18, title_font_color=MegafonColors.scantBlue2,
                      showlegend=False, height=250,
                      bargap=0.3, margin_l=0, margin_r=0, margin_t=0, margin_b=0)
    fig.update_xaxes(title='', title_font_color=MegafonColors.brandPurple, tickfont_size=16)
    fig.update_yaxes(title='', title_font_color=MegafonColors.brandPurple, tickfont_size=16)
    return fig


@st.cache_resource
def show_metric_table(metrics: pd.DataFrame):
    metrics = metrics.reset_index()
    # metrics.index.rename('Метрика', inplace=True)
    # style = metrics.style.set_caption('ddd').set_properties({'font-size': '16px'})
    header = f''
    table = \
        f'| {set_text_style("Метрика", tag="span", text_align="center")} ' \
        f'| {set_text_style("Описание", tag="span", text_align="center")} ' \
        f'| {set_text_style("Ед. изм.", tag="span", text_align="center")} ' \
        f'| {set_text_style("Влияние", tag="span", text_align="center")} |\n' \
        f'|---------|----------|:--------:|:-------:|\n'
    for _, (metric, description, units, impact) \
            in metrics[['metric', 'description', 'units', 'impact']].iterrows():
        if impact == '+':
            impact = set_text_style('▲', tag='span', color=MegafonColors.brandGreen)
        elif impact == '-':
            impact = set_text_style('▼', tag='span', color='red')
        else:
            impact = '─'
        table += f'|{metric}|{description}|{units}|{impact}|\n'

    return table


config.dataFrameSerialization = "arrow"

st.set_page_config(page_title='Исследование результатов опроса клиентов компании МегаФон',
                   page_icon='bar-chart', layout='wide')
# hide_menu_button()
remove_blank_space()

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
            "Вступление",
            "Подготовка данных и разведочный анализ",
            "Постановка цели",
            "Выбор метрик, статистик и критериев",
            "Причины недовольства сервисом мобильного интернета",
            "Оценки качества сервиса мобильного интернета",
            "Уровни удовлетворенности сервисом мобильного интернета",
            "Влияние метрик на уровень удовлетворенности",
            "---",
            "Заключение"
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
    case "Вступление":
        plain_text = 'Исследование результатов опроса клиентов компании МегаФон'
        font_formatted_text = f'**{set_text_style(plain_text, font_size=80, color=MegafonColors.brandPurple)}**'
        st.markdown(font_formatted_text, unsafe_allow_html=True)
        font_formatted_text = set_text_style('&nbsp;', font_size=32)
        st.markdown(font_formatted_text, unsafe_allow_html=True)
        author = set_text_style('Автор: ', font_size=48, color=MegafonColors.brandGreen, tag='span') + \
                 set_text_style('**Алексей Шерстобитов**', font_size=48, color=MegafonColors.brandGreen, tag='span')
        st.markdown(author, unsafe_allow_html=True)
    case "Подготовка данных и разведочный анализ":
        tabs = ['Ответ на 1-ый вопрос', 'Ответ на 2-ой вопрос', 'Значения метрик']
        tab1, tab2, tab3 = st.tabs(tabs)
        for tab in tabs:
            set_widget_style(tab, font_size=24)
        with tab1:
            st.markdown(
                set_text_style('<b>Распределение клиентов по уровню удовлетворенности качеством связи</b>',
                               font_size=24, text_align='center',
                               # color=MegafonColors.brandPurple
                               ),
                unsafe_allow_html=True
            )
            c1, c2 = st.columns([64, 36], gap='medium')
            with c1:
                col_title = set_text_style('<b>✘</b> ', tag='span', color='red') + '10-бальная шкала'
                col_title = set_text_style(col_title, font_size=24, text_align='center')
                st.markdown(col_title, unsafe_allow_html=True)
                fig = plot_csat_dist_10(data)
                st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            with c2:
                col_title = set_text_style('<b>✔</b> ', tag='span', color=MegafonColors.brandGreen) + '5-бальная шкала'
                col_title = set_text_style(col_title, font_size=24, text_align='center')
                st.markdown(col_title, unsafe_allow_html=True)
                fig = plot_csat_dist_5(data)
                st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
        with tab2:
            st.markdown(
                set_text_style('<b>Распределение причин снижения оценки</b>',
                               # color=MegafonColors.brandPurple,
                               font_size=24, text_align='center'),
                unsafe_allow_html=True
            )
            c1, c2 = st.columns([53, 47], gap='medium')
            with c1:
                col_title = set_text_style('<b>✘</b> ', tag='span', color='red') + 'Исходные'
                col_title = set_text_style(col_title, font_size=20, text_align='center')
                st.markdown(col_title, unsafe_allow_html=True)
                fig = plot_reason_dist(data)
                st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            with c2:
                col_title = set_text_style('<b>✔</b> ', tag='span', color=MegafonColors.brandGreen) + 'Объединённые'
                col_title = set_text_style(col_title, font_size=20, text_align='center')
                st.markdown(col_title, unsafe_allow_html=True)
                fig = plot_reason_combo_dist(data)
                st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
        with tab3:
            st.markdown(set_text_style('<b>Плотность распределения вероятностей метрик в наблюдаемой выборке</b>',
                                       font_size=24, text_align='center'),
                        unsafe_allow_html=True
                        )
            metrics_table = show_metric_table(metrics)
            st.markdown(metrics_table, unsafe_allow_html=True)
            st.markdown('---')
            st.markdown(
                set_text_style('<b>Плотность распределения вероятностей метрик в наблюдаемой выборке</b>',
                               # color=MegafonColors.brandPurple,
                               font_size=24, text_align='center'),
                unsafe_allow_html=True
            )
            fig = plot_metric_histograms(
                data[metrics.index], metrics,
                # title='<b>Плотность распределения вероятностей метрик в наблюдаемой выборке</b>',
                # title_y=0.95, title_font_size=24,
                labels_font_size=18,
                axes_tickfont_size=14,
                height=900, boxplot_height_fraq=0.15, n_cols=3, opacity=0.5,
                histnorm='probability density',
                add_boxplot=True, add_kde=True, add_mean=True,
                horizontal_spacing=0.07, vertical_spacing=0.15)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
    case "Постановка цели":
        tabs = ['Цель исследования',
                'Исходные категории',
                'Карта распределения']
        tab1, tab2, tab3 = st.tabs(tabs)
        for tab in tabs:
            set_widget_style(tab, font_size=24)
        with tab1:
            st.markdown(set_text_style('<b>Цель</b>', font_size=32, text_align='center',
                                       color=MegafonColors.orangeDark), unsafe_allow_html=True)
            st.markdown(set_text_style('Классифицировать пользователей в разрезе их отношения к качеству '
                                       'сервиса мобильного интернета', font_size=24),
                        unsafe_allow_html=True)
            st.markdown('&nbsp;')
            df = pd.DataFrame(
                index=[
                    "0 - Неизвестно", "1 - Недозвоны, обрывы при звонках",
                    "2 - Время ожидания гудков при звонке",
                    "3 - Плохое качество связи в зданиях, торговых центрах и т.п.",
                    "4 - Медленный мобильный Интернет", "5 - Медленная загрузка видео",
                    "6 - Затрудняюсь ответить", "7 - Свой вариант"
                ],
                columns=pd.RangeIndex(1, 11)
            )
            df.loc[("4 - Медленный мобильный Интернет", "5 - Медленная загрузка видео"), :] = '+'
            df.loc[:, (9, 10)] = '+'
            df.fillna('-', inplace=True)
            s = df.style
            s.set_table_styles([
                # {'selector': 'th, td', 'props': 'border: 1px solid lightgray;'},
                {'selector': 'th:not(.blank)',
                 'props': f'font-size: 16px; color: white; background-color: {MegafonColors.brandPurple80};'},
                {'selector': 'th.col_heading', 'props': 'text-align: center; width: 50px;'},
                {'selector': 'td', 'props': 'text-align: center; font-size: 16px; font-weight: bold;'},
                {'selector': 'th.blank', 'props': 'border-style: none'}
            ], overwrite=False)
            s = s.applymap(lambda v:
                           f'color: white; background-color: {MegafonColors.brandGreen};' if v == '+'
                           else f'color: {MegafonColors.spbSky2}; background-color: {MegafonColors.spbSky1};')
            st.markdown(s.to_html(table_uuid="table_client_choosing"), unsafe_allow_html=True)
        with tab2:
            c1, c2 = st.columns(2, gap='large')
            with c1:
                st.markdown(set_text_style('<b>Категории оценки сервиса мобильного интернета</b>', font_size=32,
                                           text_align='center', color=MegafonColors.brandPurple),
                            unsafe_allow_html=True)
                df = pd.DataFrame(
                    index=["Ужасно", "Плохо", "Нормально", "Хорошо", "Отлично"],
                    columns=pd.RangeIndex(1, 11)
                )
                df.loc["Ужасно", (1, 2)] = '+'
                df.loc["Плохо", (3, 4)] = '+'
                df.loc["Нормально", (5, 6)] = '+'
                df.loc["Хорошо", (7, 8)] = '+'
                df.loc["Отлично", (9, 10)] = '+'
                df.fillna('-', inplace=True)
                s = df.style
                s.set_table_styles([
                    # {'selector': 'th, td', 'props': 'border: 1px solid lightgray;'},
                    {'selector': 'th:not(.blank)',
                     'props': f'font-size: 16px; color: white; background-color: {MegafonColors.brandPurple80};'},
                    {'selector': 'th.col_heading', 'props': 'text-align: center; width: 50px;'},
                    {'selector': 'td', 'props': 'text-align: center; font-size: 16px; font-weight: bold;'},
                    {'selector': 'th.blank', 'props': 'border-style: none'}
                ], overwrite=False)
                s = s.applymap(lambda v:
                               f'color: white; background-color: {MegafonColors.brandGreen};' if v == '+'
                               else f'color: {MegafonColors.spbSky2}; background-color: {MegafonColors.spbSky1};')
                st.markdown(s.to_html(table_uuid="table_categories_1"), unsafe_allow_html=True)
            with c2:
                st.markdown(set_text_style('<b>Категории причин недовольства сервисом мобильного интернета</b>',
                                           font_size=32, text_align='center', color=MegafonColors.brandPurple),
                            unsafe_allow_html=True)
                df = pd.DataFrame(
                    index=["Интернет и видео", "Интернет", "Видео", "Нет"],
                    columns=[
                        "4 - Медленный мобильный Интернет", "5 - Медленная загрузка видео",
                    ]
                )
                df.loc["Интернет и видео", ("4 - Медленный мобильный Интернет", "5 - Медленная загрузка видео")] = '+'
                df.loc["Интернет", "4 - Медленный мобильный Интернет"] = '+'
                df.loc["Видео", "5 - Медленная загрузка видео"] = '+'
                df.fillna('-', inplace=True)
                s = df.style
                s.set_table_styles([
                    # {'selector': 'th, td', 'props': 'border: 1px solid lightgray;'},
                    {'selector': 'th:not(.blank)',
                     'props': f'font-size: 16px; color: white; background-color: {MegafonColors.brandPurple80};'},
                    {'selector': 'th.col_heading', 'props': 'text-align: center; width: 200px;'},
                    {'selector': 'td', 'props': 'text-align: center; font-size: 16px; font-weight: bold;'},
                    {'selector': 'th.blank', 'props': 'border-style: none'}
                ], overwrite=False)
                s = s.applymap(lambda v:
                               f'color: white; background-color: {MegafonColors.brandGreen};' if v == '+'
                               else f'color: {MegafonColors.spbSky2}; background-color: {MegafonColors.spbSky1};')
                st.markdown(s.to_html(table_uuid="table_categories_2"), unsafe_allow_html=True)
        with tab3:
            st.markdown('&nbsp;')
            s = display_cat_info(data_clean)
            st.markdown(s.to_html(table_uuid="table_categories_dist"), unsafe_allow_html=True)

    case "Выбор метрик, статистик и критериев":

        @st.cache_resource
        def show_important_metric_table(metrics: pd.DataFrame):
            df = metrics.loc[
                ['Downlink Throughput(Kbps)', 'Video Streaming Download Throughput(Kbps)',
                'Web Page Download Throughput(Kbps)'], ['description']
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


        tabs = ['Метрики и статистика',
                'Критерии и уровень значимости',
                ]
        tab1, tab2 = st.tabs(tabs)
        for tab in tabs:
            set_widget_style(tab, font_size=24)
        with tab1:
            st.markdown(set_text_style('<b>Метрики оценки скорости интернета</b>', font_size=32,
                                       color=MegafonColors.brandPurple, text_align='center'),
                        unsafe_allow_html=True)
            st.markdown(
                set_text_style('''Эксперты компании [Ookla](https://www.ookla.com/) оценивают скорость интернет, 
                предоставляемого оператором, на основании двух метрик: <b>Средняя скорость «к абоненту»</b> и 
                <b>Средняя скорость «от абонента»</b> в соотношении <b>9:1</b>''', tag='span', font_size=18),
                unsafe_allow_html=True
            )
            # st.markdown(set_text_style('<b>Важные метрики:</b>', font_size=24, color=MegafonColors.orangeDark),
            #             unsafe_allow_html=True)
            st.markdown('''       
            <ul>
            <li><span style="font-size:24px">Средняя скорость «к абоненту»</span></li>
            <li><span style="font-size:24px">Скорость загрузки потокового видео</span></li>
            <li><span style="font-size:24px">Скорость загрузки web-страниц через браузер</span></li>
            </ul>
            </p>
            ''', unsafe_allow_html=True)

            st.markdown('---')
            st.markdown(set_text_style('<b>Статистика оценки скорости интернета</b>', font_size=32,
                                       color=MegafonColors.brandPurple, text_align='center'),
                        unsafe_allow_html=True)
            st.markdown(
                set_text_style('''Компания [Ookla](https://www.ookla.com/) в методологии оценки средней скорости 
                интернет соединения провайдеров использует модифицированный вариант тримера:''', tag='span',
                               font_size=18),
                unsafe_allow_html=True
            )
            st.latex('\hat{TM}={P_{10}+8\cdot P_{50}+P_{90} \over {10}}')
            st.markdown('')
            statistic_distributions = get_stat_dist()
            fig = plot_metric_histograms_4_1()
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
        with tab2:
            st.markdown('')
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(set_text_style('<b>Критерий</b>', font_size=32, color=MegafonColors.brandPurple,
                                           text_align='center'),
                            unsafe_allow_html=True)
                st.markdown(set_text_style('Перестановочный тест (Permutation test)', font_size=24, text_align='center'),
                            unsafe_allow_html=True)
            with c2:
                st.markdown(set_text_style('<b>Оценка доверительного интервала</b>', font_size=32, color=MegafonColors.brandPurple,
                                           text_align='center'),
                            unsafe_allow_html=True)
                st.markdown(set_text_style('Бутстрап (Bootstrapping)', font_size=24, text_align='center'),
                            unsafe_allow_html=True)
            st.markdown('---')
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(set_text_style('<b>Уровень значимости</b>', font_size=32, color=MegafonColors.brandPurple,
                                           text_align='center'),
                            unsafe_allow_html=True)
                st.latex('\large \\alpha=0.05')
            with c2:
                st.markdown(set_text_style('<b>Уровень доверия</b>', font_size=32, color=MegafonColors.brandPurple,
                                           text_align='center'),
                            unsafe_allow_html=True)
                st.latex('\large \\beta=1-\\alpha=0.95')
            st.markdown('---')
            st.markdown(set_text_style('<b>Правило принятия решения</b>', font_size=32, color=MegafonColors.brandPurple,
                                       text_align='center'),
                        unsafe_allow_html=True)
            rule_text = '''
            <ul>
            <li><span style="font-size:24px">Если p-value для всех метрик ниже уровня значимости, то считаем, 
            что клиенты тестовых групп принадлежат к одной популяции<span></li>
            <li><span style="font-size:24px">В противном случае считаем, что клиенты тестовых групп принадлежат 
            к разным популяциям</li>
            </ul>
            '''
            st.markdown(rule_text, unsafe_allow_html=True)
    case "Причины недовольства сервисом мобильного интернета":
        tabs = ['Цель исследования',
                'Разведочный анализ',
                'Статистические тесты',
                'Выводы']
        tab1, tab2, tab3, tab4 = st.tabs(tabs)
        for tab in tabs:
            set_widget_style(tab, font_size=24)
        with tab1:
            c1, c2 = st.columns(2, gap='medium')
            with c1:
                st.markdown('&nbsp;')
                s = display_cat_info(data_clean).set_properties(pd.IndexSlice['Ужасно':'Хорошо', 'Интернет и видео'],
                                                                color='white',
                                                                background=px.colors.DEFAULT_PLOTLY_COLORS[0],
                                                                opacity=0.5).set_properties(
                    pd.IndexSlice['Ужасно':'Хорошо', 'Интернет'], color='white',
                    background=px.colors.DEFAULT_PLOTLY_COLORS[1], opacity=0.5).set_properties(
                    pd.IndexSlice['Ужасно':'Хорошо', 'Видео'], color='white',
                    background=px.colors.DEFAULT_PLOTLY_COLORS[2],
                    opacity=0.5)
                st.markdown(s.to_html(table_uuid="table_categories_dist"), unsafe_allow_html=True)
                categories = '''
            <br>
            <ul>
            <li><span style="color:white;background-color:rgb(31, 119, 180);opacity:0.5;font-size:18px">&nbsp;Интернет и видео&nbsp;</span> - <span style="font-size:18px">недовольные скоростью мобильного интернета и загрузкой видео</span></li>
            <li><span style="color:white;background-color:rgb(255, 127, 14);opacity:0.5;font-size:18px">&nbsp;Интернет&nbsp;</span> - <span style="font-size:18px">недовольные в первую очередь скоростью мобильного интернета</span></li>
            <li><span style="color:white;background-color:rgb(44, 160, 44);opacity:0.5;font-size:18px">&nbsp;Видео&nbsp;</span> - <span style="font-size:18px">недовольные в первую очередь скоростью загрузки видео</span></li>
            </ul>
                            '''
                st.markdown(categories, unsafe_allow_html=True)
            with c2:
                st.markdown('&nbsp;')
                goal_text = set_text_style('<b>Цель</b><br>', tag='p', color=MegafonColors.orangeDark, font_size=32,
                                           text_align='center')
                goal_text += set_text_style('Выяснить, является ли разделение статистически верным?',
                                            tag='span', font_size=24)
                st.markdown(goal_text, unsafe_allow_html=True)
                st.markdown('&nbsp;')
                task_text = set_text_style('<b>Вопросы</b><br>', tag='p',
                                           color=MegafonColors.orangeDark, font_size=32, text_align='center')
                task_text += set_text_style(f"""
                <ul>
                <li>{set_text_style('Принадлежат ли клиенты вышеуказанных групп к разным популяциям?',
                                    font_size=24)}</li>
                <li>{set_text_style('Если клиенты не принадлежат к одной популяции, то по каким метрикам особенно '
                                    'сильно различаются?', font_size=24)}</li>
                </ul>
                """, tag='span', font_size=24)
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
            st.markdown(set_text_style('<b>Доверительные интервалы статистик</b>', text_align='center',
                                       font_size=32, color=MegafonColors.brandPurple),
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
                    # title=f'<b>Плотность нулевого распределения вероятностей тестовой статистики</b>', title_y=0.9,
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
                    # title=f'<b>Плотность нулевого распределения вероятностей тестовой статистики</b>', title_y=0.9,
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
                    # title=f'<b>Плотность нулевого распределения вероятностей тестовой статистики</b>', title_y=0.9,
                    labels_font_size=16, axes_tickfont_size=14, units_font_size=14,
                    height=300, n_cols=3, opacity=0.5,
                    histnorm='probability density',
                    add_kde=True, add_statistic=True, mark_statistic=mark_statistic,
                    horizontal_spacing=0.08, vertical_spacing=0.07)


            alternatives, pvalues, mark_statistic, null_distributions, statistics = load_5_4_1()

            st.markdown(set_text_style(f'<b>Группы "{groups[0]}" и "{groups[1]}"</b>', text_align='center',
                                       font_size=32, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('<b>Гипотезы</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([40, 30, 30])
            st.markdown('&nbsp;')
            c1.latex('\Delta \hat{TM}=\hat{TM}_{Интернет \, и \, видео}-\hat{TM}_{Интернет}')
            c2.latex('H_0:\Delta \hat{TM}=0')
            c3.latex('H_1:\Delta \hat{TM}≠0')
            st.markdown(set_text_style('<b>Плотность нулевого распределения вероятностей тестовой статистики</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            fig = plot_metric_histograms_5_4_1(null_distributions, statistics, research_metrics, mark_statistic)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            group_pair_index = 0
            st.markdown(set_text_style(f'<b>Значения p-value статистического теста</b>', font_size=24,
                                       color=MegafonColors.brandGreenDarken10, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                                metrics=research_metrics, alpha=alpha,
                                caption='', col_width=400)
            st.markdown(s.to_html(table_uuid="table_pvalues_5_4_1"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Выводы:</b>', tag='p',
                                             color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'есть значимые различия</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)

            _, pvalues, mark_statistic, null_distributions, statistics = load_5_4_2()

            st.markdown('---')
            st.markdown(set_text_style(f'<b>Группы "{groups[1]}" и "{groups[2]}"</b>', text_align='center',
                                       font_size=32, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('<b>Гипотезы</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([40, 30, 30])
            st.markdown('&nbsp;')
            c1.latex('\Delta \hat{TM}=\hat{TM}_{Интернет \, и \, видео}-\hat{TM}_{Видео}')
            c2.latex('H_0:\Delta \hat{TM}=0')
            c3.latex('H_1:\Delta \hat{TM}≠0')
            st.markdown(set_text_style('<b>Плотность нулевого распределения вероятностей тестовой статистики</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            fig = plot_metric_histograms_5_4_2(null_distributions, statistics, research_metrics, mark_statistic)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            group_pair_index = 1
            st.markdown(set_text_style(f'<b>Значения p-value статистического теста</b>', font_size=24,
                                       color=MegafonColors.brandGreenDarken10, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                                metrics=research_metrics, alpha=alpha,
                                caption='', col_width=400)
            st.markdown(s.to_html(table_uuid="table_pvalues_5_4_2"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Вывод:</b><br>', tag='p', font_size=24,
                                             color=MegafonColors.brandGreenDarken10, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'нет значимых различий</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)

            _, pvalues, mark_statistic, null_distributions, statistics = load_5_4_3()

            st.markdown('---')
            st.markdown(set_text_style(f'<b>Группы "{groups[2]}" и "{groups[0]}"</b>', text_align='center',
                                       font_size=32, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('<b>Гипотезы</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([40, 30, 30])
            st.markdown('&nbsp;')
            c1.latex('\Delta \hat{TM}=\hat{TM}_{Видео}-\hat{TM}_{Интернет \, и \, видео}')
            c2.latex('H_0:\Delta \hat{TM}=0')
            c3.latex('H_1:\Delta \hat{TM}≠0')
            st.markdown(set_text_style('<b>Плотность нулевого распределения вероятностей тестовой статистики</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            fig = plot_metric_histograms_5_4_3(null_distributions, statistics, research_metrics, mark_statistic)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            group_pair_index = 2
            st.markdown(set_text_style(f'<b>Значения p-value статистического теста</b>', font_size=24,
                                       color=MegafonColors.brandGreenDarken10, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                                metrics=research_metrics, alpha=alpha,
                                caption='', col_width=400)
            st.markdown(s.to_html(table_uuid="table_pvalues_5_4_3"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Выводы:</b>', tag='p',
                                             color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'есть значимые различия</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)
        with tab4:
            st.markdown(set_text_style(f'<b>Значения p-value статистических тестов</b>', font_size=32,
                                       color=MegafonColors.brandPurple, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues, metrics=research_metrics, alpha=alpha,
                                caption='', opacity=0.5, index_width=180, col_width=300)
            st.markdown(s.to_html(table_uuid="table_pvalues_5_5"), unsafe_allow_html=True)
            st.markdown('---')
            # st.markdown('---')
            # st.markdown(set_text_style(f'<b>Категории по причинам недовольства сервисом мобильного интернета</b>', font_size=32,
            #                            color=MegafonColors.brandPurple, text_align='center'),
            #             unsafe_allow_html=True)
            # st.markdown('&nbsp;')
            c1, c2 = st.columns(2, gap='medium')
            with c1:
                s = display_cat_info(data_clean) \
                    .set_properties(pd.IndexSlice['Ужасно':'Хорошо', 'Интернет и видео'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[0], opacity=0.5) \
                    .set_properties(pd.IndexSlice['Ужасно':'Хорошо', 'Интернет': 'Видео'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[1], opacity=0.5)
                st.markdown(s.to_html(table_uuid="table_categories_5_5"), unsafe_allow_html=True)
                categories = '''
                <br>
                <ul>
                <li><span style="color:white;background-color:rgb(31, 119, 180);opacity:0.5;font-size:18px">&nbsp;Интернет и видео&nbsp;</span>&nbsp- Недовольные скоростью мобильного интернета и загрузкой видео</li>
                <li><span style="color:white;background-color:rgb(255, 127, 14);opacity:0.5;font-size:18px">&nbsp;Интернет или видео&nbsp;</span>&nbsp- Недовольные скоростью мобильного интернета или загрузкой видео</li>
                </ul>
                '''
                st.markdown(categories, unsafe_allow_html=True)
            with c2:
                st.markdown(set_text_style(f'<b>Выводы</b>', font_size=32,
                                           color=MegafonColors.orangeDark, text_align='center'),
                            unsafe_allow_html=True)
                conclusion_text = '''
                <ul>
                <li><span style="font-size:24px">Клиенты группы "<b>Интернет</b>" и "<b>Видео</b>" 
                принадлежат к <b>одной популяции</b>.</span></li>
                <li><span style="font-size:24px">Группа "<b>Интернет и видео</b>" 
                принадлежит к <b>отдельной популяции</b>.</span></li>
                </ul>
                '''
                st.markdown(conclusion_text, unsafe_allow_html=True)
    case "Оценки качества сервиса мобильного интернета":
        tabs = ['Цель исследования',
                'Разведочный анализ',
                'Статистические тесты',
                'Выводы']
        tab1, tab2, tab3, tab4 = st.tabs(tabs)
        for tab in tabs:
            set_widget_style(tab, font_size=24)
        with tab1:
            c1, c2 = st.columns(2, gap='medium')
            with c1:
                st.markdown('&nbsp;')
                s = display_cat_info(data_clean)\
                    .set_properties(pd.IndexSlice['Ужасно', 'Интернет и видео': 'Видео'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[0], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Плохо', 'Интернет и видео': 'Видео'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[1], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Нормально', 'Интернет и видео': 'Видео'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[2], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Хорошо', 'Интернет и видео': 'Видео'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[3], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Отлично', 'Нет'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[4], opacity=0.5)
                st.markdown(s.to_html(table_uuid="table_categories_dist"), unsafe_allow_html=True)
                categories = '''
            <br>
            <ul>
            <li><span style="color:white;background-color:rgb(31, 119, 180);opacity:0.5;font-size:18px">&nbsp;Ужасно&nbsp;</span></li>
            <li><span style="color:white;background-color:rgb(255, 127, 14);opacity:0.5;font-size:18px">&nbsp;Плохо&nbsp;</span></li>
            <li><span style="color:white;background-color:rgb(44, 160, 44);opacity:0.5;font-size:18px">&nbsp;Нормально&nbsp;</span></li>
            <li><span style="color:white;background-color:rgb(214, 39, 40);opacity:0.5;font-size:18px">&nbsp;Хорошо&nbsp;</span></li>
            <li><span style="color:white;background-color:rgb(148, 103, 189);opacity:0.5;font-size:18px">&nbsp;Отлично&nbsp;</span></li>
            </ul>
                            '''
                st.markdown(categories, unsafe_allow_html=True)
            with c2:
                st.markdown('&nbsp;')
                goal_text = set_text_style('<b>Цель</b><br>', tag='p', color=MegafonColors.orangeDark, font_size=32,
                                           text_align='center')
                goal_text += set_text_style('Выяснить, является ли разделение статистически верным?',
                                            tag='span', font_size=24)
                st.markdown(goal_text, unsafe_allow_html=True)
                st.markdown('&nbsp;')
                task_text = set_text_style('<b>Вопросы</b><br>', tag='p',
                                           color=MegafonColors.orangeDark, font_size=32, text_align='center')
                task_text += set_text_style(f"""
                <ul>
                <li>{set_text_style('Принадлежат ли клиенты вышеуказанных групп к разным популяциям?',
                                    font_size=24)}</li>
                <li>{set_text_style('Если клиенты не принадлежат к одной популяции, то по каким метрикам особенно '
                                    'сильно различаются?', font_size=24)}</li>
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
            st.markdown(set_text_style('<b>Доверительные интервалы статистик</b>', text_align='center',
                                       font_size=32, color=MegafonColors.brandPurple),
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
                    # title=f'<b>Плотность нулевого распределения вероятностей тестовой статистики</b>', title_y=0.9,
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
                    # title=f'<b>Плотность нулевого распределения вероятностей тестовой статистики</b>', title_y=0.9,
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
                    # title=f'<b>Плотность нулевого распределения вероятностей тестовой статистики</b>', title_y=0.9,
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
                    # title=f'<b>Плотность нулевого распределения вероятностей тестовой статистики</b>', title_y=0.9,
                    labels_font_size=16, axes_tickfont_size=14, units_font_size=14,
                    height=300, n_cols=3, opacity=0.5,
                    histnorm='probability density',
                    add_kde=True, add_statistic=True, mark_statistic=mark_statistic,
                    horizontal_spacing=0.08, vertical_spacing=0.07)


            alternatives, pvalues, mark_statistic, null_distributions, statistics = load_6_4_1()

            st.markdown(set_text_style(f'<b>Группы "{groups[0]}" и "{groups[1]}"</b>', text_align='center',
                                       font_size=32, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('<b>Гипотезы</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([40, 30, 30])
            st.markdown('&nbsp;')
            c1.latex('\Delta \hat{TM}=\hat{TM}_{Ужасно}-\hat{TM}_{Плохо}')
            c2.latex('H_0:\Delta \hat{TM}≥0')
            c3.latex('H_1:\Delta \hat{TM}<0')
            st.markdown(set_text_style('<b>Плотность нулевого распределения вероятностей тестовой статистики</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            fig = plot_metric_histograms_6_4_1(null_distributions, statistics, research_metrics, mark_statistic)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            group_pair_index = 0
            st.markdown(set_text_style(f'<b>Значения p-value статистического теста</b>', font_size=24,
                                       color=MegafonColors.brandGreenDarken10, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                                metrics=research_metrics, alpha=alpha,
                                caption='', col_width=400)
            st.markdown(s.to_html(table_uuid="table_pvalues_6_4_1"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Выводы:</b>', tag='p',
                                             color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'есть значимые различия</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)

            _, pvalues, mark_statistic, null_distributions, statistics = load_6_4_2()

            st.markdown('---')
            st.markdown(set_text_style(f'<b>Группы "{groups[1]}" и "{groups[2]}"</b>', text_align='center',
                                       font_size=32, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('<b>Гипотезы</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([40, 30, 30])
            st.markdown('&nbsp;')
            c1.latex('\Delta \hat{TM}=\hat{TM}_{Плохо}-\hat{TM}_{Нормально}')
            c2.latex('H_0:\Delta \hat{TM}≥0')
            c3.latex('H_1:\Delta \hat{TM}<0')
            st.markdown(set_text_style('<b>Плотность нулевого распределения вероятностей тестовой статистики</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            fig = plot_metric_histograms_6_4_2(null_distributions, statistics, research_metrics, mark_statistic)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            group_pair_index = 1
            st.markdown(set_text_style(f'<b>Значения p-value статистического теста</b>', font_size=24,
                                       color=MegafonColors.brandGreenDarken10, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                                metrics=research_metrics, alpha=alpha,
                                caption='', col_width=400)
            st.markdown(s.to_html(table_uuid="table_pvalues_6_4_2"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Вывод:</b><br>', tag='p', font_size=24,
                                             color=MegafonColors.brandGreenDarken10, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'нет значимых различий</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)

            _, pvalues, mark_statistic, null_distributions, statistics = load_6_4_3()

            st.markdown('---')
            st.markdown(set_text_style(f'<b>Группы "{groups[2]}" и "{groups[3]}"</b>', text_align='center',
                                       font_size=32, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('<b>Гипотезы</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([40, 30, 30])
            st.markdown('&nbsp;')
            c1.latex('\Delta \hat{TM}=\hat{TM}_{Нормально}-\hat{TM}_{Хорошо}')
            c2.latex('H_0:\Delta \hat{TM}≥0')
            c3.latex('H_1:\Delta \hat{TM}<0')
            st.markdown(set_text_style('<b>Плотность нулевого распределения вероятностей тестовой статистики</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            fig = plot_metric_histograms_6_4_3(null_distributions, statistics, research_metrics, mark_statistic)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            group_pair_index = 2
            st.markdown(set_text_style(f'<b>Значения p-value статистического теста</b>', font_size=24,
                                       color=MegafonColors.brandGreenDarken10, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                                metrics=research_metrics, alpha=alpha,
                                caption='', col_width=400)
            st.markdown(s.to_html(table_uuid="table_pvalues_6_4_3"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Выводы:</b>', tag='p',
                                             color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'есть значимые различия</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)

            _, pvalues, mark_statistic, null_distributions, statistics = load_6_4_4()

            st.markdown('---')
            st.markdown(set_text_style(f'<b>Группы "{groups[3]}" и "{groups[0]}"</b>', text_align='center',
                                       font_size=32, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('<b>Гипотезы</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([40, 30, 30])
            st.markdown('&nbsp;')
            c1.latex('\Delta \hat{TM}=\hat{TM}_{Хорошо}-\hat{TM}_{Отлично}')
            c2.latex('H_0:\Delta \hat{TM}≥0')
            c3.latex('H_1:\Delta \hat{TM}<0')
            st.markdown(set_text_style('<b>Плотность нулевого распределения вероятностей тестовой статистики</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            fig = plot_metric_histograms_6_4_4(null_distributions, statistics, research_metrics, mark_statistic)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            group_pair_index = 2
            st.markdown(set_text_style(f'<b>Значения p-value статистического теста</b>', font_size=24,
                                       color=MegafonColors.brandGreenDarken10, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                                metrics=research_metrics, alpha=alpha,
                                caption='', col_width=400)
            st.markdown(s.to_html(table_uuid="table_pvalues_6_4_4"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Выводы:</b>', tag='p',
                                             color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'есть значимые различия</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)
        with tab4:
            st.markdown(set_text_style(f'<b>Значения p-value статистических тестов</b>', font_size=32,
                                       color=MegafonColors.brandPurple, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues, metrics=research_metrics, alpha=alpha,
                                caption='', opacity=0.5, index_width=180, col_width=300)
            st.markdown(s.to_html(table_uuid="table_pvalues_6_5"), unsafe_allow_html=True)
            st.markdown('---')
            # st.markdown('---')
            # st.markdown(set_text_style(f'<b>Категории по причинам недовольства сервисом мобильного интернета</b>', font_size=32,
            #                            color=MegafonColors.brandPurple, text_align='center'),
            #             unsafe_allow_html=True)
            # st.markdown('&nbsp;')
            c1, c2 = st.columns(2, gap='medium')
            with c1:
                s = display_cat_info(data_clean)\
                    .set_properties(pd.IndexSlice['Ужасно': 'Плохо', 'Интернет и видео':'Видео'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[0], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Нормально': 'Хорошо', 'Интернет и видео':'Видео'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[1], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Отлично', 'Нет'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[2], opacity=0.5)
                st.markdown(s.to_html(table_uuid="table_categories_6_5"), unsafe_allow_html=True)
                categories = '''
                <br>
                <ul>
                <li><span style="color:white;background-color:rgb(31, 119, 180);opacity:0.5;font-size:18px">&nbsp;Неудовлетворительно&nbsp;</span>&nbsp</li>
                <li><span style="color:white;background-color:rgb(255, 127, 14);opacity:0.5;font-size:18px">&nbsp;Удовлетворительно&nbsp;</span>&nbsp</li>
                <li><span style="color:white;background-color:rgb(44, 160, 44);opacity:0.5;font-size:18px">&nbsp;Отлично&nbsp;</span>&nbsp</li>
                </ul>
                '''
                st.markdown(categories, unsafe_allow_html=True)
            with c2:
                st.markdown(set_text_style(f'<b>Выводы</b>', font_size=32,
                                           color=MegafonColors.orangeDark, text_align='center'),
                            unsafe_allow_html=True)
                conclusion_text = '''
                <ul>
                <li><span style="font-size:24px">Клиенты группы "<b>Ужасно</b>" и "<b>Плохо</b>" 
                принадлежат к <b>одной популяции</b>.</span></li>
                <li><span style="font-size:24px">Клиенты группы "<b>Нормально</b>" и "<b>Хорошо</b>" 
                принадлежат к <b>одной популяции</b>.</span></li>
                <li><span style="font-size:24px">Самое сильное влияние у метрики "<b>Скорость загрузки потокового видео</b>", 
                а самое слабое у метрики <b>Средняя скорость «к абоненту»</b>.</span></li>
                </ul>
                '''
                st.markdown(conclusion_text, unsafe_allow_html=True)
    case "Уровни удовлетворенности сервисом мобильного интернета":
        tabs = ['Цель исследования',
                'Разведочный анализ',
                'Статистические тесты',
                'Выводы']
        tab1, tab2, tab3, tab4 = st.tabs(tabs)
        for tab in tabs:
            set_widget_style(tab, font_size=24)
        with tab1:
            c1, c2 = st.columns(2, gap='medium')
            with c1:
                st.markdown('&nbsp;')
                s = display_cat_info(data_clean)\
                    .set_properties(pd.IndexSlice['Ужасно': 'Плохо', 'Интернет и видео'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[0], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Ужасно': 'Плохо', 'Интернет': 'Видео'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[1], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Нормально': 'Хорошо', 'Интернет и видео'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[2], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Нормально': 'Хорошо', 'Интернет': 'Видео'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[3], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Отлично', 'Нет'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[4], opacity=0.5)
                st.markdown(s.to_html(table_uuid="table_categories_dist"), unsafe_allow_html=True)
                categories = '''
            <br>
            <ul>
            <li><span style="color:white;background-color:rgb(31, 119, 180);opacity:0.5;font-size:18px">&nbsp;1&nbsp;</span> - Полностью неудовлетворены</li>
            <li><span style="color:white;background-color:rgb(255, 127, 14);opacity:0.5;font-size:18px">&nbsp;2&nbsp;</span> - Частично неудовлетворены</li>
            <li><span style="color:white;background-color:rgb(44, 160, 44);opacity:0.5;font-size:18px">&nbsp;3&nbsp;</span> - Ни удовлетворены, ни разочарованы</li>
            <li><span style="color:white;background-color:rgb(214, 39, 40);opacity:0.5;font-size:18px">&nbsp;4&nbsp;</span> - Частично удовлетворены</li>
            <li><span style="color:white;background-color:rgb(148, 103, 189);opacity:0.5;font-size:18px">&nbsp;5&nbsp;</span> - Полностью удовлетворены</li>
            </ul>
                            '''
                st.markdown(categories, unsafe_allow_html=True)
            with c2:
                st.markdown('&nbsp;')
                goal_text = set_text_style('<b>Цель</b><br>', tag='p', color=MegafonColors.orangeDark, font_size=32,
                                           text_align='center')
                goal_text += set_text_style('Выяснить, является ли разделение статистически верным?',
                                            tag='span', font_size=24)
                st.markdown(goal_text, unsafe_allow_html=True)
                st.markdown('&nbsp;')
                task_text = set_text_style('<b>Вопросы</b><br>', tag='p',
                                           color=MegafonColors.orangeDark, font_size=32, text_align='center')
                task_text += set_text_style(f"""
                <ul>
                <li>{set_text_style('Принадлежат ли клиенты с разными CSAT к разным популяциям?',
                                    font_size=24)}</li>
                <li>{set_text_style('Если клиенты с разными CSAT не принадлежат к одной популяции, '
                                    'то по каким метрикам особенно сильно различаются?', font_size=24)}</li>
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
            st.markdown(set_text_style('<b>Доверительные интервалы статистик</b>', text_align='center',
                                       font_size=32, color=MegafonColors.brandPurple),
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
                    # title=f'<b>Плотность нулевого распределения вероятностей тестовой статистики</b>', title_y=0.9,
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
                    # title=f'<b>Плотность нулевого распределения вероятностей тестовой статистики</b>', title_y=0.9,
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
                    # title=f'<b>Плотность нулевого распределения вероятностей тестовой статистики</b>', title_y=0.9,
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
                    # title=f'<b>Плотность нулевого распределения вероятностей тестовой статистики</b>', title_y=0.9,
                    labels_font_size=16, axes_tickfont_size=14, units_font_size=14,
                    height=300, n_cols=3, opacity=0.5,
                    histnorm='probability density',
                    add_kde=True, add_statistic=True, mark_statistic=mark_statistic,
                    horizontal_spacing=0.08, vertical_spacing=0.07)


            alternatives, pvalues, mark_statistic, null_distributions, statistics = load_7_4_1()

            st.markdown(set_text_style(f'<b>Группы "{groups[0]}" и "{groups[1]}"</b>', text_align='center',
                                       font_size=32, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('<b>Гипотезы</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([40, 30, 30])
            st.markdown('&nbsp;')
            c1.latex('\Delta \hat{TM}=\hat{TM}_{1}-\hat{TM}_{2}')
            c2.latex('H_0:\Delta \hat{TM}≥0')
            c3.latex('H_1:\Delta \hat{TM}<0')
            st.markdown(set_text_style('<b>Плотность нулевого распределения вероятностей тестовой статистики</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            fig = plot_metric_histograms_7_4_1(null_distributions, statistics, research_metrics, mark_statistic)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            group_pair_index = 0
            st.markdown(set_text_style(f'<b>Значения p-value статистического теста</b>', font_size=24,
                                       color=MegafonColors.brandGreenDarken10, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                                metrics=research_metrics, alpha=alpha,
                                caption='', col_width=400)
            st.markdown(s.to_html(table_uuid="table_pvalues_7_4_1"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Выводы:</b>', tag='p',
                                             color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'есть значимые различия</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)

            _, pvalues, mark_statistic, null_distributions, statistics = load_7_4_2()

            st.markdown('---')
            st.markdown(set_text_style(f'<b>Группы "{groups[1]}" и "{groups[2]}"</b>', text_align='center',
                                       font_size=32, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('<b>Гипотезы</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([40, 30, 30])
            st.markdown('&nbsp;')
            c1.latex('\Delta \hat{TM}=\hat{TM}_{2}-\hat{TM}_{3}')
            c2.latex('H_0:\Delta \hat{TM}≥0')
            c3.latex('H_1:\Delta \hat{TM}<0')
            st.markdown(set_text_style('<b>Плотность нулевого распределения вероятностей тестовой статистики</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            fig = plot_metric_histograms_7_4_2(null_distributions, statistics, research_metrics, mark_statistic)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            group_pair_index = 1
            st.markdown(set_text_style(f'<b>Значения p-value статистического теста</b>', font_size=24,
                                       color=MegafonColors.brandGreenDarken10, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                                metrics=research_metrics, alpha=alpha,
                                caption='', col_width=400)
            st.markdown(s.to_html(table_uuid="table_pvalues_7_4_2"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Вывод:</b><br>', tag='p', font_size=24,
                                             color=MegafonColors.brandGreenDarken10, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'нет значимых различий</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)

            _, pvalues, mark_statistic, null_distributions, statistics = load_7_4_3()

            st.markdown('---')
            st.markdown(set_text_style(f'<b>Группы "{groups[2]}" и "{groups[3]}"</b>', text_align='center',
                                       font_size=32, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('<b>Гипотезы</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([40, 30, 30])
            st.markdown('&nbsp;')
            c1.latex('\Delta \hat{TM}=\hat{TM}_{3}-\hat{TM}_{4}')
            c2.latex('H_0:\Delta \hat{TM}≥0')
            c3.latex('H_1:\Delta \hat{TM}<0')
            st.markdown(set_text_style('<b>Плотность нулевого распределения вероятностей тестовой статистики</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            fig = plot_metric_histograms_7_4_3(null_distributions, statistics, research_metrics, mark_statistic)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            group_pair_index = 2
            st.markdown(set_text_style(f'<b>Значения p-value статистического теста</b>', font_size=24,
                                       color=MegafonColors.brandGreenDarken10, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                                metrics=research_metrics, alpha=alpha,
                                caption='', col_width=400)
            st.markdown(s.to_html(table_uuid="table_pvalues_7_4_3"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Выводы:</b>', tag='p',
                                             color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'есть значимые различия</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)

            _, pvalues, mark_statistic, null_distributions, statistics = load_7_4_4()

            st.markdown('---')
            st.markdown(set_text_style(f'<b>Группы "{groups[3]}" и "{groups[0]}"</b>', text_align='center',
                                       font_size=32, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('<b>Гипотезы</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([40, 30, 30])
            st.markdown('&nbsp;')
            c1.latex('\Delta \hat{TM}=\hat{TM}_{4}-\hat{TM}_{5}')
            c2.latex('H_0:\Delta \hat{TM}≥0')
            c3.latex('H_1:\Delta \hat{TM}<0')
            st.markdown(set_text_style('<b>Плотность нулевого распределения вероятностей тестовой статистики</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            fig = plot_metric_histograms_7_4_4(null_distributions, statistics, research_metrics, mark_statistic)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            group_pair_index = 2
            st.markdown(set_text_style(f'<b>Значения p-value статистического теста</b>', font_size=24,
                                       color=MegafonColors.brandGreenDarken10, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                                metrics=research_metrics, alpha=alpha,
                                caption='', col_width=400)
            st.markdown(s.to_html(table_uuid="table_pvalues_7_4_4"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Выводы:</b>', tag='p',
                                             color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'есть значимые различия</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)
        with tab4:
            st.markdown(set_text_style(f'<b>Значения p-value статистических тестов</b>', font_size=32,
                                       color=MegafonColors.brandPurple, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues, metrics=research_metrics, alpha=alpha,
                                caption='', opacity=0.5, index_width=30, col_width=300)
            st.markdown(s.to_html(table_uuid="table_pvalues_7_5"), unsafe_allow_html=True)
            st.markdown('---')
            # st.markdown('---')
            # st.markdown(set_text_style(f'<b>Категории по причинам недовольства сервисом мобильного интернета</b>', font_size=32,
            #                            color=MegafonColors.brandPurple, text_align='center'),
            #             unsafe_allow_html=True)
            # st.markdown('&nbsp;')
            c1, c2 = st.columns(2, gap='medium')
            with c1:
                s = display_cat_info(data_clean)\
                    .set_properties(pd.IndexSlice['Ужасно': 'Плохо', 'Интернет и видео'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[0], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Ужасно': 'Плохо', 'Интернет': 'Видео'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[1], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Нормально': 'Хорошо', 'Интернет и видео'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[2], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Нормально': 'Хорошо', 'Интернет': 'Видео'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[2], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Отлично', 'Нет'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[3], opacity=0.5)
                st.markdown(s.to_html(table_uuid="table_categories_7_5"), unsafe_allow_html=True)
                categories = '''
                <br>
                <ul>
                <li><span style="color:white;background-color:rgb(31, 119, 180);opacity:0.5;font-size:18px">&nbsp;1&nbsp;</span>&nbsp- Полностью неудовлетворен</li>
                <li><span style="color:white;background-color:rgb(255, 127, 14);opacity:0.5;font-size:18px">&nbsp;2&nbsp;</span>&nbsp- Частично неудовлетворен</li>
                <li><span style="color:white;background-color:rgb(44, 160, 44);opacity:0.5;font-size:18px">&nbsp;3&nbsp;</span> - Частично удовлетворен</li>
                <li><span style="color:white;background-color:rgb(214, 39, 40);opacity:0.5;font-size:18px">&nbsp;4&nbsp;</span>&nbsp- Полностью удовлетворен</li>
                </ul>
                '''
                st.markdown(categories, unsafe_allow_html=True)
            with c2:
                st.markdown(set_text_style(f'<b>Выводы</b>', font_size=32,
                                           color=MegafonColors.orangeDark, text_align='center'),
                            unsafe_allow_html=True)
                conclusion_text = '''
                <ul>
                <li><span style="font-size:24px">Клиенты группы "<b>1</b>" и "<b>2</b>", а также "<b>4</b>" и "<b>5</b>" принадлежат к <b>одной популяции</b>.</span></li>
                <li><span style="font-size:24px">Клиенты группы "<b>2</b>" и "<b>3</b>" принадлежат к <b>одной популяции</b>.</span></li>
                <li><span style="font-size:24px">Самое сильное влияние у метрики "<b>Скорость загрузки потокового видео</b>", а самое слабое у метрики <b>Средняя скорость «к абоненту»</b>.</span></li>
                </ul>
                '''
                st.markdown(conclusion_text, unsafe_allow_html=True)
    case "Влияние метрик на уровень удовлетворенности":
        tabs = ['Цель исследования',
                'Разведочный анализ',
                'Статистические тесты',
                'Выводы']
        tab1, tab2, tab3, tab4 = st.tabs(tabs)
        for tab in tabs:
            set_widget_style(tab, font_size=24)
        with tab1:
            c1, c2 = st.columns(2, gap='medium')
            with c1:
                st.markdown('&nbsp;')
                s = display_cat_info(data_clean)\
                    .set_properties(pd.IndexSlice['Ужасно': 'Плохо', 'Интернет и видео'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[0], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Ужасно': 'Плохо', 'Интернет': 'Видео'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[1], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Нормально': 'Хорошо', 'Интернет и видео'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[2], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Нормально': 'Хорошо', 'Интернет': 'Видео'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[2], opacity=0.5)\
                    .set_properties(pd.IndexSlice['Отлично', 'Нет'], color='white',
                                    background=px.colors.DEFAULT_PLOTLY_COLORS[3], opacity=0.5)
                st.markdown(s.to_html(table_uuid="table_categories_dist"), unsafe_allow_html=True)
                categories = '''
            <br>
            <ul>
            <li><span style="color:white;background-color:rgb(31, 119, 180);opacity:0.5;font-size:18px">&nbsp;1&nbsp;</span> - Полностью неудовлетворены</li>
            <li><span style="color:white;background-color:rgb(255, 127, 14);opacity:0.5;font-size:18px">&nbsp;2&nbsp;</span> - Частично неудовлетворены</li>
            <li><span style="color:white;background-color:rgb(44, 160, 44);opacity:0.5;font-size:18px">&nbsp;3&nbsp;</span> - Частично удовлетворены</li>
            <li><span style="color:white;background-color:rgb(214, 39, 40);opacity:0.5;font-size:18px">&nbsp;4&nbsp;</span> - Полностью удовлетворены</li>
            </ul>
                            '''
                st.markdown(categories, unsafe_allow_html=True)
            with c2:
                st.markdown('&nbsp;')
                goal_text = set_text_style('<b>Цель</b><br>', tag='p', color=MegafonColors.orangeDark, font_size=32,
                                           text_align='center')
                goal_text += set_text_style('Выяснить, является ли разделение статистически верным?',
                                            tag='span', font_size=24)
                st.markdown(goal_text, unsafe_allow_html=True)
                st.markdown('&nbsp;')
                task_text = set_text_style('<b>Вопросы</b><br>', tag='p',
                                           color=MegafonColors.orangeDark, font_size=32, text_align='center')
                task_text += set_text_style(f"""
                <ul>
                <li>{set_text_style('Принадлежат ли клиенты с разными CSAT к разным популяциям?',
                                    font_size=24)}</li>
                <li>{set_text_style('Если клиенты с разными CSAT не принадлежат к одной популяции, '
                                    'то по каким метрикам особенно сильно различаются?', font_size=24)}</li>
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
            st.markdown(set_text_style('<b>Доверительные интервалы статистик</b>', text_align='center',
                                       font_size=32, color=MegafonColors.brandPurple),
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
                    # title=f'<b>Плотность нулевого распределения вероятностей тестовой статистики</b>', title_y=0.9,
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
                    # title=f'<b>Плотность нулевого распределения вероятностей тестовой статистики</b>', title_y=0.9,
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
                    # title=f'<b>Плотность нулевого распределения вероятностей тестовой статистики</b>', title_y=0.9,
                    labels_font_size=16, axes_tickfont_size=14, units_font_size=14,
                    height=300, n_cols=3, opacity=0.5,
                    histnorm='probability density',
                    add_kde=True, add_statistic=True, mark_statistic=mark_statistic,
                    horizontal_spacing=0.08, vertical_spacing=0.07)


            alternatives, pvalues, mark_statistic, null_distributions, statistics = load_8_4_1()

            st.markdown(set_text_style(f'<b>Группы "{groups[0]}" и "{groups[1]}"</b>', text_align='center',
                                       font_size=32, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('<b>Гипотезы</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([40, 30, 30])
            st.markdown('&nbsp;')
            c1.latex('\Delta \hat{TM}=\hat{TM}_{1}-\hat{TM}_{2}')
            c2.latex('H_0:\Delta \hat{TM}≥0')
            c3.latex('H_1:\Delta \hat{TM}<0')
            st.markdown(set_text_style('<b>Плотность нулевого распределения вероятностей тестовой статистики</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            fig = plot_metric_histograms_8_4_1(null_distributions, statistics, research_metrics, mark_statistic)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            group_pair_index = 0
            st.markdown(set_text_style(f'<b>Значения p-value статистического теста</b>', font_size=24,
                                       color=MegafonColors.brandGreenDarken10, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                                metrics=research_metrics, alpha=alpha,
                                caption='', col_width=400)
            st.markdown(s.to_html(table_uuid="table_pvalues_8_4_1"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Выводы:</b>', tag='p',
                                             color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'есть значимые различия</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)

            _, pvalues, mark_statistic, null_distributions, statistics = load_8_4_2()

            st.markdown('---')
            st.markdown(set_text_style(f'<b>Группы "{groups[1]}" и "{groups[2]}"</b>', text_align='center',
                                       font_size=32, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('<b>Гипотезы</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([40, 30, 30])
            st.markdown('&nbsp;')
            c1.latex('\Delta \hat{TM}=\hat{TM}_{2}-\hat{TM}_{3}')
            c2.latex('H_0:\Delta \hat{TM}≥0')
            c3.latex('H_1:\Delta \hat{TM}<0')
            st.markdown(set_text_style('<b>Плотность нулевого распределения вероятностей тестовой статистики</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            fig = plot_metric_histograms_8_4_2(null_distributions, statistics, research_metrics, mark_statistic)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            group_pair_index = 1
            st.markdown(set_text_style(f'<b>Значения p-value статистического теста</b>', font_size=24,
                                       color=MegafonColors.brandGreenDarken10, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                                metrics=research_metrics, alpha=alpha,
                                caption='', col_width=400)
            st.markdown(s.to_html(table_uuid="table_pvalues_8_4_2"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Вывод:</b><br>', tag='p', font_size=24,
                                             color=MegafonColors.brandGreenDarken10, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'нет значимых различий</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)

            _, pvalues, mark_statistic, null_distributions, statistics = load_8_4_3()

            st.markdown('---')
            st.markdown(set_text_style(f'<b>Группы "{groups[2]}" и "{groups[3]}"</b>', text_align='center',
                                       font_size=32, color=MegafonColors.brandPurple),
                        unsafe_allow_html=True)
            st.markdown(set_text_style('<b>Гипотезы</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([40, 30, 30])
            st.markdown('&nbsp;')
            c1.latex('\Delta \hat{TM}=\hat{TM}_{3}-\hat{TM}_{4}')
            c2.latex('H_0:\Delta \hat{TM}≥0')
            c3.latex('H_1:\Delta \hat{TM}<0')
            st.markdown(set_text_style('<b>Плотность нулевого распределения вероятностей тестовой статистики</b>',
                                       color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center'),
                        unsafe_allow_html=True)
            fig = plot_metric_histograms_8_4_3(null_distributions, statistics, research_metrics, mark_statistic)
            st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
            group_pair_index = 2
            st.markdown(set_text_style(f'<b>Значения p-value статистического теста</b>', font_size=24,
                                       color=MegafonColors.brandGreenDarken10, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues.loc[', '.join(group_pairs[group_pair_index])].to_frame().T,
                                metrics=research_metrics, alpha=alpha,
                                caption='', col_width=400)
            st.markdown(s.to_html(table_uuid="table_pvalues_8_4_3"), unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Выводы:</b>', tag='p',
                                             color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'есть значимые различия</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)
            st.markdown('&nbsp;')
            conclusion_text = set_text_style('<b>Выводы:</b>', tag='p',
                                             color=MegafonColors.brandGreenDarken10, font_size=24, text_align='center')
            conclusion_text += set_text_style(f'<b>Между всеми метриками</b> тестовых групп <b>'
                                              f'есть значимые различия</b>', tag='span', font_size=20)
            st.markdown(conclusion_text, unsafe_allow_html=True)
        with tab4:
            st.markdown(set_text_style(f'<b>Значения p-value статистических тестов</b>', font_size=32,
                                       color=MegafonColors.brandPurple, text_align='center'),
                        unsafe_allow_html=True)
            s = display_pvalues(pvalues, metrics=research_metrics, alpha=alpha,
                                caption=f'<b>Значение p-value статистических тестов</b>',
                                opacity=0.5, col_width=180, index_width=30)
            st.markdown(s.to_html(table_uuid="table_pvalues_8_5"), unsafe_allow_html=True)
            st.markdown('---')
            # st.markdown(set_text_style(f'<b>Категории по причинам недовольства сервисом мобильного интернета</b>', font_size=32,
            #                            color=MegafonColors.brandPurple, text_align='center'),
            #             unsafe_allow_html=True)
            # st.markdown('&nbsp;')
            c1, c2 = st.columns(2, gap='medium')
            with c1:
                st.markdown(set_text_style(f'<b>Тренд метрики<br>"'
                                           f'{research_metrics.loc["Video Streaming Download Throughput(Kbps)", "description"]}"</b>',
                                           font_size=32, text_align='center', color=MegafonColors.brandPurple),
                            unsafe_allow_html=True)
                df = ci_center['Video Streaming Download Throughput(Kbps)'].rename('value').to_frame()
                fig = px.scatter(
                    df, x=df.index, y='value',
                    title=' ',
                    labels={'x': '', 'value': 'кбит/с', 'index': 'CSAT'}, trendline="ols")
                fig.update_layout(title_x=0.5, title_y=0.95, title_font_size=14,
                                  width=500, height=350,
                                  margin_t=40, margin_b=0)
                fig.update_traces(hovertemplate='%{x}<br>%{y}<extra></extra>', selector={'mode': 'markers'})
                fig.update_traces(line_dash='dash', selector={'mode': 'lines'})
                fig.update_xaxes(tickmode='array', tickvals=[1, 2, 3, 4])
                st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=False)
                results = px.get_trendline_results(fig).iloc[0, 0]
            with c2:
                st.markdown(set_text_style(f'<b>Выводы</b>', font_size=32,
                                           color=MegafonColors.orangeDark, text_align='center'),
                            unsafe_allow_html=True)
                conclusion_text = '''
                <ul>
                <li><span style="font-size:24px">Клиенты всех групп принадлежат к <b>разным популяциям</b>.</span></li>
                <li><span style="font-size:24px">Самое сильное влияние у метрики "<b>Скорость загрузки потокового видео</b>".</span></li>
                </ul>
                '''
                st.markdown(conclusion_text, unsafe_allow_html=True)


    case "Заключение":
        @st.cache_resource
        def load_9():
            with open('data/8_3.dmp', 'rb') as fp:
                data = load(fp)
            return data

        research_data, groups, group_pairs, ci, ci_overlapping, ci_center = load_9()
        st.markdown(set_text_style(f'<b>Уровни удовлетворенности клиентов (CSAT)<br>'
                                   f'сервисом мобильного интернета</b>', font_size=32, text_align='center',
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
        st.markdown(set_text_style(f'<b>Доверительные интервалы метрики<br>"'
                                   f'{research_metrics.loc["Video Streaming Download Throughput(Kbps)", "description"]}'
                                   f'"</b>', font_size=32, text_align='center', color=MegafonColors.brandPurple),
                    unsafe_allow_html=True)
        table = display_confidence_interval(
            ci['Video Streaming Download Throughput(Kbps)'],
            metrics=research_metrics.loc['Video Streaming Download Throughput(Kbps)'],
            caption='', caption_font_size=12, opacity=0.5, precision=1, index_width=30)
        '851 кбит/с'
        s = display_confidence_interval(ci['Video Streaming Download Throughput(Kbps)'],
                                        metrics=research_metrics.loc['Video Streaming Download Throughput(Kbps)'],
                                        caption='', caption_font_size=12, opacity=0.5, precision=1, index_width=30)
        st.markdown(s.to_html(table_uuid="table_pvalues_9"), unsafe_allow_html=True)
