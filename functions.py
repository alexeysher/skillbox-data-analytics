# Импортируем необходимые библиотеки
import numpy as np
import pandas as pd
from scipy import stats
import plotly.express as px
from plotly.subplots import make_subplots
from auxiliary import MegafonColors
import streamlit as st


def trimean_mod(data, axis=0):
    '''
    Функция возвращает модифицированный тример,
    рассчитываемый как средневзвешенное медианы, 10-го и 90-го персентиля в пропорции 8:1:1.
    Расчет производится вдоль указанной оси выборки данных.

        Параметры:
        ----------
        data : pandas.Series, pandas.DataFrame или numpy.ndarray
            Выборка данных, для которой рассчитывается триммер.

        axis : {0, 1, 'index', 'columns'}, по умолчанию - 0
            Если равно 0 или 'index', то расчёт производится по строкам, 
            если 1 или 'columns', то по столбцам.
            Используется, если data - это pandas.DataFrame или numpy.array

        Возвращаемый результат:
        -----------------------
            Объект типа float, если data - это pandas.Series
            Объект типа pandas.Series, если data - это pandas.DataFrame. Индексы совпадают с индексами data 
                по противоложной выбранной оси
            Объект типа 1d numpy.ndarray, если data - это numpy.ndarray. 

    '''
    if type(data) == pd.Series:
        p10 = data.quantile(0.1)
        p50 = data.median()
        p90 = data.quantile(0.9)
    elif type(data) == pd.DataFrame:
        p10 = data.quantile(0.1, axis=axis)
        p50 = data.median(axis=axis)
        p90 = data.quantile(0.9, axis=axis)
    else:
        p10 = np.quantile(data, 0.1, axis=axis)
        p50 = np.median(data, axis=axis)
        p90 = np.quantile(data, 0.9, axis=axis)

    return (p10 + p50 * 8 + p90) / 10


def trimean_mod_diff(a, b, axis=0):
    '''
    Функция возвращает разницу модифицированных тримеров (подробнее см. 'trimean_mod') двух выборок.

        Параметры:
        ----------
        a, b : pandas.Series, pandas.DataFrame или numpy.ndarray
            Выборки данных, для которых рассчитывается разница модифицированных тримеров.

        axis : {0, 1, 'index', 'columns'}, по умолчанию - 0
            Если равно 0 или 'index', то расчёт производится по строкам, 
            если 1 или 'columns', то по столбцам.
            Используется, если data - это pandas.DataFrame или numpy.array

        Возвращаемый результат:
        -----------------------
            Объект типа float, если data - это pandas.Series
            Объект типа pandas.Series, если data - это pandas.DataFrame. Индексы совпадают с индексами data 
                по противоложной выбранной оси
            Объект типа 1d numpy.ndarray, если data - это numpy.ndarray.
    '''
    return trimean_mod(a, axis=axis) - trimean_mod(b, axis=axis)


def kde(data, n_points=100, special_points=None):
    '''
    Формирует представление Ядерной оценки плотности (ЯОП, англ. Kernel density estimate, KDE) для выборки данных
    одного или нескольких параметров.
    С помощью объекта типа Series может быть переданы данные о выборке значений одного параметра,
                    для которого формируется ЯОП. В этом случае функция также возвращает объект типа Series, содержащего 
                    В случае необходимости формирования представления ЯОП для нескольких параметров,
                    необходимо передавать выборки их значений с помощью объекта типа DataFrame. 
                    При этом выборки данных для параметров должны быть одинаковой длины и распределяться по столбцам.


        Параметры:
        ----------
        data : DataFrame или Series
            Выборка данных, для которой формируется представление ЯОП.

        n_points : int
            Количество основных (расположенных на одинаковом расстоянии) точек в возвращаемом представлении ЯОП.

        special_values : DataFrame или Series
            Дополнительные точки (например, среднее, медиана и границы доверительного интервала)
            для которых должны быть представлены в возвращаемом представлении ЯОП

        Возвращаемый результат:
        -----------------------
            Объект типа DataFrame, содержащий набор данных о представлении ЯОП.
            В случае, если ЯОП формируется для одного параметра, и выборки данных передается с помощью объекта типа Series,
            в результурующим наборе данных присутствует 2 столбца:
                value - значения точек из диапазона;
                pdf - значения ЯОП
    '''
    # Формируем список колонок
    columns = ['value', 'pdf']

    if type(data) is pd.Series:
        # Представление ЯОП формируется для одного параметра
        # Разбиваем диапазон значений параметра в выборке на (n_points-1) равных отрезков
        values = pd.Series(np.linspace(data.min(), data.max(), n_points))
        if special_points is not None:
            values = pd.concat([values, special_points])
        # Делаем заготовку возвращаемого датасета
        result = pd.DataFrame(columns=columns)
    else:
        # Представление ЯОП формируется для нескольких параметров
        # Разбиваем диапазон значений каждого параметра в выборке на (n_points-1) равных отрезков
        values = pd.DataFrame(np.linspace(data.min(), data.max(), n_points),
                              columns=data.columns)
        # Делаем заготовку возвращаемого датасета
        result = pd.DataFrame(
            columns=pd.MultiIndex.from_product([columns, data.columns]))

    # Добавляем в набор "специальные" значения
    if special_points is not None:
        values = pd.concat([values, special_points])

    # Находим значение представления ЯОП для сформированного набора значений параметра(-ов)
    if type(data) is pd.Series:
        # Представление ЯОП формируется для одного параметра
        kde = stats.gaussian_kde(data)
        pdf = kde.pdf(values)
    else:
        # Представление ЯОП формируется для нескольких параметров
        kde = data.apply(lambda s: stats.gaussian_kde(s))
        pdf = data.apply(lambda s: kde[s.name].pdf(values[s.name]))
        pdf.index = values.index

    # Заполняем результрующий датасет
    result.index.name = 'point'
    result['value'] = values  # Крайние точки отрезков
    result['pdf'] = pdf  # Значения ЯОП в крайних точках отрезков

    return result


def my_bootstrap(data, statistic, n_resamples=9999, axis=0):
    '''
    Возвращает распределение заданной статистики для популяции, 
    представленной наблюдаемой выборкой с одной или несколькими метриками,
    с помощью метода bootstrap.

        Параметры:
        ----------
        data : pandas.Series, pandas.DataFrame или numpy.ndarray
            Наблюдаемая выборка данных.

        statistic : function
            Функция, реализующая расчёт статистики по одной метрике

        n_resamples : int
            Количество повторных выборок. По умолчанию - 9999

        Возвращаемый результат:
        -----------------------
            Объект типа pandas.Series длиной n_resamples, если data - это pandas.Series
            Объект типа pandas.DataFrame размерностью по выбранной оси n_resamples, если data - это pandas.DataFrame.
                Размерность и индексы противоположной оси такая же как в `data`.
            Объект типа numpy.ndarray размерностью по выбранной оси n_resamples, если data - это numpy.ndarray. 
                Размерность противоположной оси такая же как в `data`.         
    '''

    def _my_bootstrap_1d(arr_1d, statistic, n_resamples=9999):
        '''
        Возвращает распределение заданной статистики для популяции, представленной наблюдаемой выборкой с одной метрикой,
        с помощью метода bootstrap.

        Параметры:
        ----------
        arr_1d : 1d numpy.ndarray
            Одномерный массив значений метрики в наблюдаемой выборке.

        statistic : function
            Функция, реализующая расчёт статистики

        n_resamples : int
            Количество повторных выборок. По умолчанию - 9999

        Возвращаемый результат:
        -----------------------
            Одномерный массив numpy.ndarray содержащий n_resamples значений статистики.
        '''
        return np.array([statistic(np.random.choice(arr_1d, arr_1d.size)) for index in range(n_resamples + 1)])

    if type(data) == np.ndarray:
        # Выборка - ndarray (одна или несколько метрик)
        # Применяем _my_bootstrap_1d для каждой метрики
        return np.apply_along_axis(_my_bootstrap_1d, axis, data, statistic)
    elif type(data) == pd.Series:
        # Выборка - Series (одна метрика)
        # Применяем к ней _my_bootstrap_1d
        return pd.Series(_my_bootstrap_1d(data.values, statistic, n_resamples), name=data.name)
    else:
        # Выборка - DataFrame (несколько метрик)
        # Применяем _my_bootstrap_1d для значений каждой метрики
        arr = np.apply_along_axis(_my_bootstrap_1d, axis, data.values, statistic)
        # Полученный результат преобразуем в датафрейм
        if axis == 0:
            return pd.DataFrame(arr, columns=data.columns)
        else:
            return pd.DataFrame(arr, index=data.index)
    return result


def permutation_test(data, functions, alternatives=None, n_resamples=9999, random_state=0):
    '''
    Реализует "Перестановочный тест" (permutation test) для двух независимых групп по одной или нескольким метрикам.
    Является оберткой для функции permutation_test из библиотеки scipy.stats.

        Параметры:
        ----------
        data : pandas.Series или pandas.DataFrame
            Набор наблюдаемых выборок. В качестве индексов должны использоваться названия групп.
            В DataFrame метрики должны располагаться по столбцам.

        functions : callable или pandas.Series of callable
            Функция тестовой статистически.
            callable, если выборки - это pandas.Series.
            pandas.Series of callable, если выборки - это pandas.DataFrame. Индексы должны быть названиями метрик,
            т.е. совпадать с названиями столбцов в выборках.

        alternatives : {'two-sided', 'less', 'greater'} или Series of {'two-sided', 'less', 'greater'} или None. По умолчанию None
            Тип теста: 'two-sided' или None - двухсторонний, 'less' - левосторонний, 'greater' - правосторонний
            string, если выборки - это pandas.Series.
            pandas.Series, если выборки - это pandas.DataFrame. Индексы должны быть названиями метрик,
            т.е. совпадать с названиями столбцов в выборках.

        n_resamples : int
            Количество повторных выборок. По умолчанию - 9999

        Возвращаемый результат:
        -----------------------
        pvalue: float или pandas.Series
            Значение p-value. 
            float, если выборки - это pandas.Series
            pandas.Series of float, если выборки - это pandas.DataFrame. Индексы - это названия метрик (столбцов) 
            в наблюдаемых выборках.
        null_distribution : pandas.Series или pandas.DataFrame
            Нулевое распределение тестовой статистики.
            pandas.Series of float, если выборки - это pandas.Series. Количество элементов - n_resamples.
            pandas.DataFrame of float, если выборки - это pandas.DataFrame. Количество строк - n_resamples. 
            Столбцы - названия метрик (столбцов) в наблюдаемых выборках.
        statistic : float или pandas.Series
            Наблюдаемое значение тестовой статистики. 
            float, если выборки - это pandas.Series.
            pandas.Series of float, если выборки - это pandas.DataFrame. Индексы - это названия метрик (столбцов) 
            в наблюдаемых выборках.

    '''

    def _permutation_test_for_1_metric(data, function, alternative=None, n_resamples=9999):
        '''
        Вспомогательная функция, 
        которая реализует "Перестановочный тест" (permutation test) по одной метрике.

        Параметры:
        ----------
        data : pandas.Series
            Набор наблюдаемых выборок. В качестве индексов должны использоваться названия групп.

        functions : callable
            Функция тестовой статистически.

        alternatives : {'two-sided', 'less', 'greater'} или None. По умолчанию None
            Тип теста: 'two-sided' или None - двухсторонний, 'less' - левосторонний, 'greater' - правосторонний
            string, если выборки - это pandas.Series.
            pandas.Series, если выборки - это pandas.DataFrame. Индексы должны быть названиями метрик,
            т.е. совпадать с названиями столбцов в выборках.

        n_resamples : int
            Количество повторных выборок. По умолчанию - 9999

        Возвращаемый результат:
        -----------------------
        pvalue: float
            Значение p-value.
        null_distribution : pandas.Series
            Нулевое распределение тестовой статистики.
        statistic : float
            Наблюдаемое значение тестовой статистики. 
        '''
        # Применяем функцию stats.permutation_test
        # Тип теста - независимый ('independent'), для одной метрики (vectorized=False)
        result = stats.permutation_test([data.loc[group] for group in data.index.unique()],
                                        statistic=function,
                                        permutation_type='independent',
                                        alternative=alternative,
                                        vectorized=False,
                                        n_resamples=n_resamples)
        # Возвращаем результат
        return result.pvalue, pd.Series(result.null_distribution), result.statistic

    # Если выборка содержит данные только одной метрики, 
    # вызываем _permutation_test_for_1_metric и возвращаем результат ее выполнения
    if type(data) == pd.Series:
        return _permutation_test_for_1_metric(data, functions, alternatives, n_resamples)

    # Выборка содержит данные для нескольких метрик
    # Создаем заготовки результата
    pvalues = pd.Series(name='pvalue', index=data.columns, dtype='float')
    null_distributions = pd.DataFrame(columns=data.columns, dtype='float')
    statistics = pd.Series(name='statistic', index=data.columns, dtype='float')
    # Проводим тесты для каждой метрики в выборке:
    # вызываем _permutation_test_for_1_metric и сохраняем результат ее выполнения
    for metric in data.columns:
        pvalues[metric], null_distributions[metric], statistics[metric] = _permutation_test_for_1_metric(data[metric],
                                                                                                         functions[
                                                                                                             metric],
                                                                                                         alternatives[
                                                                                                             metric],
                                                                                                         n_resamples)
    # Возвращаем результат
    return pvalues, null_distributions, statistics


def confidence_interval(data, groups, statistic, confidence_level=0.95, n_resamples=9999):
    '''
    Возвращает доверительный интервал заданной статистики для одной или нескольких метрик
    популяции, представленной наблюдаемой выборкой, с помощью метода bootstrap.
    Является "обёрткой" для функции scipy.stats.bootstrap, которая

        Параметры:
        ----------
        data : pandas.Series, pandas.DataFrame
            Наблюдаемая выборка данных.

        groups : list[str],
            Список исследуемых групп.

        statistic : callable
            Функция, реализующая расчёт статистики по одной метрике.

        n_resamples : int
            Количество повторных выборок. По умолчанию - 9999

        Возвращаемый результат:
        -----------------------
            Объект типа pandas.Series длиной n_resamples, если data - это pandas.Series
            Объект типа pandas.DataFrame размерностью по выбранной оси n_resamples, если data - это pandas.DataFrame.
                Размерность и индексы противоположной оси такая же как в `data`.
            Объект типа numpy.ndarray размерностью по выбранной оси n_resamples, если data - это numpy.ndarray.
                Размерность противоположной оси такая же как в `data`.
            Каждый элемент представляет собой Tupple из границ ДИ.
    '''

    def _confidence_interval(data, statistic, confidence_level=0.95, n_resamples=9999):
        return tuple(
            stats.bootstrap((data.to_numpy(),), statistic=statistic,
                            confidence_level=confidence_level,
                            n_resamples=n_resamples, vectorized=False,
                            method='basic').confidence_interval
        )

    '''
    Возвращает доверительный интервал заданной статистики для одной метрики
    популяции, представленной наблюдаемой выборкой, с помощью метода bootstrap.
    Является "обёрткой" для функции scipy.stats.bootstrap, которая 

        Параметры:
        ----------
        data : pandas.Series
            Наблюдаемая выборка данных.
            
        statistic : callable
            Функция, реализующая расчёт статистики.

        n_resamples : int
            Количество повторных выборок. По умолчанию - 9999

        Возвращаемый результат:
        -----------------------
            Объект типа pandas.Series длиной n_resamples. Каждый элемент представляет собой Tupple из границ ДИ.
    '''

    if type(data) == pd.Series:
        # В выборке данные только для одной метрики
        # Возвращаем Series из ДИ этой метрики для всех групп
        return pd.Series(
            [_confidence_interval(data.loc[group], statistic, confidence_level, n_resamples)
             for group in data.index.unique()],
            name='ci', index=groups, dtype='object')

    # В выборке данные только для нескольких метрик
    # Возвращаем DataFrame из ДИ. Метрики по столбцам.
    result = [np.apply_along_axis(_confidence_interval, 0, data.loc[group],
                                  statistic, confidence_level, n_resamples).tolist()
              for group in data.index.unique()]
    return pd.DataFrame([list(zip(group_result[0], group_result[1])) for group_result in result],
                        index=groups, columns=data.columns, dtype='object')


def confidence_interval_overlapping(confidence_interval_1, confidence_interval_2, metrics):
    '''
    Функция проверяет наличие пересечения двух доверительных интервалов одной или нескольких метрик.

        Параметры:
        ----------
        confidence_interval_1, confidence_interval_2 : Series of Tupple of 2 float
            Доверительные интервалы. Имена - названия популяций.

        metrics : DataFrame
            Информация о метриках. Индексы - названия метрик в confidence_interval_1, confidence_interval_2.

        Возвращаемый результат:
        -----------------------
            Объект типа Series of Boolean.
            Имя - названия популяций через запятую с пробелом
            Индексы - названия метрик (индексы строк в metrics).
            Элементы - результат проверки пересечения
    '''

    def _confidence_interval_overlapping(confidence_interval_1, confidence_interval_2):
        return not ((confidence_interval_1[1] < confidence_interval_2[0]) or
                    (confidence_interval_2[1] < confidence_interval_1[0]))
        '''
        Вспомогательная функция проверяет наличие пересечения доверительных интервалов одной метрики.

            Параметры:
            ----------
            confidence_interval_1, confidence_interval_2 : Tupple of 2 float
                Доверительные интервалы. Имена - названия популяций.

            Возвращаемый результат:
            -----------------------
                False - не перекрываются, True - перекрываются.
        '''

    # Возвращаем Series of Boolean, в котором индексы - это названия метрик.
    # Имя Series - названия популяций через запятую с пробелом
    return pd.Series(
        [_confidence_interval_overlapping(confidence_interval_1[metric], confidence_interval_2[metric])
         for metric in metrics.index],
        index=metrics.index, name=f'{confidence_interval_1.name}, {confidence_interval_2.name}'
    )


def confidence_interval_center_diffs(confidence_interval_1, confidence_interval_2, metrics):
    '''
    Рассчитывает расстояние между центрами двух доверительных интервалов для одной или нескольких метрик.

        Параметры:
        ----------
        confidence_interval_1, confidence_interval_2 : Series of Tupple of 2 float
            Доверительные интервалы. Имена - названия популяций.

        metrics : DataFrame
            Информация о метриках. Индексы - названия метрик в confidence_interval_1, confidence_interval_2.

        Возвращаемый результат:
        -----------------------
            Объект типа Series of Float.
            Имя - названия популяций через запятую с пробелом
            Индексы - названия метрик (индексы строк в metrics).
            Элементы - расстояние между центрами доверительных интервалов
    '''

    def _confidence_interval_center_diff(confidence_interval_1, confidence_interval_2):
        return (confidence_interval_1[0] + confidence_interval_1[1] - confidence_interval_2[0] - confidence_interval_2[
            1]) / 2
        '''
        Вспомогательная функция проверяет наличие пересечения доверительных интервалов одной метрики.

            Параметры:
            ----------
            confidence_interval_1, confidence_interval_2 : Tupple of 2 float
                Доверительные интервалы. Имена - названия популяций.

            Возвращаемый результат:
            -----------------------
                Расстояние между центрами ДИ.
        '''

    # Возвращаем Series of Float, в котором индексы - это названия метрик.
    # Имя Series - названия популяций через запятую с пробелом
    return pd.Series(
        [_confidence_interval_center_diff(confidence_interval_1[metric], confidence_interval_2[metric])
         for metric in metrics.index],
        index=metrics.index, name=f'{confidence_interval_1.name}, {confidence_interval_2.name}'
    )


def confidence_interval_info(data, metrics, groups, group_pairs):
    '''
    Функция:
    - рассчитывает доверительные интервалы (ci);
    - проверяет наличие перекрытий доверительных интервалов (ci_overlapping) заданных пар групп (group_pairs);
    - рассчитывает центры доверительных интервалов (ci_center);
    - рассчитывает расстояние между центрами доверительных интервалов (ci_center) заданных пар групп (group_pairs).

        Параметры:
        ----------
        data : DataFrame или Series
            Наблюдаемая выборка данных.
            При использовании Series может быть передан набор данных только одной метрики.
            При необходимости расчёта ДИ для наборов нескольких метрик одинакового размера
            следует использовать DataFrame. При этом наборы данных метрик должны распологаться
            в отдельных столбцах.

        metrics : DataFrame
            Информация о метриках. Индексы - названия метрик. Столбец 'statistic' - статистическая функция.

        group_pairs : List или Tupple
            Список пар групп групп, для которых проверяется пересечение доверительных интервалов и
            расстояний между центрами доверительных интервалов.

        Возвращаемый результат:
        -----------------------
        ci : Series
            Доверительные интервалы групп. Индексы - названия групп (индексы из data)

        ci_overlapping : DataFrame
            Наличия пересечений доверительные интервалы заданных пар групп.

        ci_center : Series
            Центры доверительных интервалов групп. Индексы - названия групп (индексы из data)

        ci_center_diffs : DataFrame
            Расстояния между центрами доверительных интервалов заданных пар групп.
    '''
    # Расчёт доверительных интервалов статистик
    ci = data.apply(lambda s: confidence_interval(s, groups, statistic=metrics.loc[s.name, 'statistic']))
    # Проверка наличия перекрытия доверительных интервалов статистик заданных пар групп
    ci_overlapping = pd.DataFrame([
        confidence_interval_overlapping(ci.loc[group_pair[0]], ci.loc[group_pair[1]], metrics)
        for group_pair in group_pairs])
    # Расчёт центров доверительных интервалов статистик
    ci_center = ci.applymap(lambda x: (x[0] + x[1]) / 2)
    # Расчёт расстояний между центрами доверительных интервалов статистик заданных пар групп
    ci_center_diffs = pd.DataFrame([
        confidence_interval_center_diffs(ci.loc[group_pair[0]], ci.loc[group_pair[1]], metrics)
        for group_pair in group_pairs])
    return ci, ci_overlapping, ci_center, ci_center_diffs


def display_cat_info(data):
    '''
    Формирует стилизованное табличное представление карты распределения клиентов 
    по категориям оценки качества сервиса мобильного интернета и причин такой оценки.

        Параметры:
        ----------
        data : DataFrame
            Датасет сведений об категориях клиентов. Должен содержать поля 'Internet score' и 'Dissatisfaction reasons'.


        Возвращаемый результат:
        -----------------------
        io.formats.style.Styler
            Стилизованное табличное представление карты распределения клиентов.
    '''
    df = data.groupby(['Internet score', 'Dissatisfaction reasons'], sort=False).size().unstack(
        level=1, fill_value=0)
    df.index.name = None
    df.columns.name = ''
    df = df.applymap(lambda x: x if x > 0 else '-')
    s = df.style.set_table_styles([
        {'selector': 'th:not(.index_name)',
         'props': f'font-size: 16px; color: white; background-color: {MegafonColors.brandPurple80};'},
        {'selector': 'th.col_heading', 'props': 'text-align: center; width: 100px;'},
        {'selector': 'td', 'props': 'text-align: center; font-size: 16px; font-weight: bold;'},
        {'selector': 'th.index_name', 'props': 'border-style: none'}
    ],
       overwrite=False
    )
    s = s.applymap(lambda v:
                   f'color: white; background-color: {MegafonColors.brandGreen};' if v != '-'
                   else f'color: {MegafonColors.spbSky2}; background-color: {MegafonColors.spbSky1};')
    return s


def display_statistics(data, axis=0, metrics=None, precision=1, caption=None, caption_font_size=12,
                       opacity=1.0, index_width=120, col_width=130):
    '''
    Выводит значения статистики для одной или нескольких метрик одной или нескольких популяций 
    в виде стилизованной таблицы с заголовком.
    Лучшие и худщие значения для каждой метрики выделяются зеленым и красным цветом шрифта соответственно.
    Цвет фона названия популяций задается из палитры px.colors.DEFAULT_PLOTLY_COLORS
    по порядку их следования в наборе данных.

        Параметры:
        ----------
        data : DataFrame
            Набор отображаемых значений статистики. 
            По одной оси должны распологаться названия метрик, по другой названия популяций.

        axis : {0, 1}. По умолчанию - 0
            Показывает, что расположено по строкам и столбцам набора данных.
            0 - индексы - это названия популяций, данные метрик распределены по колонкам
            1 - колонки - это названия популяций, данные метрик распределены по строкам

        precision : int. По умолчанию - 4
            Количество знаков после запятой выводимых значений статистики.

        caption : string или None. По умолчанию - None
            Заголовок таблицы

        caption_font_size : int. По умолчанию - 12
            Размер шрифта заголовка таблицы

        opacity : float. По умолчанию - 1.0
            Уровень непрозрачности (от 0.0 до 1.0) фона названия популяций

        index_width : int. По умолчанию - 120
            Ширина колонки индексов

        col_width : int. По умолчанию - 130
            Ширина колонок значений

        Возвращаемый результат:
        -----------------------
            Нет.
    '''

    df = data.copy()
    if axis == 0:
        df.columns = metrics['description']
        df.columns.name = 'Метрика'
        df.index.name = 'Группа'
        positive_subset = pd.IndexSlice[:, metrics.loc[metrics.impact == '+', 'description'].to_list()]
        negative_subset = pd.IndexSlice[:, metrics.loc[metrics.impact == '-', 'description'].to_list()]
    else:
        df.index = metrics['description']
        df.index.name = 'Метрика'
        df.columns.name = 'Группа'
        positive_subset = pd.IndexSlice[metrics.loc[metrics.impact == '+', 'description'].to_list(), :]
        negative_subset = pd.IndexSlice[metrics.loc[metrics.impact == '-', 'description'].to_list(), :]

    style = df.style.applymap_index(lambda
                                        group: f'color: white; background-color:         {px.colors.DEFAULT_PLOTLY_COLORS[df.axes[axis].get_loc(group)]}; opacity: {opacity}',
                                    axis=axis).set_caption(caption).set_table_styles([
        {'selector': 'caption', 'props': f'font-size:{caption_font_size}pt; text-align: center; color: black'},
        {'selector': '.row_heading, td', 'props': f'width: {index_width}px; text-align: center;'},
        {'selector': '.col_heading, td', 'props': f'width: {col_width}px; text-align: center;'}
    ], overwrite=False) \
        .format(precision=precision)

    if len(positive_subset[1]) > 0:
        style = style.highlight_min(props='color: red; font-weight: bold', subset=positive_subset,
                                    axis=axis).highlight_max(props='color: green; font-weight: bold',
                                                             subset=positive_subset, axis=axis)
    if len(negative_subset[1]) > 0:
        style = style.highlight_min(props='color: green; font-weight: bold', subset=negative_subset,
                                    axis=axis).highlight_max(props='color: red; font-weight: bold',
                                                             subset=negative_subset, axis=axis)

    if df.axes[axis].size == 1:
        style = style.hide(axis='index')

    return style


def display_pvalues(data, axis=0, metrics=None, precision=4, alpha=0.05, caption=None, caption_font_size=12,
                    opacity=1.0, index_width=120, col_width=130):
    '''
    Выводит значения p-value для одной или нескольких метрик одного или нескольких тестов 
    в виде стилизованной таблицы с заголовком.
    Значения меньше уровня значимости выделяются красным цветом шрифта.
    Цвет фона названия популяций задается из палитры px.colors.DEFAULT_PLOTLY_COLORS
    по порядку их следования в наборе данных.

        Параметры:
        ----------
        data : DataFrame
            Набор отображаемых значений p-value. 
            По одной оси должны распологаться названия метрик, по другой названия пар популяций через запятую с пробелом.

        axis : {0, 1}. По умолчанию - 0
            Показывает, что расположено по строкам и столбцам набора данных.
            0 - индексы - это названия пар популяций через запятую с пробелом, данные метрик распределены по колонкам
            1 - колонки - это названия пар популяций через запятую с пробелом, данные метрик распределены по строкам

        precision : int. По умолчанию - 4
            Количество знаков после запятой выводимых значений статистики.

        alpha : float. По умолчанию - 0.05
            Уровень значимости.

        caption : string или None. По умолчанию - None
            Заголовок таблицы

        caption_font_size : int. По умолчанию - 12
            Размер шрифта заголовка таблицы

        opacity : float. По умолчанию - 1.0
            Уровень непрозрачности (от 0.0 до 1.0) фона названия популяций

        index_width : int. По умолчанию - 120
            Ширина колонки индексов

        col_width : int. По умолчанию - 130
            Ширина колонок значений

        Возвращаемый результат:
        -----------------------
            Нет.
    '''

    df = data.copy()
    if axis == 0:
        df.index = pd.MultiIndex.from_tuples(df.index.str.split(', ').map(lambda x: tuple(x)), name=[None, None])
        df.columns = pd.Index(metrics['description'].to_list(), name=None)
        groups = pd.Index(df.index.get_level_values(0).to_list() + df.index.get_level_values(1).to_list()
                          ).drop_duplicates()
    else:
        df.columns = pd.MultiIndex.from_tuples(df.columns.str.split(', ').map(lambda x: tuple(x)), name=[None, None])
        df.index = pd.Index(metrics['description'].to_list(), name=None)
        groups = pd.Index(df.columns.get_level_values(0).to_list() + df.columns.get_level_values(1).to_list()
                          ).drop_duplicates()

    style = df.style.applymap_index(lambda
                                        group: f'color: white; background-color:         {px.colors.DEFAULT_PLOTLY_COLORS[groups.get_loc(group)]}; opacity: {opacity}',
                                    axis=axis, level=0).applymap_index(lambda
                                                                           group: f'color: white; background-color:         {px.colors.DEFAULT_PLOTLY_COLORS[groups.get_loc(group)]}; opacity: {opacity}',
                                                                       axis=axis, level=1).set_caption(
        caption).set_table_styles([
        {'selector': 'caption', 'props': f'font-size:{caption_font_size}pt; text-align:center; color:black'},
        {'selector': 'td', 'props': 'text-align: center; border: 1px solid lightgray; border-collapse: collapse;'},
        {'selector': '.row_heading, td', 'props': f'width: {index_width}px; text-align: center;'},
        {'selector': '.col_heading, td', 'props': f'width: {col_width}px; text-align: center;'}
    ], overwrite=False) \
        .applymap_index(lambda s: 'border: 1px solid lightgray; border-collapse: collapse;', axis=0) \
        .applymap_index(lambda s: 'border: 1px solid lightgray; border-collapse: collapse;', axis=1) \
        .format(precision=precision) \
        .highlight_between(right=alpha, inclusive='right', props='color: red; font-weight: bold')

    if df.axes[axis].size == 1:
        style = style.hide(axis='index')

    return style


def display_confidence_interval(values, axis=0, metrics=None, precision=1, caption=None, caption_font_size=12,
                                opacity=1.0, index_width=120, col_width=80):
    '''
    Выводит значения доверительные интервалы для одной или нескольких метрик одной или нескольких популяций 
    в виде стилизованной таблицы с заголовком.
    Для каждого доверительно интервала в отдельной столбце или строке выводится меньшая граница, центр и большая граница.
    Лучшие и худщие значения центра ДИ для каждой метрики выделяются зеленым и красным цветом шрифта соответственно.
    Цвет фона названия популяций задается из палитры px.colors.DEFAULT_PLOTLY_COLORS
    по порядку их следования в наборе данных.

        Параметры:
        ----------
        data : DataFrame
            Набор отображаемых значений p-value. 
            По одной оси должны распологаться названия метрик, по другой названия пар популяций через запятую с пробелом.

        axis : {0, 1}. По умолчанию - 0
            Показывает, что расположено по строкам и столбцам набора данных.
            0 - индексы - это названия пар популяций через запятую с пробелом, данные метрик распределены по колонкам
            1 - колонки - это названия пар популяций через запятую с пробелом, данные метрик распределены по строкам

        precision : int. По умолчанию - 1
            Количество знаков после запятой выводимых значений статистики.

        caption : string или None. По умолчанию - None
            Заголовок таблицы

        caption_font_size : int. По умолчанию - 12
            Размер шрифта заголовка таблицы

        opacity : float. По умолчанию - 1.0
            Уровень непрозрачности (от 0.0 до 1.0) фона названия популяций

        index_width : int. По умолчанию - 120
            Ширина колонки индексов

        col_width : int. По умолчанию - 80
            Ширина колонок значений

        Возвращаемый результат:
        -----------------------
            Нет.
    '''
    df = pd.DataFrame()
    if axis == 0:
        if type(metrics) == pd.DataFrame:
            df = pd.DataFrame(
                columns=pd.MultiIndex.from_product(
                    [metrics['description'], ['Начало', 'Середина', 'Конец']],
                    names=['', '']),
                index=pd.Index(values.index.to_list(), name=None)
            )
            positive_subsets = [pd.IndexSlice[:, (description, 'Середина')]
                                for description in metrics.loc[metrics.impact == '+', 'description']]
            negative_subsets = [pd.IndexSlice[:, (description, 'Середина')]
                                for description in metrics.loc[metrics.impact == '-', 'description']]
        else:
            df = pd.DataFrame(
                columns=pd.Index(
                    ['Начало', 'Середина', 'Конец'],
                    name=''),
                index=pd.Index(values.index.to_list(), name=None)
            )
            positive_subsets = [pd.IndexSlice[:, 'Середина']] if metrics.impact == '+' else []
            negative_subsets = [pd.IndexSlice[:, 'Середина']] if metrics.impact == '-' else []
    else:
        if type(metrics) == pd.DataFrame:
            df = pd.DataFrame(
                index=pd.MultiIndex.from_product(
                    [metrics['description'], ['Начало', 'Середина', 'Конец']],
                    names=['', '']),
                columns=pd.Index(values.index.to_list(), name=None)
            )
            negative_subsets = [pd.IndexSlice[(description, 'Середина'), :]
                                for description in metrics.loc[metrics.impact == '+', 'description']]
            positive_subsets = [pd.IndexSlice[(description, 'Середина'), :]
                                for description in metrics.loc[metrics.impact == '-', 'description']]
        else:
            df = pd.DataFrame(
                index=pd.Index(
                    ['Начало', 'Середина', 'Конец'],
                    name=''),
                columns=pd.Index(values.index.to_list(), name=None)
            )
            positive_subsets = [pd.IndexSlice[:, 'Середина']] if metrics.impact == '+' else []
            negative_subsets = [pd.IndexSlice[:, 'Середина']] if metrics.impact == '-' else []

    if df.columns.nlevels == 2:
        df = df.swaplevel(axis=1)

        df.loc[:, 'Начало'] = values.applymap(lambda x: x[0]).to_numpy()
        df.loc[:, 'Конец'] = values.applymap(lambda x: x[1]).to_numpy()
        df.loc[:, 'Середина'] = (df.loc[:, 'Начало'] + df.loc[:, 'Конец']).to_numpy() / 2

        df = df.swaplevel(axis=1)
    else:
        df.loc[:, 'Начало'] = values.apply(lambda x: x[0]).to_numpy()
        df.loc[:, 'Конец'] = values.apply(lambda x: x[1]).to_numpy()
        df.loc[:, 'Середина'] = (df.loc[:, 'Начало'] + df.loc[:, 'Конец']).to_numpy() / 2

    style = df.style.applymap_index(lambda group: f'''color: white; background-color: 
                            {px.colors.DEFAULT_PLOTLY_COLORS[df.axes[axis].get_loc(group)]}; 
                            opacity: {opacity}''', axis=axis) \
        .set_caption(caption) \
        .set_table_styles([
        {'selector': 'caption', 'props': f'font-size:{caption_font_size}pt; text-align:center; color:black'},
        {'selector': 'td', 'props': 'text-align: center; border: 1px solid lightgray; border-collapse: collapse;'},
        {'selector': '.row_heading, td', 'props': f'width: {index_width}px; text-align: center;'},
        {'selector': '.col_heading, td', 'props': f'width: {col_width}px; text-align: center;'}
    ], overwrite=False) \
        .applymap_index(lambda s: 'border: 1px solid lightgray; border-collapse: collapse;', axis=0) \
        .applymap_index(lambda s: 'border: 1px solid lightgray; border-collapse: collapse;', axis=1) \
        .format(precision=precision)

    for positive_subset in positive_subsets:
        style = style.highlight_min(props=f'color: red; font-weight: bold;',
                                    subset=positive_subset, axis=axis) \
            .highlight_max(props=f'color: green; font-weight: bold;',
                           subset=positive_subset, axis=axis)

    for negative_subset in negative_subsets:
        style = style.highlight_max(props=f'color: red; font-weight: bold;',
                                    subset=negative_subset, axis=axis) \
            .highlight_min(props=f'color: green; font-weight: bold;',
                           subset=negative_subset, axis=axis)

    if df.axes[axis].size == 1:
        style = style.hide(axis='index')

    return style


def display_confidence_interval_overlapping(values, axis=0, metrics=None, caption='', caption_font_size=12,
                                            opacity=1.0, index_width=120, col_width=130):
    df = values.applymap(lambda x: 'Да' if x == 1 else 'Нет')
    if axis == 0:
        df.index = pd.MultiIndex.from_tuples(df.index.str.split(', ').map(lambda x: tuple(x)), name=[None, None])
        df.columns = pd.Index(metrics['description'].to_list(), name=None)
        groups = pd.Index(df.index.get_level_values(0).to_list() + df.index.get_level_values(1).to_list()
                          ).drop_duplicates()
    else:
        df.columns = pd.MultiIndex.from_tuples(df.columns.str.split(', ').map(lambda x: tuple(x)), name=[None, None])
        df.index = pd.Index(metrics['description'].to_list(), name=None)
        groups = pd.Index(df.columns.get_level_values(0).to_list() + df.columns.get_level_values(1).to_list()
                          ).drop_duplicates()

    style = df.style.applymap_index(lambda
                                        group: f'color: white; background-color: {px.colors.DEFAULT_PLOTLY_COLORS[groups.get_loc(group)]}; opacity: {opacity}',
                                    axis=axis, level=0).applymap_index(lambda
                                                                           group: f'color: white; background-color: {px.colors.DEFAULT_PLOTLY_COLORS[groups.get_loc(group)]}; opacity: {opacity}',
                                                                       axis=axis, level=1).set_caption(
        caption).set_table_styles([
        {'selector': 'caption', 'props': f'font-size:{caption_font_size}pt; text-align:center; color:black'},
        {'selector': 'td', 'props': 'text-align: center; border: 1px solid lightgray; border-collapse: collapse;'},
        {'selector': '.row_heading, td', 'props': f'width: {index_width}px; text-align: center;'},
        {'selector': '.col_heading, td', 'props': f'width: {col_width}px; text-align: center;'}
    ], overwrite=False) \
        .applymap_index(lambda s: 'border: 1px solid lightgray; border-collapse: collapse;', axis=0) \
        .applymap_index(lambda s: 'border: 1px solid lightgray; border-collapse: collapse;', axis=1) \
        .applymap(lambda x: f'color: red; font-weight: bold;' if x == 'Нет' else None)

    if df.axes[axis].size == 1:
        style = style.hide(axis='index')

    return style


def plot_metric_histograms(data, metrics, title=None, title_y=None, yaxis_title=None,
                           title_font_size=14, labels_font_size=12, units_font_size=12, axes_tickfont_size=12,
                           height=None, width=None, horizontal_spacing=None, vertical_spacing=None,
                           n_cols=1, opacity=0.5, histnorm='percent',
                           add_boxplot=False, boxplot_height_fraq=0.25, add_mean=False, add_kde=False,
                           mark_confidence_interval=False, confidence_level=0.95,
                           add_statistic=False, mark_statistic=None, statistic=None):
    '''
    Функция предназначена для построения гистограмм для нескольких метрик, разбитых на несколько групп.
    Для каждой метрики создается отдельное полотно. Полотно можно распередить на несколько столбцов.
    В каждом полотне строится несколько гистограмм для каждой группы.
    Дополнительно над гистограммами можно разместить boxplot'ы для каждой группы,
    аналогично тому как это делает функция px.histogram при укзании параметра margin равным boxplot. 
    Еще одной опцией является возможность построения ядерных оценок распределения (KDE) 
    для каждой группы на одном полотне с гистограммами. Построение KDE возможно только при построении
    гистограмм плотности вероятности (probability density).

    Параметры:
    ----------
    data : DataFrame или Series 
        Выборка данных, для которой рассчитываются границы доверительного интервала.
        При использовании Series может быть передан набор данных только одной метрики.
        При необходимости расчёта ДИ для наборов нескольких метрик одинакового размера
        следует использовать DataFrame. При этом наборы данных метрик должны распологаться
        в отдельных столбцах.

    metrics : DataFrame
        Информация о метриках. Индексы - названия метрик.

    title : string или None. По умолчанию - None
        Заголовок диаграммы

    title_y : float или None
        Относительное положение заголовока диаграммы по высоте (от 0.0 (внизу) до 1.0 (вверху))

    yaxis_title : string или None. По умолчанию - None
        Заголовок оси y

    title_font_size : int. По умолчанию - 14
        Размер шрифта заголовка диаграммы

    labels_font_size : int. По умолчанию - 12
        Размер шрифта надписей

    units_font_size : int. По умолчанию - 12
        Размер шрифта названий единиц измерения

    axes_tickfont_size : int. По умолчанию - 12
        Размер шрифта меток на осях

    height : int или None. По умолчанию - None
        Высота диаграммы

    width : int или None. По умолчанию - None
        Ширина диаграммы

    horizontal_spacing : float или None. По умолчанию - None
        Расстояние между в столбцами полотен в долях от ширины (от 0.0 до 1.0)

    vertical_spacing : float или None. По умолчанию - None
        Расстояние между в строка полотен в долях от высоты (от 0.0 до 1.0)

    n_cols : int. По умолчанию - 1
        Количество столбцов полотен

    opacity : float. По умолчанию - 0.5
        Уровень непрозрачности (от 0.0 до 1.0) цвета столбцов

    histnorm : {'percent', 'probability', 'density' или 'probability density'} или None. По умолчанию - 'percent'
        Тип гистораммы (см. plotly.express.histogram)

    boxplot_height_fraq : float. По умолчанию - 0.25
        Доля высоты boxplot. Используется только, если add_boxplot=True

    add_boxplot : boolean. По умолчанию - False
        Добавить boxplot над каждой гистограммой.

    add_mean : boolean. По умолчанию - False
        Добавить на boxlot отметку среднего значения. Используется только, если add_boxplot=True

    add_kde : boolean. По умолчанию - False
        Добавить на гистограмму кривую KDE. Используется только, если histnorm='probability density'

    mark_confidence_interval : boolean. По умолчанию - False
        Закрасить области KDE, вне доверительного интервала в цвет гистограммы с половинной прозрачностью.
        Используется только, если add_kde=True

    confidence_level : float. По умолчанию - 0.95
        Уровень доверия. Используется только, если mark_confidence_interval=True

    add_statistic : boolean. По умолчанию - False
        Отметить статистику на гистограмму в виде вертикальной штриховой линии.

    mark_statistic : {'tomin', 'tomax', 'tonearest'}. По умолчанию - False
        Закрасить область KDE слева ('tomin') или справа ('tomax'), 
        или минимальную ('min') или максимальную ('max') по размеру в цвет гистограммы с половинной прозрачностью.

    statistic : Series
        Значение статистики, отображаемой на гистограмме. Индексы - названия метрик
        Используется, только если add_statistic=True и/или mark_statistic=True

    Возвращаемый результат:
    -----------------------
        Нет.
    '''

    def _confidence_interval(data, confidence_level=0.95):
        '''
        Рассчитывает границы доверительного интервала набора данных

        Параметры:
        ----------
        data : Series или DataFrame
            Выборка данных, для которой рассчитываются границы доверительного интервала.
            При использовании Series может быть передан набор данных только одной метрики.
            При необходимости расчёта ДИ для наборов нескольких метрик одинакового размера
            следует использовать DataFrame. При этом наборы данных метрик должны распологаться
            в отдельных столбцах.

        confidence_level : float. По умолчанию - 0.95
            Уровень доверия.

        Возвращаемый результат:
        -----------------------
            Если data - это Series, то Series с двумя элементами: 'low' - нижняя граница, 'high' - верхняя граница.
            Если data - это DataFrame, то DataFrame с двумя строками: 'low' - нижняя граница, 'high' - верхняя граница.
        '''
        alpha = 1 - confidence_level
        result = data.quantile([alpha / 2, 1 - alpha / 2])
        result = result.rename({alpha / 2: 'low', 1 - alpha / 2: 'high'})
        return result

    # Список метрик - это названия колонок в датасете
    n_metrics = metrics.shape[0]
    # Вычисляем количество строк и их высоты
    n_rows = int(np.ceil(n_metrics / n_cols))
    if add_boxplot:
        row_heights = [boxplot_height_fraq / n_rows, (1 - boxplot_height_fraq) / n_rows] * n_rows
        n_rows *= 2
    else:
        row_heights = [1 / n_rows] * n_rows
    titles = []
    specs = []
    # Формируем список заголовков и спецификаций графиков
    for index in range(0, n_metrics, n_cols):
        titles += metrics['label'].iloc[index:index + n_cols].to_list()
        if add_boxplot:
            titles += [''] * n_cols
            specs.append([{'b': 0.004}] * n_cols)
        specs.append([{'b': vertical_spacing}] * n_cols)
    # Создаем полотно с n_row*n_cols графиков
    fig = make_subplots(cols=n_cols, rows=n_rows, row_heights=row_heights, subplot_titles=titles,
                        horizontal_spacing=horizontal_spacing, vertical_spacing=0,
                        specs=specs)
    # Отображаем гистограммы метрик, располагая над ними и \"ящики с усами\"
    for index, metric in enumerate(metrics.index):
        # Идем по метрикам
        col = index % n_cols + 1
        row = (index // n_cols) * (2 if add_boxplot else 1) + 1
        # Добавляем гистограмму
        fig.add_histogram(x=data[metric], row=row + (1 if add_boxplot else 0), col=col, histnorm=histnorm,
                          bingroup=index + 1,
                          marker_color=px.colors.DEFAULT_PLOTLY_COLORS[index],
                          marker_line_color='white', marker_line_width=1,
                          opacity=opacity, showlegend=False, name=metrics.loc[metric, 'description'])
        # К гистограмме добавляем KDE
        if add_kde and histnorm == 'probability density':
            special_points = None
            if mark_confidence_interval:
                confidence_interval = _confidence_interval(data[metric], confidence_level)
                special_points = confidence_interval
            if mark_statistic is not None:
                if special_points is None:
                    special_points = pd.Series(statistic[metric])
                else:
                    special_points = special_points.append(
                        pd.Series(statistic[metric]))
            metric_kde = kde(data[metric],
                             special_points=special_points)
            metric_kde.sort_values(['value'], inplace=True)
            fig.add_scatter(x=metric_kde['value'], y=metric_kde['pdf'], row=row + (1 if add_boxplot else 0), col=col,
                            mode='lines', marker_color=px.colors.DEFAULT_PLOTLY_COLORS[index], marker_line_width=1,
                            opacity=opacity, showlegend=False, name=metrics.loc[metric, 'description'])
            if mark_confidence_interval:
                df = metric_kde[metric_kde['value'] <= confidence_interval['low']]
                fig.add_scatter(x=df['value'], y=df['pdf'], row=row + (1 if add_boxplot else 0), col=col, mode='lines',
                                marker_color=px.colors.DEFAULT_PLOTLY_COLORS[index], marker_line_width=1,
                                opacity=opacity, name=metrics.loc[metric, 'description'],
                                showlegend=False, fill='tozeroy')
                df = metric_kde[metric_kde['value'] >= confidence_interval['high']]
                fig.add_scatter(x=df['value'], y=df['pdf'], row=row + (1 if add_boxplot else 0), col=col, mode='lines',
                                marker_color=px.colors.DEFAULT_PLOTLY_COLORS[index], marker_line_width=1,
                                opacity=opacity, name=metrics.loc[metric, 'description'],
                                showlegend=False, fill='tozeroy')
            if mark_statistic is not None and statistic is not None:
                if mark_statistic[metric] == 'tomin':
                    df = metric_kde[metric_kde['value'] <= statistic[metric]]
                elif mark_statistic[metric] == 'tomax':
                    df = metric_kde[metric_kde['value'] >= statistic[metric]]
                elif mark_statistic[metric] == 'min':
                    if sum(data[metric] <= statistic[metric]) <= sum(data[metric] >= statistic[metric]):
                        df = metric_kde[metric_kde['value'] <= statistic[metric]]
                    else:
                        df = metric_kde[metric_kde['value'] >= statistic[metric]]
                elif mark_statistic[metric] == 'max':
                    if sum(data[metric] <= statistic[metric]) >= sum(data[metric] >= statistic[metric]):
                        df = metric_kde[metric_kde['value'] >= statistic[metric]]
                    else:
                        df = metric_kde[metric_kde['value'] <= statistic[metric]]
                fig.add_scatter(x=df['value'], y=df['pdf'], row=row + (1 if add_boxplot else 0), col=col, mode='lines',
                                marker_color=px.colors.DEFAULT_PLOTLY_COLORS[index], marker_line_width=1,
                                opacity=opacity, name=metrics.loc[metric, 'description'],
                                showlegend=False, fill='tozeroy')
            if add_statistic and statistic is not None:
                # Добавляем статистику
                fig.add_vline(x=statistic[metric], row=row + (1 if add_boxplot else 0), col=col,
                              line_color=px.colors.DEFAULT_PLOTLY_COLORS[index], line_width=2, line_dash='dash',
                              opacity=opacity)
        if add_boxplot:
            # Добавляем \"ящик с усами\" над гистограммой
            fig.add_box(x=data[metric], row=row, col=col, marker_color=px.colors.DEFAULT_PLOTLY_COLORS[index],
                        line_width=1, name=metrics.loc[metric, 'description'],
                        boxmean=add_mean, showlegend=False)
            # У \"ящиков с усами\" устанавливаем  такой же диапазон значений по оси x как у гисторграмм,
            # показываем сетку по оси x, но скрываем на ней метки
            fig.update_xaxes(matches=list(fig.select_traces(row=row + 1, col=col))[0].xaxis,
                             showgrid=True, showticklabels=False, row=row, col=col)
            # У \"ящиков с усами\" скрываем название оси y и метки на ней
            fig.update_yaxes(title='', row=row, col=col, showticklabels=False)
        fig.update_xaxes(title=metrics['units'].iloc[index], titlefont_size=units_font_size,
                         row=row + (1 if add_boxplot else 0), col=col)
        fig.update_yaxes(title=yaxis_title, titlefont_size=units_font_size,
                         row=row + (1 if add_boxplot else 0), col=col)

    fig.update_xaxes(tickfont_size=axes_tickfont_size)
    fig.update_yaxes(tickfont_size=axes_tickfont_size)
    fig.update_annotations(font_size=labels_font_size)
    fig.update_layout(barmode='overlay',
                      # title=title, title_font_size=title_font_size,
                      # title_x=0.5, title_y=title_y, title_xanchor='center',
                      width=width, height=height,
                      margin_l=0, margin_r=0, margin_t=60, margin_b=60)
    return fig


def plot_metric_confidence_interval(data, metrics, title=None, title_y=None, yaxis_title=None,
                                    title_font_size=14, labels_font_size=12, units_font_size=12, axes_tickfont_size=12,
                                    height=None, width=None, horizontal_spacing=None, vertical_spacing=None,
                                    n_cols=1, opacity=0.5):
    '''
    Функция предназначена для построения доверительных интервалов для нескольких метрик нескольких групп.
    Для каждой метрики создается отдельное полотно. Полотно можно распередить на несколько столбцов.
    Доверительный интевал указывается в виде горизонтального отрезка с вертикальными отсечками на концах.
    Центр доверительного интервала выделяется точкой.

        Параметры:
        ----------
        data : pandas.Series или pandas.DataFrame
            Конфиденциальные интервалы популяций.
            pandas.Series - для построения ДИ для одной метрики
            pandas.DataFrame - для построения ДИ для нескольких метрик. Данные метрик располагаются в столбцах. 
            В качестве индекса датасета должны использоваться названия популяций.

        metrics : DataFrame
            Информация о метриках. Индексы - названия метрик.

        title : string или None. По умолчанию - None
            Заголовок диаграммы

        title_y : float или None
            Относительное положение заголовока диаграммы по высоте (от 0.0 (внизу) до 1.0 (вверху))

        title_font_size : int. По умолчанию - 14
            Размер шрифта заголовка диаграммы

        labels_font_size : int. По умолчанию - 12
            Размер шрифта надписей

        units_font_size : int. По умолчанию - 12
            Размер шрифта названий единиц измерения

        axes_tickfont_size : int. По умолчанию - 12
            Размер шрифта меток на осях

        height : int или None. По умолчанию - None
            Высота диаграммы

        width : int или None. По умолчанию - None
            Ширина диаграммы

        horizontal_spacing : float или None. По умолчанию - None
            Расстояние между в столбцами полотен в долях от ширины (от 0.0 до 1.0)

        vertical_spacing : float или None. По умолчанию - None
            Расстояние между в строка полотен в долях от высоты (от 0.0 до 1.0)

        n_cols : int. По умолчанию - 1
            Количество столбцов полотен

        opacity : float. По умолчанию - 0.5
            Уровень непрозрачности (от 0.0 до 1.0) цвета столбцов

        Возвращаемый результат:
        -----------------------
            Нет.
    '''
    # Вычисляем количество строк и их высоты
    n_rows = int(np.ceil(metrics.shape[0] / n_cols))
    row_heights = [1 / n_rows] * n_rows
    titles = []
    specs = []
    # Формируем список заголовков и спецификаций графиков
    for index in range(0, metrics.shape[0], n_cols):
        titles += (metrics.iloc[index:index + n_cols, :]['label'].to_list())
        if vertical_spacing:
            specs.append([{'b': vertical_spacing}] * n_cols)
        else:
            specs.append([{}] * n_cols)
    # Создаем полотно с n_row*n_cols графиков
    fig = make_subplots(cols=n_cols, rows=n_rows, row_heights=row_heights,
                        subplot_titles=titles, specs=specs,
                        horizontal_spacing=horizontal_spacing, vertical_spacing=0.004)
    # Отображаем точечный график метрик, располагая над ними и \"ящики с усами\"
    for index, metric in enumerate(metrics.index):
        # Идем по метрикам
        col = index % n_cols + 1
        row = index // n_cols + 1
        if type(data) == pd.Series:
            # Добавляем точечный график на полотно
            fig.add_scatter(x=[(data[metric][0] + data[metric][1]) / 2], y=[0],
                            error_x={'type': 'constant', 'value': abs(data[metric][0] - data[metric][1]) / 2},
                            row=row, col=col, name='',
                            marker_color=px.colors.DEFAULT_PLOTLY_COLORS[index],
                            marker_line_color='white', marker_line_width=1,
                            opacity=opacity, showlegend=False)
        else:
            # Добавляем точечный для каждой группы
            for group_index, group in enumerate(data.index.unique()):
                fig.add_scatter(x=[(data.loc[group, metric][0] + data.loc[group, metric][1]) / 2], y=[-group_index],
                                error_x={'type': 'constant',
                                         'value': abs(data.loc[group, metric][0] - data.loc[group, metric][1]) / 2},
                                customdata=[data.loc[group, metric]],
                                row=row, col=col, name=group,
                                marker_color=px.colors.DEFAULT_PLOTLY_COLORS[group_index],
                                marker_line_color='white', marker_line_width=1, marker_size=10, opacity=opacity,
                                showlegend=index == 0, legendgroup=group,
                                hovertemplate='%{x:.1f}, (%{customdata[0]:.1f}, %{customdata[1]:.1f})')
        fig.update_xaxes(title=metrics.loc[metric, 'units'], titlefont_size=units_font_size, row=row, col=col)
        fig.update_yaxes(title=yaxis_title, titlefont_size=units_font_size, row=row, col=col)
    fig.update_xaxes(tickfont_size=axes_tickfont_size)
    fig.update_yaxes(visible=False, tickfont_size=axes_tickfont_size)
    fig.update_annotations(font_size=labels_font_size, y=1.1)
    fig.update_layout(title=title, title_y=title_y, title_font_size=title_font_size, title_x=0.5,
                      width=width, height=height, margin_l=10, margin_r=10, margin_t=60, margin_b=60,
                      legend_font_size=labels_font_size, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def plot_group_size_barchart(data, title=None, title_y=None, title_font_size=14, opacity=0.5, orientation='h',
                             labels_font_size=12, xaxis_title=None, yaxis_title=None,
                             axes_title_font_size=12, axes_tickfont_size=12,
                             height=None, width=None):
    '''
    Функция строит столбчатую диаграмму, отображающую размеры групп, представленных в наборе данных.

        Параметры:
        ----------
        data : DataFrame
            Набор данных. В качестве индекса датасета должны использоваться названия групп.

        title : string или None. По умолчанию - None
            Заголовок диаграммы

        title_y : float или None
            Относительное положение заголовока диаграммы по высоте (от 0.0 (внизу) до 1.0 (вверху))

        title_font_size : int. По умолчанию - 14
            Размер шрифта заголовка диаграммы

        opacity : float. По умолчанию - 0.5
            Уровень непрозрачности (от 0.0 до 1.0) цвета столбцов

        orientation : {'h', 'v'}. По умолчанию - 'h'
            Ориентация диаграммы: 'h'-горизонтальная, 'v'-вертикальная

        labels_font_size : int. По умолчанию - 12
            Размер шрифта надписей

        xaxis_title : string или None. По умолчанию - None
            Заголовок оси x

        yaxis_title : string или None. По умолчанию - None
            Заголовок оси y

        axes_title_font_size : int. По умолчанию - 12
            Размер шрифта названий осей

        axes_tickfont_size : int. По умолчанию - 12
            Размер шрифта меток на осях

        height : int или None. По умолчанию - None
            Высота диаграммы

        width : int или None. По умолчанию - None
            Ширина диаграммы

        Возвращаемый результат:
        -----------------------
            Нет.
    '''

    # Строим столбчатую диаграмму
    # Если диаграмма располагается горизонтально, то меняем порядок индексов на обратный
    df = data.index if orientation == 'v' else data.index[::-1]
    colors = px.colors.DEFAULT_PLOTLY_COLORS[:df.nunique()]
    if orientation == 'h':
        colors.reverse()
    fig = px.histogram(df, title=title, opacity=opacity, orientation=orientation, height=height, width=width)
    fig.update_traces(texttemplate="%{x}", hovertemplate='%{y} - %{x:} клиентов',
                      marker_color=colors, showlegend=False)
    fig.update_layout(bargap=0.2, boxgroupgap=0.2,
                      title_font_size=title_font_size,
                      title_x=0.5, title_y=title_y,
                      margin_t=60, margin_b=60)
    fig.update_xaxes(title=xaxis_title, title_font_size=axes_title_font_size, tickfont_size=axes_tickfont_size)
    fig.update_yaxes(title=yaxis_title, title_font_size=axes_title_font_size, tickfont_size=axes_tickfont_size)
    fig.update_annotations(font_size=labels_font_size)
    fig.show()