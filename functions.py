import numpy as np
import pandas as pd
from scipy import stats
import plotly.express as px
from plotly.subplots import make_subplots
from auxiliary import MegafonColors


def trimean_mod(data, axis=0):
    """
    '''
    Returns modified trimmer,
    obtained weighted average of 10th, 50th and 90th percentile in 1:8:1 proportion.
    Calculation is performing on the given axis of data sample.

        Parameters:
        ----------
        data : pandas.Series, pandas.DataFrame or numpy.ndarray
            The given data sample.

        axis : {0, 1, 'index', 'columns'}, default - 0
            If 0 or 'index' the calculation is performed on rows.
            If 1 or 'columns' the calculation is performed on columns.
            Is used if data is pandas.DataFrame or numpy.array

        Returns:
        -----------------------
            Float type value if data is pandas.Series
            Pandas.Series with index of opposite axis of data if data is pandas.DataFrame .
            1d numpy.ndarray if data is numpy.ndarray.
    """
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
    """
    Returns a difference between modified trimmers for the two given data samples.

        Parameters:
        ----------
        a, b : pandas.Series, pandas.DataFrame or numpy.ndarray
            The given data samples.

        axis  : {0, 1, 'index', 'columns'}, default - 0
            If 0 or 'index' the calculation is performed on rows.
            If 1 or 'columns' the calculation is performed on columns.
            Is used if data is pandas.DataFrame or numpy.array

        Returns:
        -----------------------
            Float type value if data is pandas.Series
            Pandas.Series with index of opposite axis of data if data is pandas.DataFrame .
            1d numpy.ndarray if data is numpy.ndarray.
    """
    return trimean_mod(a, axis=axis) - trimean_mod(b, axis=axis)


def kde(data, n_points=100, special_points=None):
    """
    Generates a Kernel Density Estimate (KDE) representation for a sample of data of one or more parameters.
    A Series object can be used to pass data on a sample of values of one parameter,
    for which the KDE is generated. In this case, the function also returns a Series object containing
    If it is necessary to generate a KDE representation for several parameters,
    it is necessary to pass samples of their values using a DataFrame object.
    In this case, the data samples for the parameters must be of the same length and distributed across columns.

        Parameters:
        ----------
        data : DataFrame or Series
            The given data sample

        n_points : int
            The number of points in the returned LOP representation.

        special_values : DataFrame or Series
            Additional points (e.g. mean, median, and confidence interval bounds)
            that should be represented in the returned KDE representation

        Returns:
        -----------------------
            A DataFrame object containing a set of data about the KDE representation.
            If the LOP is formed for one parameter, and the data selection is transferred using a Series object,
            the resulting data set contains 2 columns:
                value - the values of the points from the range;
                pdf - the KDE values
    """
    # Forming a list of columns
    columns = ['value', 'pdf']

    if type(data) is pd.Series:
        # The LOP representation is generated for one parameter
        # We divide the range of parameter values in the sample into (n_points-1) equal segments
        values = pd.Series(np.linspace(data.min(), data.max(), n_points))
        if special_points is not None:
            values = pd.concat([values, special_points])
        # We prepare the returned dataset
        result = pd.DataFrame(columns=columns)
    else:
        # The LOP representation is generated for several parameters
        # We divide the range of values of each parameter in the sample into (n_points-1) equal segments
        values = pd.DataFrame(np.linspace(data.min(), data.max(), n_points),
                              columns=data.columns)
        # We prepare the returned dataset
        result = pd.DataFrame(
            columns=pd.MultiIndex.from_product([columns, data.columns]))

    # Add "special" values to the set
    if special_points is not None:
        values = pd.concat([values, special_points])

    # Find the value of the LOP representation for the generated set of parameter(s) values
    if type(data) is pd.Series:
        # The LOP representation is generated for one parameter
        kde = stats.gaussian_kde(data)
        pdf = kde.pdf(values)
    else:
        # The LOP representation is generated for several parameters
        kde = data.apply(lambda s: stats.gaussian_kde(s))
        pdf = data.apply(lambda s: kde[s.name].pdf(values[s.name]))
        pdf.index = values.index

    # Fill the resulting dataset
    result.index.name = 'point'
    result['value'] = values # Endpoints of segments
    result['pdf'] = pdf # Values of the LOP at the extreme points of the segments

    return result


def my_bootstrap(data, statistic, n_resamples=9999, axis=0):
    """
    Returns the distribution of the given statistic for a population,
    represented by an observed sample with one or more metrics,
    using the bootstrap method.

        Parameters:
        ----------
        data : pandas.Series, pandas.DataFrame or numpy.ndarray
            Observed data sample.

        statistic : function
            A function that implements the calculation of statistics for one metric

        n_resamples : int
            Number of resamples. Default is 9999

        Returns:
        -----------------------
            A pandas.Series object of length n_resamples if data is a pandas.Series
            A pandas.DataFrame object of type n_resamples along the selected axis, if data is a pandas.DataFrame.
                The dimensions and indices of the opposite axis are the same as in `data`.
            An object of type numpy.ndarray with dimensions along the selected axis n_resamples, if data is a numpy.ndarray.
                The dimensions of the opposite axis are the same as in `data`.
    """

    def _my_bootstrap_1d(arr_1d, statistic, n_resamples=9999):
        """
        Returns the distribution of a given statistic for a population represented by an observed sample with a single metric,
        using the bootstrap method.

        Parameters:
        ----------
        arr_1d : 1d numpy.ndarray
            A one-dimensional array of metric values in the observed sample.

        statistic : function
            Function implementing the calculation of statistics

        n_resamples : int
            Number of resamples. Default is 9999

        Returns:
        -----------------------
            A one-dimensional numpy.ndarray array containing n_resamples statistics values.
        """
        return np.array([statistic(np.random.choice(arr_1d, arr_1d.size)) for index in range(n_resamples + 1)])

    if type(data) == np.ndarray:
        # Sample - ndarray (one or more metrics)
        # Apply _my_bootstrap_1d for each metric
        return np.apply_along_axis(_my_bootstrap_1d, axis, data, statistic)
    elif type(data) == pd.Series:
        # Sample - Series (one metric)
        # Apply _my_bootstrap_1d to it
        return pd.Series(_my_bootstrap_1d(data.values, statistic, n_resamples), name=data.name)
    else:
        # Selection - DataFrame (multiple metrics)
        # Apply _my_bootstrap_1d to each metric values
        arr = np.apply_along_axis(_my_bootstrap_1d, axis, data.values, statistic)
        # We transform the obtained result into a dataframe
        if axis == 0:
            return pd.DataFrame(arr, columns=data.columns)
        else:
            return pd.DataFrame(arr, index=data.index)
    return result


def permutation_test(data, functions, alternatives=None, n_resamples=9999, random_state=0):
    """
    Implements a "permutation test" for two independent groups on one or more metrics.
    It is a wrapper for the permutation_test function from the scipy.stats library.

        Parameters:
        ----------
        data : pandas.Series or pandas.DataFrame
            A set of observed samples. Group names should be used as indices.
            In a DataFrame, metrics must be arranged in columns.

        functions : callable or pandas.Series of callable
            Test function statistically.
            callable if samples are pandas.Series.
            pandas.Series of callable if samples are pandas.DataFrame. Indexes should be metric names,
            i.e. match the column names in the samples.

        alternatives : {'two-sided', 'less', 'greater'} or Series of {'two-sided', 'less', 'greater'} or None. Default None
            Test type: 'two-sided' or None - two-sided, 'less' - left-sided, 'greater' - right-sided
            string if samples are pandas.Series.
            pandas.Series if samples are pandas.DataFrame. Indexes should be metric names,
            i.e. match the column names in the samples.

        n_resamples : int
            Number of resamples. Default is 9999

        Returns:
        -----------------------
        pvalue: float or pandas.Series
            p-value meaning.
            float if samples are pandas.Series
            pandas.Series of float if samples are pandas.DataFrame. Indexes are names of metrics (columns)
            in the observed samples.
        null_distribution : pandas.Series or pandas.DataFrame
            Null distribution of test statistics.
            pandas.Series of float if samples are pandas.Series. Number of elements is n_resamples.
            pandas.DataFrame of float if samples are a pandas.DataFrame. Number of rows is n_resamples.
            Columns are the names of the metrics (columns) in the observed samples.
        statistic : float or pandas.Series
            The observed value of the test statistic.
            float if samples are pandas.Series.
            pandas.Series of float if samples are pandas.DataFrame. Indexes are names of metrics (columns)
            in the observed samples.
    """

    def _permutation_test_for_1_metric(data, function, alternative=None, n_resamples=9999):
        """
        Auxiliary function,
        which implements the "permutation test" for one metric.

        Parameters:
        ----------
        data : pandas.Series
            A set of observed samples. Group names should be used as indices.

        functions : callable
            Test function statistically.

        alternatives : {'two-sided', 'less', 'greater'} or None. Defaults to None
            Test type: 'two-sided' or None - two-sided, 'less' - left-sided, 'greater' - right-sided
            string if samples are pandas.Series.
            pandas.Series if samples are pandas.DataFrame. Indexes should be metric names,
            i.e. match the column names in the samples.

        n_resamples : int
            Number of resamples. Default is 9999

        Returns:
        -----------------------
        pvalue: float
            p-value meaning.
        null_distribution : pandas.Series
            Null distribution of test statistics.
        statistic : float
            The observed value of the test statistic.
        """

        # Apply the stats.permutation_test function
        # Test type - independent ('independent'), for one metric (vectorized=False)
        result = stats.permutation_test([data.loc[group] for group in data.index.unique()],
                                        statistic=function,
                                        permutation_type='independent',
                                        alternative=alternative,
                                        vectorized=False,
                                        n_resamples=n_resamples)
        # Return the result
        return result.pvalue, pd.Series(result.null_distribution), result.statistic

    # If the sample contains data for only one metric,
    # call _permutation_test_for_1_metric and return the result of its execution
    if type(data) == pd.Series:
        return _permutation_test_for_1_metric(data, functions, alternatives, n_resamples)

    # The sample contains data for several metrics
    # Create result templates
    pvalues = pd.Series(name='pvalue', index=data.columns, dtype='float')
    null_distributions = pd.DataFrame(columns=data.columns, dtype='float')
    statistics = pd.Series(name='statistic', index=data.columns, dtype='float')
    # We run tests for each metric in the sample:
    # call _permutation_test_for_1_metric and save the result of its execution
    for metric in data.columns:
        pvalues[metric], null_distributions[metric], statistics[metric] = \
        _permutation_test_for_1_metric(data[metric], functions[metric], alternatives[metric], n_resamples)
    # Return the result
    return pvalues, null_distributions, statistics


def confidence_interval(data, groups, statistic, confidence_level=0.95, n_resamples=9999):
    """
    Returns the confidence interval of the specified statistic for one or more metrics.
    the population represented by the observed sample using the bootstrap method.
    It is a "wrapper" for the scipy.stats.bootstrap function, which
    
        Parameters:
        ----------
        data : pandas.Series, pandas.DataFrame
            Observed data sample.
            
        statistic : callable
            A function that implements the calculation of statistics for one metric.
                    
        n_resamples : int
            Number of resamples. Default is 9999
                    
        Returns:
        -----------------------
            A pandas.Series object of length n_resamples if data is a pandas.Series
            A pandas.DataFrame object of type n_resamples along the selected axis, if data is a pandas.DataFrame.
                The dimensions and indices of the opposite axis are the same as in `data`.
            An object of type numpy.ndarray with dimensions along the selected axis n_resamples, if data is a numpy.ndarray.
                The dimensions of the opposite axis are the same as in `data`.
            Each element is a Tuple of DI boundaries.
    """

    def _confidence_interval(data, statistic, confidence_level=0.95, n_resamples=9999):
        return tuple(
            stats.bootstrap((data.to_numpy(),), statistic=statistic,
                            confidence_level=confidence_level,
                            n_resamples=n_resamples, vectorized=False,
                            method='basic').confidence_interval
        )

    '''
    Returns the confidence interval of the given statistic for a single metric
    the population represented by the observed sample using the bootstrap method.
    It is a "wrapper" for the scipy.stats.bootstrap function, which
    
        Parameters:
        ----------
        data : pandas.Series
            Observed data sample.
            
        statistic : callable
            A function that implements the calculation of statistics.
                    
        n_resamples : int
            Number of resamples. Default is 9999
                    
        Returns:
        -----------------------
            A pandas.Series object of length n_resamples. Each element is a Tuple of CI bounds.
    '''

    if type(data) == pd.Series:
        # The sample contains data for only one metric
        # Return Series from DI of this metric for all groups
        return pd.Series(
            [_confidence_interval(data.loc[group], statistic, confidence_level, n_resamples)
             for group in data.index.unique()],
            name='ci', index=groups, dtype='object')

    # The sample contains data for only a few metrics
    # Return DataFrame from DI. Metrics by columns.
    result = [np.apply_along_axis(_confidence_interval, 0, data.loc[group],
                                  statistic, confidence_level, n_resamples).tolist()
              for group in data.index.unique()]
    return pd.DataFrame([list(zip(group_result[0], group_result[1])) for group_result in result],
                        index=groups, columns=data.columns, dtype='object')


def confidence_interval_info(data, metrics, groups, group_pairs):
    """
    Function:
    - calculates confidence intervals (ci);
    - checks for overlapping confidence intervals (ci_overlapping) of the specified pairs of groups (group_pairs);
    - calculates confidence interval centers (ci_center);
    - calculates the distance between the centers of the confidence intervals (ci_center) of the given pairs of groups (group_pairs).

        Parameters:
        ----------
        data : DataFrame or Series
            Observed data sample.
            When using Series, only a single metric dataset can be passed.
            When it is necessary to calculate CI for sets of several metrics of the same size
            DataFrame should be used. In this case, the metrics data sets should be located
            in separate columns.

        metrics : DataFrame
            Information about metrics. Indexes - names of metrics. Column 'statistic' - statistical function.

        group_pairs : List or Tuple
            List of pairs of groups of groups for which the intersection of confidence intervals is tested and
            distances between the centers of confidence intervals.

        Returns:
        -----------------------
        ci : Series
            Confidence intervals of groups. Indexes - names of groups (indices from data)

        ci_overlapping : DataFrame
            The presence of intersections of confidence intervals of given pairs of groups.

        ci_center : Series
            Centers of confidence intervals of groups. Indexes - names of groups (indices from data)

        ci_center_diffs : DataFrame
            Distances between the centers of confidence intervals of given pairs of groups.
    """
    # Calculating confidence intervals of statistics
    ci = data.apply(lambda s: confidence_interval(s, groups, statistic=metrics.loc[s.name, 'statistic']))
    # Checking for overlapping confidence intervals of statistics of given pairs of groups
    ci_overlapping = pd.DataFrame([
        confidence_interval_overlapping(ci.loc[group_pair[0]], ci.loc[group_pair[1]], metrics)
        for group_pair in group_pairs])
    # Calculation of confidence interval centers of statistics
    ci_center = ci.map(lambda x: (x[0] + x[1])/2)
    # Calculation of distances between centers of confidence intervals of statistics of given pairs of groups
    ci_center_diffs = pd.DataFrame([
        confidence_interval_center_diffs(ci.loc[group_pair[0]], ci.loc[group_pair[1]], metrics)
        for group_pair in group_pairs])
    return ci, ci_overlapping, ci_center, ci_center_diffs


def confidence_interval_overlapping(confidence_interval_1, confidence_interval_2, metrics):
    """
    The function checks for the intersection of two confidence intervals of one or more metrics.

        Parameters:
        ----------
        confidence_interval_1, confidence_interval_2 : Series of Tuple of 2 float
            Confidence intervals. Names are names of populations.

        metrics : DataFrame
            Information about metrics. Indexes - names of metrics in confidence_interval_1, confidence_interval_2.

        Returns:
        -----------------------
            An object of type Series of Boolean.
            Name - names of populations separated by commas and spaces
            Indexes are the names of metrics (row indices in metrics).
            Elements - result of intersection check
    """

    def _confidence_interval_overlapping(confidence_interval_1, confidence_interval_2):
        """
        The helper function checks for the intersection of confidence intervals of one metric.

            Parameters:
            ----------
            confidence_interval_1, confidence_interval_2 : Tuple of 2 float
                Confidence intervals. Names are names of populations.

            Returns:
            -----------------------
                False - do not overlap, True - overlap.
        """
        return not ((confidence_interval_1[1] < confidence_interval_2[0]) or
                    (confidence_interval_2[1] < confidence_interval_1[0]))

    # Return a Series of Boolean, where the indices are the names of the metrics.
    # Series name - population names separated by commas and spaces
    return pd.Series(
        [_confidence_interval_overlapping(confidence_interval_1[metric], confidence_interval_2[metric])
         for metric in metrics.index],
        index=metrics.index, name=f'{confidence_interval_1.name}, {confidence_interval_2.name}'
    )


def confidence_interval_center_diffs(confidence_interval_1, confidence_interval_2, metrics):
    """
    Calculates the distance between the centers of two confidence intervals for one or more metrics.

        Parameters:
        ----------
        confidence_interval_1, confidence_interval_2 : Series of Tuple of 2 float
            Confidence intervals. Names are names of populations.

        metrics : DataFrame
            Information about metrics. Indexes - names of metrics in confidence_interval_1, confidence_interval_2.

        Returns:
        -----------------------
            An object of type Series of Float.
            Name - names of populations separated by commas and spaces
            Indexes are the names of metrics (row indices in metrics).
            Elements - distance between centers of confidence intervals
    """

    def _confidence_interval_center_diff(confidence_interval_1, confidence_interval_2):
        """
        The helper function checks for the intersection of confidence intervals of one metric.

           Parameters:
           ----------
           confidence_interval_1, confidence_interval_2 : Tuple of 2 float
               Confidence intervals. Names are names of populations.

           Returns:
           -----------------------
               Distance between the centers of the DI.
        """
        return (confidence_interval_1[0] + confidence_interval_1[1] - confidence_interval_2[0] - confidence_interval_2[
            1]) / 2

    # Return a Series of Boolean, where the indices are the names of the metrics.
    # Series name - population names separated by commas and spaces
    return pd.Series(
        [_confidence_interval_center_diff(confidence_interval_1[metric], confidence_interval_2[metric])
         for metric in metrics.index],
        index=metrics.index, name=f'{confidence_interval_1.name}, {confidence_interval_2.name}'
    )


def display_cat_info(data):
    """
    Generates a stylized tabular representation of the customer distribution map
    by categories of mobile Internet service quality assessment and reasons for such assessment.

        Parameters:
        ----------
        data : DataFrame
            Dataset of information about customer categories. Should contain the fields 'Internet score' and 'Dissatisfaction reasons'.


        Returns:
        -----------------------
        io.formats.style.Styler
            A stylized tabular representation of a customer distribution map.
    """
    df = data.groupby(['Internet score', 'Dissatisfaction reasons'], sort=False).size().unstack(
        level=1, fill_value=0)
    df.index.name = None
    df.columns.name = ''
    df = df.map(lambda x: x if x > 0 else '-')
    s = df.style.set_table_styles(
        [
            {'selector': 'th:not(.index_name)',
             'props': f'font-weight: normal; color: white; background-color: {MegafonColors.brandPurple80};'},
            {'selector': 'th.col_heading', 'props': 'text-align: center; width: 9.3rem;'},
            {'selector': 'td', 'props': 'text-align: center; font-weight: normal;'},
            {'selector': 'th.index_name', 'props': 'border-top-style: hidden; border-left-style: hidden'}
        ], overwrite=False
    )
    s = s.map(lambda v:
              f'color: white; background-color: {MegafonColors.brandGreen};' if v != '-'
              else f'color: {MegafonColors.spbSky2}; background-color: {MegafonColors.spbSky1};')
    return s


def display_statistics(data, axis=0, metrics=None, precision=1, caption=None, caption_font_size=12,
                       opacity=1.0, index_width=120, col_width=130):
    """
    Outputs statistics values for one or more metrics from one or more populations
    in the form of a stylized table with a heading.
    The best and worst values for each metric are highlighted in green and red font colors, respectively.
    The background color of the population name is set from the px.colors.DEFAULT_PLOTLY_COLORS palette
    in the order in which they appear in the data set.

        Parameters:
        ----------
        data : DataFrame
            The set of statistics values to display.
            The names of the metrics should be located on one axis, and the names of the populations on the other.

        axis: {0, 1}. Default - 0
            Shows what is located in the rows and columns of a data set.
            0 - indices are the names of populations, metric data are distributed across columns
            1 - columns are the names of populations, metric data are distributed across rows

        precision : int. Default - 4
            The number of decimal places for the output statistics values.

        caption : string or None. Default is None
            Table Header

        caption_font_size : int. Default - 12
            Table Header Font Size

        opacity : float. Default is 1.0
            Opacity level (from 0.0 to 1.0) of the population name background

        index_width : int. Default - 120
            Index column width

        col_width : int. Default - 130
            Width of value columns

        Returns:
        -----------------------
            No.
    """

    df = data.copy()
    if axis == 0:
        df.columns = metrics['name']
        df.columns.name = 'Metric'
        df.index.name = 'Group'
        positive_subset = pd.IndexSlice[:, metrics.loc[metrics.impact == '+', 'name'].to_list()]
        negative_subset = pd.IndexSlice[:, metrics.loc[metrics.impact == '-', 'name'].to_list()]
    else:
        df.index = metrics['name']
        df.index.name = 'Metric'
        df.columns.name = 'Group'
        positive_subset = pd.IndexSlice[metrics.loc[metrics.impact == '+', 'name'].to_list(), :]
        negative_subset = pd.IndexSlice[metrics.loc[metrics.impact == '-', 'name'].to_list(), :]

    style = df.style.map_index(
        lambda group: f'color: white; '
                      f'background-color: {px.colors.DEFAULT_PLOTLY_COLORS[df.axes[axis].get_loc(group)]}; '
                      f'opacity: {opacity}', axis=axis
    ).set_caption(caption).set_table_styles(
        [
            {'selector': 'caption', 'props': f'font-size:{caption_font_size}pt; color: black'},
            {'selector': '.row_heading, td', 'props': f'width: {index_width}px; font-weight: normal;'},
            {'selector': '.col_heading, td', 'props': f'width: {col_width}px; text-align: center; font-weight: normal;'},
            {'selector': 'th.index_name', 'props': 'border-top-style: hidden; border-left-style: hidden'}
        ], overwrite=False).format(precision=precision)

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
    """
    Outputs p-values for one or more metrics from one or more tests.
    in the form of a stylized table with a heading.
    Values below the significance level are highlighted in red font.
    The background color of the population name is set from the px.colors.DEFAULT_PLOTLY_COLORS palette
    in the order in which they appear in the data set.

        Parameters:
        ----------
        data : DataFrame
            The set of displayed p-values.
            The names of the metrics should be located on one axis, and the names of the pairs of populations, separated by a comma and a space, should be located on the other.

        axis: {0, 1}. Default - 0
            Shows what is located in the rows and columns of a data set.
            0 - indices are the names of population pairs separated by commas and spaces, metric data are distributed across columns
            1 - columns are the names of population pairs separated by commas and spaces, metric data are distributed across rows

        precision : int. Default - 4
            The number of decimal places for the output statistics values.

        alpha : float. Default is 0.05
            Level of significance.

        caption : string or None. Default is None
            Table Header

        caption_font_size : int. Default - 12
            Table Header Font Size

        opacity : float. Default is 1.0
            Opacity level (from 0.0 to 1.0) of the population name background

        index_width : int. Default - 120
            Index column width

        col_width : int. Default - 130
            Width of value columns

        Returns:
        -----------------------
            No.
    """

    df = data.copy()
    if axis == 0:
        df.index = pd.MultiIndex.from_tuples(df.index.str.split(', ').map(lambda x: tuple(x)), name=[None, None])
        df.columns = pd.Index(metrics['name'].to_list(), name=None)
        groups = pd.Index(df.index.get_level_values(0).to_list() + df.index.get_level_values(1).to_list()
                          ).drop_duplicates()
    else:
        df.columns = pd.MultiIndex.from_tuples(df.columns.str.split(', ').map(lambda x: tuple(x)), name=[None, None])
        df.index = pd.Index(metrics['name'].to_list(), name=None)
        groups = pd.Index(df.columns.get_level_values(0).to_list() + df.columns.get_level_values(1).to_list()
                          ).drop_duplicates()

    style = df.style.map_index(
        lambda group: f'color: white; '
                      f'background-color: {px.colors.DEFAULT_PLOTLY_COLORS[groups.get_loc(group)]}; '
                      f'opacity: {opacity}', axis=axis, level=0
    ).map_index(
        lambda group: f'color: white; '
                      f'background-color: {px.colors.DEFAULT_PLOTLY_COLORS[groups.get_loc(group)]}; '
                      f'opacity: {opacity}', axis=axis, level=1
    ).set_caption(
        caption
    ).set_table_styles(
        [
            {'selector': 'caption', 'props': f'font-size:{caption_font_size}pt; text-align:center; color:black'},
            # {'selector': 'td', 'props': 'text-align: center; border: 1px solid lightgray; border-collapse: collapse;'},
            {'selector': '.row_heading, td', 'props': f'width: {index_width}px; font-weight: normal;'},
            {'selector': '.col_heading, td', 'props': f'width: {col_width}px; text-align: center; font-weight: normal;'},
            {'selector': 'th.blank', 'props': 'border-top-style: hidden; border-left-style: hidden'}
        ], overwrite=False
    ).format(
        precision=precision
    ).highlight_between(right=alpha, inclusive='right', props='color: red;')

    if df.axes[axis].size == 1:
        style = style.hide(axis='index')

    return style


def display_confidence_interval(values, axis=0, metrics=None, precision=1, caption=None, caption_font_size=12,
                                opacity=1.0, index_width=120, col_width=80):
    '''
    Outputs confidence interval values for one or more metrics in one or more populations.
    in the form of a stylized table with a heading.
    For each confidence interval, the minor boundary, the center, and the major boundary are displayed in a separate column or row.
    The best and worst values of the CI center for each metric are highlighted in green and red font colors, respectively.
    The background color of the population name is set from the px.colors.DEFAULT_PLOTLY_COLORS palette
    in the order in which they appear in the data set.

        Parameters:
        ----------
        data : DataFrame
            The set of displayed p-values.
            The names of the metrics should be located on one axis, and the names of the pairs of populations, separated by a comma and a space, should be located on the other.

        axis: {0, 1}. Default - 0
            Shows what is located in the rows and columns of a data set.
            0 - indices are the names of population pairs separated by commas and spaces, metric data are distributed across columns
            1 - columns are the names of population pairs separated by commas and spaces, metric data are distributed across rows

        precision : int. Default - 1
            The number of decimal places for the output statistics values.

        caption : string or None. Default is None
            Table Header

        caption_font_size : int. Default - 12
            Table Header Font Size

        opacity : float. Default is 1.0
            Opacity level (from 0.0 to 1.0) of the population name background

        index_width : int. Default - 120
            Index column width

        col_width : int. Default - 80
            Width of value columns

        Returns:
        -----------------------
            No.
    '''
    df = pd.DataFrame()
    if axis == 0:
        if type(metrics) == pd.DataFrame:
            df = pd.DataFrame(
                columns=pd.MultiIndex.from_product(
                    [metrics['name'], ['Low bound', 'Midpoint', 'Hi bound']],
                    names=['', '']),
                index=pd.Index(values.index.to_list(), name=None)
            )
            positive_subsets = [pd.IndexSlice[:, (description, 'Midpoint')]
                                for description in metrics.loc[metrics.impact == '+', 'name']]
            negative_subsets = [pd.IndexSlice[:, (description, 'Midpoint')]
                                for description in metrics.loc[metrics.impact == '-', 'name']]
        else:
            df = pd.DataFrame(
                columns=pd.Index(
                    ['Low bound', 'Midpoint', 'Hi bound'],
                    name=''),
                index=pd.Index(values.index.to_list(), name=None)
            )
            positive_subsets = [pd.IndexSlice[:, 'Midpoint']] if metrics.impact == '+' else []
            negative_subsets = [pd.IndexSlice[:, 'Midpoint']] if metrics.impact == '-' else []
    else:
        if type(metrics) == pd.DataFrame:
            df = pd.DataFrame(
                index=pd.MultiIndex.from_product(
                    [metrics['name'], ['Low bound', 'Midpoint', 'Hi bound']],
                    names=['', '']),
                columns=pd.Index(values.index.to_list(), name=None)
            )
            negative_subsets = [pd.IndexSlice[(description, 'Midpoint'), :]
                                for description in metrics.loc[metrics.impact == '+', 'name']]
            positive_subsets = [pd.IndexSlice[(description, 'Midpoint'), :]
                                for description in metrics.loc[metrics.impact == '-', 'name']]
        else:
            df = pd.DataFrame(
                index=pd.Index(
                    ['Low bound', 'Midpoint', 'Hi bound'],
                    name=''),
                columns=pd.Index(values.index.to_list(), name=None)
            )
            positive_subsets = [pd.IndexSlice[:, 'Midpoint']] if metrics.impact == '+' else []
            negative_subsets = [pd.IndexSlice[:, 'Midpoint']] if metrics.impact == '-' else []

    if df.columns.nlevels == 2:
        df = df.swaplevel(axis=1)

        df.loc[:, 'Low bound'] = values.map(lambda x: x[0]).to_numpy()
        df.loc[:, 'Hi bound'] = values.map(lambda x: x[1]).to_numpy()
        df.loc[:, 'Midpoint'] = (df.loc[:, 'Low bound'] + df.loc[:, 'Hi bound']).to_numpy() / 2

        df = df.swaplevel(axis=1)
    else:
        df.loc[:, 'Low bound'] = values.apply(lambda x: x[0]).to_numpy()
        df.loc[:, 'Hi bound'] = values.apply(lambda x: x[1]).to_numpy()
        df.loc[:, 'Midpoint'] = (df.loc[:, 'Low bound'] + df.loc[:, 'Hi bound']).to_numpy() / 2

    style = df.style.map_index(
        lambda group: f'color: white; '
                      f'background-color: {px.colors.DEFAULT_PLOTLY_COLORS[df.axes[axis].get_loc(group)]}; '
                      f'opacity: {opacity}', axis=axis
    ).set_caption(
        caption
    ).set_table_styles(
        [
            {'selector': 'caption', 'props': f'font-size:{caption_font_size}pt; text-align:center;'},
            {'selector': '.row_heading, td', 'props': f'width: {index_width}px; font-weight: normal;'},
            {'selector': '.col_heading, td', 'props': f'width: {col_width}px; text-align: center; font-weight: normal;'},
            {'selector': 'th.index_name', 'props': 'border-top-style: hidden; border-left-style: hidden'}
        ], overwrite=False
    ).format(precision=precision)

    for positive_subset in positive_subsets:
        style = style \
            .highlight_min(props=f'color: red;',
                           subset=positive_subset, axis=axis) \
            .highlight_max(props=f'color: green;',
                           subset=positive_subset, axis=axis)

    for negative_subset in negative_subsets:
        style = style \
            .highlight_max(props=f'color: red; font-weight: bold;',
                           subset=negative_subset, axis=axis) \
            .highlight_min(props=f'color: green; font-weight: bold;',
                           subset=negative_subset, axis=axis)

    if df.axes[axis].size == 1:
        style = style.hide(axis='index')

    return style


def display_confidence_interval_overlapping(values, axis=0, metrics=None, caption='', caption_font_size=12,
                                            opacity=1.0, index_width=120, col_width=130):
    df = values.map(lambda x: 'Yes' if x == 1 else 'No')
    if axis == 0:
        df.index = pd.MultiIndex.from_tuples(df.index.str.split(', ').map(lambda x: tuple(x)), name=[None, None])
        df.columns = pd.Index(metrics['name'].to_list(), name=None)
        groups = pd.Index(df.index.get_level_values(0).to_list() + df.index.get_level_values(1).to_list()
                          ).drop_duplicates()
    else:
        df.columns = pd.MultiIndex.from_tuples(df.columns.str.split(', ').map(lambda x: tuple(x)), name=[None, None])
        df.index = pd.Index(metrics['name'].to_list(), name=None)
        groups = pd.Index(df.columns.get_level_values(0).to_list() + df.columns.get_level_values(1).to_list()
                          ).drop_duplicates()

    style = df.style.map_index(
        lambda group: f'color: white; '
                      f'background-color: {px.colors.DEFAULT_PLOTLY_COLORS[groups.get_loc(group)]}; '
                      f'opacity: {opacity}', axis=axis, level=0
    ).map_index(
        lambda group: f'color: white; '
                      f'background-color: {px.colors.DEFAULT_PLOTLY_COLORS[groups.get_loc(group)]}; '
                      f'opacity: {opacity}', axis=axis, level=1
    ).set_caption(
        caption
    ).set_table_styles(
        [
            {'selector': 'caption', 'props': f'font-size:{caption_font_size}pt; text-align:center;'},
            {'selector': '.row_heading, td', 'props': f'width: {index_width}px; font-weight: normal;'},
            {'selector': '.col_heading, td', 'props': f'width: {col_width}px; text-align: center; font-weight: normal;'},
            {'selector': 'th.blank', 'props': 'border-top-style: hidden; border-left-style: hidden'}
        ], overwrite=False
    ).map(lambda x: f'color: red;' if x == 'No' else None)

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
    """
    The function is designed to build histograms for several metrics, divided into several groups.
    A separate canvas is created for each metric. The canvas can be divided into several columns.
    In each canvas, several histograms are built for each group.
    Additionally, boxplots for each group can be placed above the histograms,
    similar to what the px.histogram function does when specifying the margin parameter equal to boxplot.
    Another option is the ability to construct kernel distribution estimates (KDE)
    for each group on one canvas with histograms. KDE construction is only possible when building
    probability density histogram.

    Parameters:
    ----------
    data : DataFrame or Series
        The data sample for which confidence interval bounds are calculated.
        When using Series, only a single metric dataset can be passed.
        When it is necessary to calculate CI for sets of several metrics of the same size
        DataFrame should be used. In this case, the metrics data sets should be located
        in separate columns.

    metrics : DataFrame
        Information about metrics. Indexes are names of metrics.

    title : string or None. Default is None
        Chart Title

    title_y : float or None
        Relative position of the chart title by height (from 0.0 (bottom) to 1.0 (top))

    yaxis_title : string or None. Defaults to None
        Y-axis title

    title_font_size : int. Default - 14
        Chart Title Font Size

    labels_font_size : int. Default - 12
        Font size of inscriptions

    units_font_size : int. Default - 12
        Font size of unit names

    axes_tickfont_size : int. Default - 12
        Font size of axes labels

    height : int or None. Default is None
        Height of the chart

    width : int or None. Default is None
        Diagram width

    horizontal_spacing : float or None. Defaults to None
        Distance between columns of canvases in fractions of width (from 0.0 to 1.0)

    vertical_spacing : float or None. Defaults to None
        Distance between the rows of canvases in fractions of the height (from 0.0 to 1.0)

    n_cols : int. Default - 1
        Number of columns of canvases

    opacity : float. Default is 0.5
        Opacity level (0.0 to 1.0) of the column color

    histnorm : {'percent', 'probability', 'density' or 'probability density'} or None. Defaults to 'percent'
        Histogram type (see plotly.express.histogram)

    boxplot_height_fraq : float. Default is 0.25
        Fraction of boxplot height. Only used if add_boxplot=True

    add_boxplot : boolean. Default is False
        Add a boxplot above each histogram.

    add_mean : boolean. Default is False
        Add a mean value marker to the boxlot. Only used if add_boxplot=True

    add_kde : boolean. Default is False
        Add KDE curve to histogram. Only used if histnorm='probability density'

    mark_confidence_interval : boolean. Default is False
        Color KDE regions outside the confidence interval with the histogram color at half transparency.
        Only used if add_kde=True

    confidence_level : float. Default is 0.95
        Confidence level. Only used if mark_confidence_interval=True

    add_statistic : boolean. Default is False
        Mark the statistics on the histogram as a vertical dashed line.

    mark_statistic : {'tomin', 'tomax', 'tonearest'}. Default - False
        Color the KDE area on the left ('tomin') or right ('tomax'),
        or minimum ('min') or maximum ('max') in size in the histogram color with half transparency.

    statistic : Series
        The value of the statistics displayed on the histogram. Indexes are the names of the metrics.
        Used only if add_statistic=True and/or mark_statistic=True

    Returns:
    -----------------------
        No.
    """

    def _confidence_interval(data, confidence_level=0.95):
        """
        Calculates the confidence interval bounds of a data set

        Parameters:
        ----------
        data : Series or DataFrame
            The data sample for which confidence interval bounds are calculated.
            When using Series, only a single metric dataset can be passed.
            When it is necessary to calculate CI for sets of several metrics of the same size
            DataFrame should be used. In this case, the metrics data sets should be located
            in separate columns.

        confidence_level : float. Default is 0.95
            Level of trust.

        Returns:
        -----------------------
            If data is a Series, then a Series with two elements: 'low' is the lower bound, 'high' is the upper bound.
            If data is a DataFrame, then a DataFrame with two rows: 'low' is the lower bound, 'high' is the upper bound.
        """

        alpha = 1 - confidence_level
        result = data.quantile([alpha / 2, 1 - alpha / 2])
        result = result.rename({alpha / 2: 'low', 1 - alpha / 2: 'high'})
        return result

    # The list of metrics is the names of the columns in the dataset
    n_metrics = metrics.shape[0]
    # Calculate the number of lines and their heights
    n_rows = int(np.ceil(n_metrics / n_cols))
    if add_boxplot:
        row_heights = [boxplot_height_fraq / n_rows, (1 - boxplot_height_fraq) / n_rows] * n_rows
        n_rows *= 2
    else:
        row_heights = [1 / n_rows] * n_rows
    titles = []
    specs = []
    # Generate a list of titles and chart specifications
    for index in range(0, n_metrics, n_cols):
        titles += metrics['name'].iloc[index:index + n_cols].to_list()
        if add_boxplot:
            titles += [''] * n_cols
            specs.append([{'b': 0.004}] * n_cols)
        specs.append([{'b': vertical_spacing}] * n_cols)
    # Create a canvas with n_row*n_cols graphs
    fig = make_subplots(cols=n_cols, rows=n_rows, row_heights=row_heights, subplot_titles=titles,
                        horizontal_spacing=horizontal_spacing, vertical_spacing=0,
                        specs=specs)
    # Display metrics histograms with boxes and whiskers above them
    for index, metric in enumerate(metrics.index):
        # We go by metrics
        col = index % n_cols + 1
        row = (index // n_cols) * (2 if add_boxplot else 1) + 1
        # Add a histogram
        fig.add_histogram(x=data[metric], row=row + (1 if add_boxplot else 0), col=col, histnorm=histnorm,
                          bingroup=index + 1,
                          marker_color=px.colors.DEFAULT_PLOTLY_COLORS[index],
                          marker_line_color='white', marker_line_width=1,
                          opacity=opacity, showlegend=False, name=metrics.loc[metric, 'name'])
        # Add KDE to the histogram
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
                            opacity=opacity, showlegend=False, name=metrics.loc[metric, 'name'])
            if mark_confidence_interval:
                df = metric_kde[metric_kde['value'] <= confidence_interval['low']]
                fig.add_scatter(x=df['value'], y=df['pdf'], row=row + (1 if add_boxplot else 0), col=col, mode='lines',
                                marker_color=px.colors.DEFAULT_PLOTLY_COLORS[index], marker_line_width=1,
                                opacity=opacity, name=metrics.loc[metric, 'name'],
                                showlegend=False, fill='tozeroy')
                df = metric_kde[metric_kde['value'] >= confidence_interval['high']]
                fig.add_scatter(x=df['value'], y=df['pdf'], row=row + (1 if add_boxplot else 0), col=col, mode='lines',
                                marker_color=px.colors.DEFAULT_PLOTLY_COLORS[index], marker_line_width=1,
                                opacity=opacity, name=metrics.loc[metric, 'name'],
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
                                opacity=opacity, name=metrics.loc[metric, 'name'],
                                showlegend=False, fill='tozeroy')
            if add_statistic and statistic is not None:
                # Add statistics
                fig.add_vline(x=statistic[metric], row=row + (1 if add_boxplot else 0), col=col,
                              line_color=px.colors.DEFAULT_PLOTLY_COLORS[index], line_width=2, line_dash='dash',
                              opacity=opacity)
        if add_boxplot:
            # Add a "box with whiskers" above the histogram
            fig.add_box(x=data[metric], row=row, col=col, marker_color=px.colors.DEFAULT_PLOTLY_COLORS[index],
                        line_width=1, name=metrics.loc[metric, 'name'],
                        boxmean=add_mean, showlegend=False)
            # For "boxes with whiskers" we set the same range of values on the x-axis as for the histograms,
            # show the grid on the x axis, but hide the labels on it
            fig.update_xaxes(matches=list(fig.select_traces(row=row + 1, col=col))[0].xaxis,
                             showgrid=True, showticklabels=False, row=row, col=col)
            # For \"boxes with whiskers\" we hide the y-axis title and labels on it
            fig.update_yaxes(title='', row=row, col=col, showticklabels=False)
        fig.update_xaxes(title=metrics['units'].iloc[index], title_font_size=units_font_size,
                         row=row + (1 if add_boxplot else 0), col=col)
        fig.update_yaxes(title=yaxis_title, title_font_size=units_font_size,
                         row=row + (1 if add_boxplot else 0), col=col)

    fig.update_xaxes(tickfont_size=axes_tickfont_size)
    fig.update_yaxes(tickfont_size=axes_tickfont_size)
    fig.update_annotations(font_size=labels_font_size)
    fig.update_layout(barmode='overlay',
                      title=title, title_font_size=title_font_size,
                      title_x=0.5, title_y=title_y, title_xanchor='center',
                      width=width, height=height)
    return fig


def plot_metric_confidence_interval(data, metrics, title=None, title_y=None, yaxis_title=None,
                                    title_font_size=14, labels_font_size=12, units_font_size=12, axes_tickfont_size=12,
                                    height=None, width=None, horizontal_spacing=None, vertical_spacing=None,
                                    n_cols=1, opacity=0.5):
    """
    The function is designed to construct confidence intervals for several metrics of several groups.
    A separate canvas is created for each metric. The canvas can be divided into several columns.
    The confidence interval is indicated as a horizontal segment with vertical cutoffs at the ends.
    The center of the confidence interval is highlighted by a dot.

        Parameters:
        ----------
        data : pandas.Series or pandas.DataFrame
            Confidential intervals of populations.
            pandas.Series - for constructing CI for one metric
            pandas.DataFrame - for constructing CI for several metrics. Metric data are arranged in columns.
            Population names should be used as the dataset index.

        metrics : DataFrame
            Information about metrics. Indexes are names of metrics.

        title : string or None. Default is None
            Chart Title

        title_y : float or None
            Relative position of the chart title by height (from 0.0 (bottom) to 1.0 (top))

        title_font_size : int. Default - 14
            Chart Title Font Size

        labels_font_size : int. Default - 12
            Font size of inscriptions

        units_font_size : int. Default - 12
            Font size of unit names

        axes_tickfont_size : int. Default - 12
            Font size of axes labels

        height : int or None. Default is None
            Height of the chart

        width : int or None. Default is None
            Diagram width

        horizontal_spacing : float or None. Defaults to None
            Distance between columns of canvases in fractions of width (from 0.0 to 1.0)

        vertical_spacing : float or None. Defaults to None
            Distance between the rows of canvases in fractions of the height (from 0.0 to 1.0)

        n_cols : int. Default - 1
            Number of columns of canvases

        opacity : float. Default is 0.5
            Opacity level (0.0 to 1.0) of the column color

        Returns:
        -----------------------
            No.
    """
    # Calculate the number of lines and their heights
    n_rows = int(np.ceil(metrics.shape[0] / n_cols))
    row_heights = [1 / n_rows] * n_rows
    titles = []
    specs = []
    # Generate a list of titles and chart specifications
    for index in range(0, metrics.shape[0], n_cols):
        titles += (metrics.iloc[index:index + n_cols, :]['name'].to_list())
        if vertical_spacing:
            specs.append([{'b': vertical_spacing}] * n_cols)
        else:
            specs.append([{}] * n_cols)
    # Create a canvas with n_row*n_cols graphs
    fig = make_subplots(cols=n_cols, rows=n_rows, row_heights=row_heights,
                        subplot_titles=titles, specs=specs,
                        horizontal_spacing=horizontal_spacing, vertical_spacing=0.004)
    # Display a scatter plot of metrics, placing boxes with whiskers above them
    for index, metric in enumerate(metrics.index):
        # We go by metrics
        col = index % n_cols + 1
        row = index // n_cols + 1
        if type(data) == pd.Series:
            # Add a dot plot to the canvas
            fig.add_scatter(x=[(data[metric][0] + data[metric][1]) / 2], y=[0],
                            error_x={'type': 'constant', 'value': abs(data[metric][0] - data[metric][1]) / 2},
                            row=row, col=col, name='',
                            marker_color=px.colors.DEFAULT_PLOTLY_COLORS[index],
                            marker_line_color='white', marker_line_width=1,
                            opacity=opacity, showlegend=False)
        else:
            # Add a dot for each group
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
        fig.update_xaxes(title=metrics.loc[metric, 'units'], title_font_size=units_font_size, row=row, col=col)
        fig.update_yaxes(title=yaxis_title, title_font_size=units_font_size, row=row, col=col)
    fig.update_xaxes(tickfont_size=axes_tickfont_size)
    fig.update_yaxes(visible=False, tickfont_size=axes_tickfont_size)
    fig.update_annotations(font_size=labels_font_size, y=1.1)
    fig.update_layout(
        title=title, title_y=title_y, title_font_size=title_font_size, title_x=0.5,
        width=width, height=height,
        legend_font_size=labels_font_size, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def plot_group_size_barchart(data, title=None, title_y=None, title_font_size=14, opacity=0.5, orientation='h',
                             labels_font_size=12, xaxis_title=None, yaxis_title=None,
                             axes_title_font_size=12, axes_tickfont_size=12,
                             height=None, width=None):
    """
    The function plots a bar chart that displays the sizes of the groups present in the data set.

        Parameters:
        ----------
        data : DataFrame
            Dataset. Group names should be used as the dataset index.

        title : string or None. Default is None
            Chart Title

        title_y : float or None
            Relative position of the chart title by height (from 0.0 (bottom) to 1.0 (top))

        title_font_size : int. Default - 14
            Chart Title Font Size

        opacity : float. Default is 0.5
            Opacity level (0.0 to 1.0) of the column color

        orientation : {'h', 'v'}. Default is 'h'
            Chart orientation: 'h'-horizontal, 'v'-vertical

        labels_font_size : int. Default - 12
            Font size of inscriptions

        xaxis_title : string or None. Defaults to None
            x-axis title

        yaxis_title : string or None. Defaults to None
            Y-axis title

        axes_title_font_size : int. Default - 12
            Axis Title Font Size

        axes_tickfont_size : int. Default - 12
            Font size of axes labels

        height : int or None. Default is None
            Height of the chart

        width : int or None. Default is None
            Diagram width

        Returns:
        -----------------------
            No.
    """

    # Build a bar chart
    # If the diagram is horizontal, then we change the order of the indices to the reverse
    df = data.index if orientation == 'v' else data.index[::-1]
    colors = px.colors.DEFAULT_PLOTLY_COLORS[:df.nunique()]
    if orientation == 'h':
        colors.reverse()
    fig = px.histogram(df, title=title, opacity=opacity, orientation=orientation, height=height, width=width)
    fig.update_traces(texttemplate="%{x}", hovertemplate='%{y} - %{x:} customers',
                      marker_color=colors, showlegend=False)
    fig.update_layout(bargap=0.2, boxgroupgap=0.2,
                      title_font_size=title_font_size,
                      title_x=0.5, title_y=title_y)
    fig.update_xaxes(title=xaxis_title, title_font_size=axes_title_font_size, tickfont_size=axes_tickfont_size)
    fig.update_yaxes(title=yaxis_title, title_font_size=axes_title_font_size, tickfont_size=axes_tickfont_size)
    fig.update_annotations(font_size=labels_font_size)
    return fig
