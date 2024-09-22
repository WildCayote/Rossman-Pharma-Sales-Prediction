import pandas as pd
import numpy as np
from scipy.stats import f_oneway, ttest_ind

def ab_test(dependent_col: str, independent_col: str, data: pd.DataFrame):
    """
    Conducts an A/B test (two-sample t-test) to determine if there is a statistically significant
    difference between the means of the dependent variable for two groups defined by the independent variable.
    
    Parameters:
    ----------
    dependent_col : str
        The name of the column containing the dependent variable (numeric) to test.
    independent_col : str
        The name of the column containing the independent variable (categorical) that should have exactly two groups.
    data : pd.DataFrame
        The DataFrame containing the dataset with both the dependent and independent variables.
    
    Returns:
    -------
    t_stat : float
        The calculated t-statistic from the two-sample t-test.
    p_value : float
        The p-value from the t-test, indicating the significance of the difference between the two groups.
    
    Raises:
    -------
    ValueError:
        If the independent variable has more than two unique values, the function will raise an error
        since A/B testing requires exactly two groups.
    
    Example:
    --------
    t_stat, p_value = ab_test('ConversionRate', 'Group', data)
    
    Notes:
    ------
    - A/B tests compare the means of two independent groups (A and B) for a given numeric variable.
    - If there are missing values in the dependent variable, they are automatically omitted from the t-test.
    """
    # check if there are only 2 unique values in the independet col
    unique_count = data[independent_col].nunique()
    if unique_count != 2:
        print(f"Inorder to conduct an A/B test you need exactly two groups. You however have {unique_count} groups.")
    
    else:
        # obtain the two group types
        groups = data[independent_col].unique().tolist()

        # obtain group data
        group_a = data[data[independent_col] == groups[0]][dependent_col]
        group_b = data[data[independent_col] == groups[1]][dependent_col]

        # run ab test
        t_stat, p_value = ttest_ind(a=group_a, b=group_b, nan_policy='omit')

        return t_stat, p_value

def annova_test(dependent_col: str, independent_col: str, data: pd.DataFrame):
    """
    Performs a one-way ANOVA test to determine if there are statistically significant differences 
    between the means of the dependent variable across the groups formed by the independent variable.
    
    Parameters:
    ----------
    dependent_col : str
        The name of the column containing the dependent variable (numeric) to test.
    independent_col : str
        The name of the column containing the independent variable (categorical) used to group the data.
    data : pd.DataFrame
        The DataFrame containing the dataset with both the dependent and independent variables.
    
    Returns:
    -------
    f_statistics : float
        The F-statistic value from the ANOVA test.
    p_value : float
        The p-value from the ANOVA test, which indicates the statistical significance.
    """

    # get the grouping along the independet_col
    data_groups = data.groupby(by=independent_col)

    # get the group names formed by using the independet_col
    group_names = list(data_groups.groups.keys())
    
    # get the groups and then put the dependent col values into a global list
    data_points = []
    for group_name in group_names:
        # group dataframe
        group_data = data_groups.get_group(name=group_name)

        # add the data points of the dependent column
        values = group_data[dependent_col]
        data_points.append(values)
    
    # perform an ANOVA test
    f_statistics, p_value = f_oneway(*data_points)

    return f_statistics, p_value

def test_hypothesis(null_hypothesis: str, p_value: np.float64):
    """
    Tests the given null hypothesis based on the p-value from the statistical test.
    
    Parameters:
    ----------
    null_hypothesis : str
        A description of the null hypothesis being tested.
    p_value : float
        The p-value from the statistical test used to determine whether to reject the null hypothesis.
    
    Returns:
    -------
    None
    
    Prints:
    -------
    Whether the null hypothesis is accepted or rejected based on the significance level (alpha = 0.05).
    """
    # if the p_value is less than 0.05 reject the null hypothesis
    if p_value < 0.05:
        print(f"Rejected the null hypothesis: {null_hypothesis} \np_value: {p_value}")
    else:
        print(f"Accepted the null hypothesis: {null_hypothesis} \np_value: {p_value}")