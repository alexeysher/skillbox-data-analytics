import streamlit as st
import pandas as pd
from functions import display_cat_info
from auxiliary import MegafonColors


st.markdown(
    '''
    # Setting of objectives
    
    The metrics present in the dataset are related exclusively to the quality of mobile Internet service. 
    In addition, there are customer scores regarding the quality of communication 
    and the reasons that determined the scores. 
    Based on this, we will try to recognize how to classify users in terms of 
    their assessment of the quality of the mobile Internet service. 
    Having this information will make it possible to build a classifier 
    for use in predicting customer churn and ways to retain them.

    Initially it's only possible to divide users into classes (categories) based on the answers to questions. 
    And only then it will be able to see which categories of users have similar metric values, 
    i.e. which categories of customers belong to the same population and can be combined into one class.
    
    ## Selecting customers for the research
    
    In the context of the research, only those customers are interested whose assessment of the mobile Internet service 
    can be determined.
    The most obvious thing is that customers who gave `9` and `10` scores in response to the 1st question 
    also highly assess the mobile Internet service.
    Also it's appropriate to analyze that customers who, in answer to the 2nd question, 
    specified slow mobile Internet (`4`) and/or slow video loading speed (`5`).
    The exact assessment of the remaining customers who gave from `1` to `8` score in the answer to the 1st question, 
    but did not specify either slow mobile Internet (`4`) or slow video loading speed (`5`) 
    in their answer to the 2nd question is unknown. 
    It's possible only to assume that their score of the mobile Internet service is higher than 
    the score they gave for the connection quality.
    '''
)
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
     'props': f'font-size: 1rem; color: white; background-color: {MegafonColors.brandPurple80}; font-weight: normal;'},
    {'selector': 'th.col_heading', 'props': 'text-align: center; width: 2.5rem;'},
    {'selector': 'td', 'props': 'text-align: center; font-size: 1rem; font-weight: normal;'},
    {'selector': 'th.blank', 'props': 'border-style: hidden'},
], overwrite=False)
s = s.map(lambda v:
          f'color: white; background-color: {MegafonColors.brandGreen};' if v == '+'
          else f'color: {MegafonColors.spbSky2}; background-color: {MegafonColors.spbSky1};')
st.markdown(s.to_html(table_uuid="table_client_choosing"), unsafe_allow_html=True)

st.markdown(
    '''
    ## Dividing customers into categories depending on the answer to the 1st question

    When answering the 1st question, the customers were asked to score the quality of the service on a 10-point scale 
    (where 10 is "Excellent" and 1 is "Terrible"). 
    But as previously established, 
    when using such a scale, it was quite difficult for the user to rate, 
    and therefore it is better to convert the scores to a 5-point scale. 
    Therefore, divide the selected customers into `5` categories of mobile Internet service ratings 
    depending on the answer to the 1st question:
    
    | Category | Scores |
    |:-----------------|:-------|
    | Very unsatisfied |1, 2 |
    | Unsatisfied |3, 4 |
    | Neutral |5, 6 |
    | Satisfied |7, 8 |
    | Very satisfied |9, 10 |
    
    According to above, will append to the dataset a column `Internet score` 
    indicating to which category the customer belongs.
    '''
)
# st.subheader('Categories of CSAT according to Q1 answer')
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
     'props': f'font-size: 1rem; color: white; background-color: {MegafonColors.brandPurple80}; font-weight: normal;'},
    {'selector': 'th.col_heading', 'props': 'text-align: center; width: 2.5rem;'},
    {'selector': 'td', 'props': 'text-align: center; font-size: 1rem; font-weight: normal;'},
    {'selector': 'th.blank', 'props': 'border-style: hidden'}
], overwrite=False)
s = s.map(lambda v:
          f'color: white; background-color: {MegafonColors.brandGreen};' if v == '+'
          else f'color: {MegafonColors.spbSky2}; background-color: {MegafonColors.spbSky1};')
st.markdown(s.to_html(table_uuid="table_categories_1"), unsafe_allow_html=True)
st.markdown(
    '''
    ## Dividing customers into categories based on answers to the 2nd question
    
    In answers to the 2nd question, the selected customers could specify either slow internet (`4`) 
    or slow video loading (`5`) separately, or both reasons together.<br>
    
    It's possible to assume with the greatest certainty that they are fully satisfied with 
    the mobile internet service and, accordingly, that there are no reasons for dissatisfaction with 
    the internet only for those customers who scored the quality of the mobile service as `Very satisfied`.
    
    Based on all of the above, it's appropriate split the selected customers into the following categories 
    based on the reasons for dissatisfaction with the mobile internet service:
    
    |Category |Description |Answers |
    |:----------------|:----------------------------------------------------------|:-------------------------|
    | Internet and Video|Unsatisfied with Mobile Internet and Video loading in the same way|Contain 4 and 5 |
    | Internet |More unsatisfied with Mobile Internet than Video loading|Contain 4, do not contain 5 |
    | Video |More unsatisfied with Video loading than Mobile Internet|Contain 5, do not contain 4 |
    | No |Satisfied with mobile Internet and Video loading |- |
    
    According to above, will append to the dataset a column `Dissatisfaction reasons` 
    indicating to which category the customer belongs.

    '''
)
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
     'props': f'font-size: 1rem; color: white; background-color: {MegafonColors.brandPurple80}; font-weight: normal;'},
    {'selector': 'th.col_heading', 'props': 'text-align: center; width: 8rem;'},
    {'selector': 'td', 'props': 'text-align: center; font-size: 1rem; font-weight: normal;'},
    {'selector': 'th.blank', 'props': 'border-style: hidden'}
], overwrite=False)
s = s.map(lambda v:
          f'color: white; background-color: {MegafonColors.brandGreen};' if v == '+'
          else f'color: {MegafonColors.spbSky2}; background-color: {MegafonColors.spbSky1};')
st.markdown(s.to_html(table_uuid="table_categories_2"), unsafe_allow_html=True)
st.markdown(
    '''
    ## Customers distribution map
    
    All the customers under consideration belong to one of the categories of assessing the quality 
    of the Internet service and one of the categories of reasons for dissatisfaction with it. 
    Based on this information, a map of the distribution of customers can be created:
    '''
)
s = display_cat_info(st.session_state.data_clean)
st.markdown(s.to_html(table_uuid="table_categories_dist"), unsafe_allow_html=True)
# '''
#
#     - Classify customers according to their assessment of the quality of mobile Internet service.
#     - Determine witch metrics have the strongest influence to the customers assessment.
# '''