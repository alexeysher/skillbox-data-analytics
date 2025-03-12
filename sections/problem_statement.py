import streamlit as st

st.markdown(
    """
    # Problem statement
    
    ## The main task description
    
    [Megafon](https://en.wikipedia.org/wiki/MegaFon) is a large mobile phone and telecom operator.
    Like any business, this company wants to increase customer satisfaction with its service quality.
    This is an important task for retaining users, both long-standing and newly acquired. 
    After all, marketing and promotion costs will not be justified 
    if the customer leaves due to poor connection quality. 
    However, in the real world, resources are always limited, and the technical department 
    can solve a finite number of tasks per unit of time.
    
    For this reason, the company have managed a short customers survey. 
    asking them to rate their level of satisfaction with connection quality.
    Technical indicators were collected for each customer who completed the survey.
    
    The main task of this work is doing research for [Megafon](https://en.wikipedia.org/wiki/MegaFon) 
    and analyze how (and whether) 
    the customer satisfaction depends on the collected data.
    
    ## More details about the survey
    
    First of all, the customers were asked to score their satisfaction for communication service quality 
    on 10-point scale (10 - excellent, 1 - terrible). 
    
    If the customer score the quality of communication at 9 or 10 points, the survey ended. 
    If the customer score it below 9, a second question was asked about the reasons for dissatisfaction.
    For the second question the numbered answer options were provided:
    
    0. Unknown
    1. Missed calls, disconnected calls
    2. Waiting time for ringtones
    3. Poor connection quality in buildings, shopping centers, etc.
    4. Slow mobile Internet
    5. Slow video loading
    6. Difficult to answer
    7. Your own option
    
    The answer could be given in a free format or by listing the answer numbers separated by commas.
    """
)
