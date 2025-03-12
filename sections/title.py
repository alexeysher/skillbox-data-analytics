import streamlit as st
from auxiliary import set_text_style, MegafonColors

st.markdown(
    f'''
    <p style="color: {MegafonColors.brandPurple}; font-size: 5rem;">
        <br>Research of MegaFon customer success survey<br>
    </p>
    <p style="color: {MegafonColors.brandGreen}; font-size: 4rem;">
        Author: <b>Alexey Sherstobitov</b>
    </p>
    ''', unsafe_allow_html=True
)
