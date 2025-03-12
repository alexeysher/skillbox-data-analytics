import streamlit as st
from auxiliary import set_text_style, MegafonColors

plain_text = "Research of MegaFon customer success survey"
font_formatted_text = f'**{set_text_style(plain_text, font_size=80, color=MegafonColors.brandPurple)}**'
st.markdown(font_formatted_text, unsafe_allow_html=True)
font_formatted_text = set_text_style('&nbsp;', font_size=24)
st.markdown(font_formatted_text, unsafe_allow_html=True)
author = set_text_style('Author: ', font_size=48, color=MegafonColors.brandGreen, tag='span') + \
         set_text_style('**Alexey Sherstobitov**', font_size=48, color=MegafonColors.brandGreen, tag='span')
st.markdown(author, unsafe_allow_html=True)
