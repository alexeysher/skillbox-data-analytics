import streamlit as st
import streamlit.components.v1 as components
from google.cloud import storage
from google.oauth2 import service_account
from pickle import load
from pathlib import Path
import textwrap


class MegafonColors:
    """
    Megafon brand colors
    """
    base = '#FFF'
    content = '#333'
    brandGreen = '#00b956'
    brandPurple = '#731982'
    brandGreenDarken10 = '#00863e'
    brandGreen80 = '#0cdc78'
    brandGreen20 = '#ddffec'
    brandPurple80 = '#aa67c1'
    brandPurple20 = '#fef'
    scantBlue2 = '#a3a5aa'
    spbSky0 = '#f6f6f6'
    spbSky1 = 'ededed'
    spbSky2 = '#d8d8d8'
    spbSky3 = '#999'
    orangeDark = '#e39338'


def connect_gcs(
        credential_info,
        bucket_id: str,
) -> storage.Bucket:
    """
    Establish connection to Bucket with given ID on Google Cloud Storage using given Credentials
    """
    credentials = service_account.Credentials.from_service_account_info(credential_info)
    storage_client = storage.Client(credential_info['project_id'], credentials=credentials)
    return storage_client.bucket(bucket_id)


def load_data_from_gcs(file_name: str, bucket: storage.Bucket, data_path: str):
    """
    Loads data from pickle-file in the given Bucket on Google Cloud Storage
    """
    file_path = f'{data_path}/{file_name}'
    blob = bucket.blob(file_path)
    blob.download_to_filename(file_name)
    with open(file_name, 'rb') as fp:
        data = load(fp)
    Path(file_name).unlink()
    return data


def css_styling():
    """
    Styles UI.
    """
    st.html(f"""
    <style>
        MainMenu {{visibility: hidden;}}
        # header {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        .st-emotion-cache-15fru4 {{
            font-size: 0.75em;
        }}
        .st-emotion-cache-1104ytp {{
            font-size: 1.0rem;
        }}
        .st-emotion-cache-t1wise {{
            padding-top: 4rem;
        }}
        .st-emotion-cache-1104ytp h1 {{
            color: {MegafonColors.brandGreen};
            # font-size: 2.5rem;
            # font-weight: strong;
            # padding: 1.25rem 0px 1rem;
        }}
        .st-emotion-cache-1104ytp h2 {{
            color: {MegafonColors.brandGreen};
            # font-size: 2.0rem;
            # padding: 1rem 0px;
        }}
        .st-emotion-cache-1104ytp h3 {{
            color: {MegafonColors.brandPurple};
            # font-size: 1.5rem;
            # padding: 0.5rem 0px 1rem;
        }}
        # .st-emotion-cache-1104ytp h4 {{
        #     font-size: 1.5rem;
        #     padding: 0.5rem 0px 1rem;
        # }}
        # .st-emotion-cache-1104ytp p {{
        #     word-break: break-word;
        #     margin-top: 0px;
        #     margin-left: 0px;
        #     margin-right: 0px;
        # }}
        # dataframe {{
        #     font-size: 16px;
        # }}
        # Code
    </style>
    """)


def wrap_text(text, length=50):
    """
    Splits the text into lines of a given length and replaces line breaks with the HTML element <br>

        Parameters:
        ----------
        text : string
            Text to process.

        length : int
            Maximum string length

        Return:
        -----------------------
            Object of type string.
    """
    return textwrap.fill(text, length).replace('\n', '<br>')


def set_text_style(text: str, tag: str = 'p', font_family: str = None, font_size: int = None,
                   color: str = None, background_color: str = None, text_align: str = None):
    """
    Returns HTML-tag for given text with specified format.
    """
    variables = []
    if font_family is not None:
        variables.append(f'font-family: {font_family}')
    if font_size is not None:
        variables.append(f'font-size: {font_size}px')
    if color is not None:
        variables.append(f'color: {color}')
    if background_color is not None:
        variables.append(f'background-color: {background_color}')
    if text_align is not None:
        variables.append(f'text-align: {text_align}')
    variables.append('')
    style = '; '.join(variables)
    return f'<{tag} style="{style}">{text}</{tag}>'


def set_widget_style(widget_text, font_family: str = None, font_size: int = None,
                     color: str = None, background_color: str = None, text_align: str = None):
    """
    Sets the style of the widget containing the specified text.
    """
    html = \
        """
    <script>
        var elements = window.parent.document.querySelectorAll('*'), i;
        for (i = 0; i < elements.length; ++i) { 
            if (elements[i].innerText == '""" + widget_text + "') {"
    if font_family is not None:
        html += \
            f"""
                elements[i].style.fontFamily='{font_family}';
        """
    if font_size is not None:
        html += \
            f"""
                elements[i].style.fontSize='{font_size}px';
        """
    if color is not None:
        html += \
            f"""
                elements[i].style.color='{color}';
        """
    if background_color is not None:
        html += \
            f"""
                elements[i].style.backgroundColor='{background_color}';
        """
    if text_align is not None:
        html += \
            f"""
                elements[i].style.textAlign='{text_align}';
        """
    html += \
        """
            } 
        } 
    </script> 
    """
    components.html(html, height=0, width=0)
