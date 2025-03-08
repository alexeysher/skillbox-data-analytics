import streamlit as st
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
