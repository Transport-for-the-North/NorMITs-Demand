# -*- coding: utf-8 -*-
"""
Created on: 20/06/2022
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins

# Third Party

# Local Imports


def centre_pad_string(s: str, padding: int, padding_char: str = None) -> str:
    """Pads a string on either side by a set number of characters

    Parameters
    ----------
    s:
        The string to pad

    padding:
        The amount of `padding_char` to apply to each side of `s`

    padding_char:
        The character to use to pad `s` with

    Returns
    -------
    padded_s:
        The input string `s` with padding either side.
    """
    if padding_char is None:
        return f"{s:^{len(s) + padding * 2}}"

    return f"{s:{padding_char}^{len(s) + padding * 2}}"


def title_padding(s: str, padding: int = 10) -> str:
    """Pads a string to make the default title style for codebase

    This involves a few processes:
    - Make all characters upper case letters
    - Pads the string with one space on either side
    - Pads the string with `padding` number of '~' characters

    Parameters
    ----------
    s:
        The string to pad

    padding:
        The amount of padding to apply to each side of `s`

    Returns
    -------
    padded_s:
        The input string `s` with padding either side.
    """
    temp_s = centre_pad_string(s, 1)
    return centre_pad_string(temp_s, padding, "~")
