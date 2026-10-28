# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The no-remote-password refusal names the installed brand, not the stock one."""

from __future__ import annotations

from kiss.core.brand import PRODUCT_NAME
from kiss.server import sorcar


def test_no_password_refusal_names_the_product() -> None:
    text = sorcar._NO_PASSWORD_REFUSAL["text"]
    assert f"on the computer running {PRODUCT_NAME} (or run install.sh there)" in text
    assert ("KISS Sorcar" in text) == (PRODUCT_NAME == "KISS Sorcar")
