# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 3: ``UsefulTools.Edit`` with an empty ``old_string``.

``str.count("")`` returns ``len(content) + 1``, so an empty
``old_string``:

* with ``replace_all=True`` interleaves ``new_string`` between EVERY
  character of the file — silent file corruption; and
* on an empty file (``count("") == 1``) silently overwrites the file
  with ``new_string`` (acting as Write instead of failing).

The Edit tool must reject an empty ``old_string`` with an explicit
error and leave the file untouched.
"""

from pathlib import Path

from kiss.agents.sorcar.useful_tools import UsefulTools


def test_edit_empty_old_string_replace_all_does_not_corrupt(tmp_path: Path) -> None:
    """replace_all with old_string='' must error out, not interleave text."""
    f = tmp_path / "data.txt"
    f.write_text("abc")
    tools = UsefulTools()

    out = tools.Edit(str(f), "", "X", replace_all=True)

    assert out.startswith("Error:"), out
    assert f.read_text() == "abc"


def test_edit_empty_old_string_on_empty_file_errors(tmp_path: Path) -> None:
    """old_string='' on an empty file must not silently act as Write."""
    f = tmp_path / "empty.txt"
    f.write_text("")
    tools = UsefulTools()

    out = tools.Edit(str(f), "", "new content")

    assert out.startswith("Error:"), out
    assert f.read_text() == ""


def test_edit_empty_old_string_single_replace_errors(tmp_path: Path) -> None:
    """old_string='' without replace_all must also be an explicit error."""
    f = tmp_path / "data.txt"
    f.write_text("hello")
    tools = UsefulTools()

    out = tools.Edit(str(f), "", "X")

    assert out.startswith("Error:"), out
    assert f.read_text() == "hello"
