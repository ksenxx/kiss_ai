# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""String-template helpers used by the GEPA prompt-evolution module."""

from string import Formatter as StringFormatter


def get_template_field_names(text: str) -> list[str]:
    """Get the field names from the text.

    Args:
        text (str): The text containing template field placeholders.

    Returns:
        list[str]: A list of field names found in the text.
    """
    return [
        field_name
        for _, field_name, _, _ in StringFormatter().parse(text)
        if field_name is not None
    ]


def escape_invalid_template_field_names(text : str, valid_field_names : set[str]) -> str:
    """Escape invalid field names from the text.

    Args:
        text (str): The text containing template field placeholders.
        valid_field_names (set[str]): A list of valid field names.

    Returns:
        An escaped string with invalid field placeholders escaped
    """

    template_result = []
    for literal_text, field_name, format_spec, conversion in StringFormatter().parse(text):
        literal_text = literal_text.replace("{", "{{").replace("}", "}}")
        template_result.append(literal_text)

        if field_name is None:
            continue

        # A placeholder is only fillable when its own field AND every
        # field nested inside its format spec are valid.  Doubled
        # braces are NOT unescaped inside a format spec by
        # ``str.format``, so escaping just the invalid nested field
        # would leave an invalid spec that raises at format time —
        # the whole placeholder must be escaped verbatim instead.
        spec_field_names = get_template_field_names(format_spec) if format_spec else []
        is_valid = field_name in valid_field_names and all(
            name in valid_field_names for name in spec_field_names
        )

        placeholder = "{" + field_name
        if conversion:
            placeholder += f"!{conversion}"
        if format_spec:
            placeholder += f":{format_spec}"
        placeholder += "}"

        if not is_valid:
            placeholder = placeholder.replace("{", "{{").replace("}", "}}")
        template_result.append(placeholder)

    return "".join(template_result)
