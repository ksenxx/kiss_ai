# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Configuration builder for KISS agent settings with CLI support."""

from argparse import ArgumentParser
from typing import Any, cast, get_args

from pydantic import BaseModel, Field, create_model
from pydantic_settings import BaseSettings, SettingsConfigDict

from kiss.core import config as config_module
from kiss.core.config import Config


def _add_model_arguments(parser: ArgumentParser, model: type[BaseModel], prefix: str = "") -> None:
    """Recursively add arguments for all fields in a Pydantic model."""
    for field_name, field_info in model.model_fields.items():
        arg_name = f"{prefix}.{field_name}" if prefix else field_name
        dest_name = arg_name.replace(".", "__")
        field_type = field_info.annotation

        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            _add_model_arguments(parser, field_type, arg_name)
            continue

        if hasattr(field_type, "__origin__"):
            args = get_args(field_type)
            non_none = [a for a in args if a is not type(None)]
            if non_none:  # pragma: no branch – Optional always has a non-None arg
                field_type = non_none[0]

        arg_name_dashes = arg_name.replace("_", "-")
        if arg_name_dashes != arg_name:
            names = [f"--{arg_name_dashes}", f"--{arg_name}"]
        else:
            names = [f"--{arg_name}"]
        help_text = f"{field_info.description or field_name} (default: {field_info.default})"

        if field_type is bool:
            parser.add_argument(
                *names, action="store_true", dest=dest_name, default=None, help=help_text
            )
            if arg_name_dashes != arg_name:
                no_names = [f"--no-{arg_name_dashes}", f"--no-{arg_name}"]
            else:
                no_names = [f"--no-{arg_name}"]
            parser.add_argument(
                *no_names, action="store_false", dest=dest_name, help=f"Disable {field_name}"
            )
        else:
            if field_type is int:
                arg_type: type[int] | type[float] | type[str] = int
            elif field_type is float:
                arg_type = float
            else:
                arg_type = str
            parser.add_argument(*names, type=arg_type, dest=dest_name, default=None, help=help_text)


def _flat_to_nested_dict(
    flat: dict[str, Any], model: type[BaseModel], prefix: str = ""
) -> dict[str, Any]:
    """Convert flat argparse namespace dict to nested dict matching model structure.

    Args:
        flat: A flat dictionary from argparse namespace with double-underscore separated keys.
        model: The Pydantic model class defining the target structure.
        prefix: Optional prefix for nested field key construction.

    Returns:
        dict[str, Any]: A nested dictionary matching the Pydantic model structure.
    """
    nested: dict[str, Any] = {}

    for field_name, field_info in model.model_fields.items():
        if prefix:
            arg_key = f"{prefix}__{field_name}"
        else:
            arg_key = field_name
        field_type = field_info.annotation

        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            nested_dict = _flat_to_nested_dict(flat, field_type, arg_key)
            if nested_dict:
                nested[field_name] = nested_dict
        elif arg_key in flat and flat[arg_key] is not None:
            nested[field_name] = flat[arg_key]

    return nested


def build_config() -> None:
    """Parse CLI arguments for Config fields and update DEFAULT_CONFIG.

    Makes all fields in the base :class:`Config` class accessible via
    command-line flags.  For example ``--max-budget 300.0`` overrides the
    default ``max_budget``.  Only explicitly provided arguments take effect;
    omitted flags keep their defaults.

    If ``DEFAULT_CONFIG`` has already been extended by prior
    :func:`add_config` calls, those extra fields are preserved and also
    made available on the command line.
    """
    current_type = type(config_module.DEFAULT_CONFIG)

    parser = ArgumentParser(description="KISS Configuration")
    _add_model_arguments(parser, current_type)
    parsed_args, _ = parser.parse_known_args()

    overrides = _flat_to_nested_dict(vars(parsed_args), current_type)

    if overrides:
        defaults = config_module.DEFAULT_CONFIG.model_dump()

        def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
            result = base.copy()
            for k, v in override.items():
                if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                    result[k] = merge(cast(dict[str, Any], result[k]), cast(dict[str, Any], v))
                else:
                    result[k] = v
            return result

        config_module.DEFAULT_CONFIG = current_type.model_validate(
            merge(defaults, overrides)
        )


def add_config(name: str, config_class: type[BaseModel]) -> None:
    """Build the KISS config, optionally overriding with command-line arguments.

    This function accumulates configs - each call adds a new config field while
    preserving existing fields from previous calls.

    Args:
        name: Name of the config class.
        config_class: Class of the config.
    """
    existing_fields: dict[str, Any] = {}
    current_config = config_module.DEFAULT_CONFIG
    if current_config is not None:  # pragma: no branch – DEFAULT_CONFIG always set at import
        base_fields = set(Config.model_fields.keys())
        for field_name in type(current_config).model_fields.keys():
            if field_name not in base_fields:
                field_info = type(current_config).model_fields[field_name]
                field_type = field_info.annotation
                current_value = getattr(current_config, field_name, None)
                if current_value is not None:
                    existing_fields[field_name] = (
                        field_type,
                        Field(
                            default_factory=lambda v=current_value: (  # type: ignore[misc]
                                v.model_copy() if hasattr(v, "model_copy") else v  # type: ignore[union-attr]
                            )
                        ),
                    )
                else:  # pragma: no cover – all config fields have non-None defaults
                    existing_fields[field_name] = (
                        field_type,
                        Field(default_factory=lambda ft=field_type: ft()),  # type: ignore[misc]
                    )

    all_fields: dict[str, Any] = {
        **existing_fields,
        name: (config_class, Field(default_factory=config_class)),
    }

    config_with_name = create_model(  # type: ignore[call-overload]
        "ConfigWithName",
        __base__=Config,
        **all_fields,
    )

    dynamic_config = type(
        "DynamicConfig",
        (BaseSettings, config_with_name),
        {"model_config": SettingsConfigDict(extra="ignore")},
    )

    parser = ArgumentParser(description="KISS Config Builder")
    _add_model_arguments(parser, dynamic_config)
    parsed_args, _ = parser.parse_known_args()

    overrides = _flat_to_nested_dict(vars(parsed_args), dynamic_config)

    if overrides:
        defaults = dynamic_config().model_dump()

        def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
            result = base.copy()
            for k, v in override.items():
                if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                    result[k] = merge(cast(dict[str, Any], result[k]), cast(dict[str, Any], v))
                else:
                    result[k] = v
            return result

        config_instance = dynamic_config.model_validate(merge(defaults, overrides))  # type: ignore[attr-defined]
    else:
        config_instance = dynamic_config()

    config_module.DEFAULT_CONFIG = config_instance
