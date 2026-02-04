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
    """Recursively add arguments for all fields in a Pydantic model.

    Args:
        parser: The ArgumentParser instance to add arguments to.
        model: The Pydantic model class to extract fields from.
        prefix: Optional prefix for nested model field names.
    """
    for field_name, field_info in model.model_fields.items():
        arg_name = f"{prefix}.{field_name}" if prefix else field_name
        # Use dots in dest to preserve structure (argparse converts dots to underscores by default)
        dest_name = arg_name.replace(".", "__")  # Use double underscore as separator for dest
        field_type = field_info.annotation

        # Handle nested BaseModel fields
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            _add_model_arguments(parser, field_type, arg_name)
            continue

        # Get type for argument (handle Optional/Union)
        if hasattr(field_type, "__origin__"):
            args = get_args(field_type)
            non_none = [a for a in args if a is not type(None)]
            if non_none:
                field_type = non_none[0]

        # Determine type and action
        if field_type is bool:
            help_text = f"{field_info.description or field_name} (default: {field_info.default})"
            parser.add_argument(
                f"--{arg_name}",
                action="store_true",
                dest=dest_name,
                default=None,
                help=help_text,
            )
            parser.add_argument(
                f"--no-{arg_name}",
                action="store_false",
                dest=dest_name,
                help=f"Disable {field_name}",
            )
        else:
            # Determine the argument type for argparse
            if field_type is int:
                arg_type: type[int] | type[float] | type[str] = int
            elif field_type is float:
                arg_type = float
            else:
                arg_type = str
            help_text = f"{field_info.description or field_name} (default: {field_info.default})"
            parser.add_argument(
                f"--{arg_name}",
                type=arg_type,
                dest=dest_name,
                default=None,
                help=help_text,
            )


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
        # Build the key as stored in argparse namespace (dots -> double underscores)
        if prefix:
            arg_key = f"{prefix}__{field_name}"
        else:
            arg_key = field_name
        field_type = field_info.annotation

        # Handle nested BaseModel
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            nested_dict = _flat_to_nested_dict(flat, field_type, arg_key)
            if nested_dict:
                nested[field_name] = nested_dict
        elif arg_key in flat and flat[arg_key] is not None:
            nested[field_name] = flat[arg_key]

    return nested


def add_config(name: str, config_class: type[BaseModel]) -> None:
    """Build the KISS config, optionally overriding with command-line arguments.

    This function accumulates configs - each call adds a new config field while
    preserving existing fields from previous calls.

    Args:
        name: Name of the config class.
        config_class: Class of the config.
    """
    # Get existing config fields from current DEFAULT_CONFIG (if any custom fields exist)
    existing_fields: dict[str, Any] = {}
    current_config = config_module.DEFAULT_CONFIG
    if current_config is not None:
        # Get all fields that are not part of the base Config class
        base_fields = set(Config.model_fields.keys())
        for field_name in current_config.model_fields.keys():
            if field_name not in base_fields:
                field_info = current_config.model_fields[field_name]
                field_type = field_info.annotation
                # Get the current value as default
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
                else:
                    # For None values, create a lambda that calls the type constructor
                    existing_fields[field_name] = (
                        field_type,
                        Field(default_factory=lambda ft=field_type: ft()),  # type: ignore[misc]
                    )

    # Add the new config field
    all_fields: dict[str, Any] = {
        **existing_fields,
        name: (config_class, Field(default_factory=config_class)),
    }

    # Create BaseModel subclass dynamically using create_model
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

    # Parse command-line arguments (use parse_known_args to ignore unknown args like pytest's)
    parser = ArgumentParser(description="KISS Config Builder")
    _add_model_arguments(parser, dynamic_config)
    parsed_args, _ = parser.parse_known_args()

    # Convert flat argparse namespace to nested dict and validate with BaseSettings
    overrides = _flat_to_nested_dict(vars(parsed_args), dynamic_config)

    # BaseSettings.model_validate handles defaults - merge overrides with defaults
    if overrides:
        defaults = dynamic_config().model_dump()

        # Recursive merge: override values take precedence, nested dicts are merged
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
