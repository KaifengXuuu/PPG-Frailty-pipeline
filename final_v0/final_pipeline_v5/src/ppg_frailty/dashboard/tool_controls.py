"""Render public CLI/YAML parameters without duplicating their validation rules.

Argparse remains the command authority. These helpers only describe widgets and
serialize their values; the selected command parses and validates the result.
Slider extents are suggestions and never clip typed values.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Mapping

from .workflow_controls import _bounds


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return copy.deepcopy(value)


def _spec(path: str, value: Any, *, group: str, kind: str | None = None,
          choices: list | None = None) -> dict[str, Any]:
    if kind is None:
        kind = ('boolean' if isinstance(value, bool) else 'integer' if isinstance(value, int)
                else 'number' if isinstance(value, float) else 'list' if isinstance(value, (list, tuple))
                else 'text')
    bounds = _bounds(value, path, kind == 'integer') if kind in {'integer', 'number'} else (None, None, None)
    return {'path': path, 'value': _plain(value), 'kind': kind, 'choices': choices or [],
            'min': bounds[0], 'max': bounds[1], 'step': bounds[2], 'group': group,
            'stage': 'tools', 'nullable': value is None}


def command_parameter_specs(parser: argparse.ArgumentParser, command: str) -> list[dict[str, Any]]:
    """Describe every visible argument of a real public subcommand.

    ``path`` is argparse's destination and ``arg`` its preferred long spelling.
    Required/mutually-exclusive metadata is informational: argparse performs the
    actual checks. Nullable boolean flags retain their third, inherited state.
    """
    selected = parser
    actions = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            selected = action.choices[command]
            actions.extend(selected._actions)
        else:
            actions.append(action)
    result = []
    for action in actions:
        if action.dest == 'help' or isinstance(action, argparse._SubParsersAction) or action.help == argparse.SUPPRESS:
            continue
        value = None if action.default == argparse.SUPPRESS else _plain(action.default)
        choices = list(action.choices) if action.choices is not None else []
        array = isinstance(action, (argparse._AppendAction, argparse._ExtendAction)) or action.nargs in {'+', '*'} or isinstance(action.nargs, int) and action.nargs > 0
        if isinstance(action, argparse.BooleanOptionalAction):
            action_name = 'boolean_optional'
            kind = 'select' if value is None else 'boolean'
            choices = [None, True, False] if value is None else []
        elif isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            action_name, kind = ('store_true' if action.const else 'store_false'), 'boolean'
        elif isinstance(action, argparse._CountAction):
            action_name, kind = 'count', 'integer'
        elif isinstance(action, argparse._StoreConstAction):
            action_name, kind = 'store_const', 'boolean'
            value = action.default == action.const
        else:
            action_name = 'extend' if isinstance(action, argparse._ExtendAction) else 'append' if isinstance(action, argparse._AppendAction) else 'store'
            kind = ('multi' if array and choices else 'list' if array else 'select' if choices
                    else 'integer' if action.type is int else 'number' if action.type is float
                    else None)
        options = list(action.option_strings)
        flag = next((option for option in options if option.startswith('--') and not option.startswith('--no-')), None)
        flag = flag or next((option for option in options if option.startswith('--')), None) or next(iter(options), None)
        spec = _spec(action.dest, value, group=command, kind=kind, choices=_plain(choices))
        spec.update(arg=flag, options=options, action=action_name, nargs=action.nargs,
                    required=action.required, default=_plain(value), array=array,
                    type=getattr(action.type, '__name__', 'str'), help=action.help or '',
                    const=_plain(action.const))
        for index, group in enumerate(selected._mutually_exclusive_groups):
            if action in group._group_actions:
                spec.update(exclusive_group=index, exclusive_required=group.required)
        result.append(spec)
    return result


def _argument_text(value: Any, spec: Mapping[str, Any]) -> str:
    if spec['type'] == 'int' and isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def arguments_from_values(specs: list[dict[str, Any]], mapping: Mapping[str, Any]) -> tuple[str, ...]:
    """Serialize controls to argv (without the command name or shell quoting).

    Unchanged optional defaults are omitted. Invalid combinations are deliberately
    left to the original parser, rather than being repaired or silently reset.
    """
    result: list[str] = []
    for spec in specs:
        value = mapping.get(spec['path'], spec['value'])
        if value is None or (value == spec['default'] and not spec['required'] and spec['arg']):
            continue
        flag, action = spec['arg'], spec['action']
        if action == 'boolean_optional':
            option = flag if value else next(option for option in spec['options'] if option.startswith('--no-'))
            result.append(option)
        elif action in {'store_true', 'store_false', 'store_const'}:
            active = not value if action == 'store_false' else bool(value)
            if active:
                result.append(flag)
        elif action == 'count':
            result.extend([flag] * int(value))
        elif action == 'append':
            for item in value:
                result.extend([flag] if flag else [])
                items = item if isinstance(item, (list, tuple)) else [item]
                result.extend(_argument_text(element, spec) for element in items)
        else:
            items = value if isinstance(value, (list, tuple)) else [value]
            if not items and spec['nargs'] != '*':
                continue
            result.extend([flag] if flag else [])
            result.extend(_argument_text(item, spec) for item in items)
    return tuple(result)


def plan_parameter_specs(mapping: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Describe YAML leaves in document order, including nested case mappings.

    A YAML scalar has no enum schema: unknown strings remain editable text rather
    than invented choices. Scalar lists stay one editable list; lists of objects
    expose their member parameters. The YAML editor can add/remove list members.
    Literal dots/tilde in YAML keys are escaped as ~1/~0, preserving flat override
    keys such as ``training.batch_size`` rather than creating nested mappings.
    """
    result: list[dict[str, Any]] = []

    def visit(path: str, value: Any) -> None:
        if isinstance(value, Mapping) and value:
            for key, item in value.items():
                encoded = str(key).replace('~', '~0').replace('.', '~1')
                visit(f'{path}.{encoded}' if path else encoded, item)
        elif isinstance(value, (tuple, list)) and any(isinstance(item, Mapping) for item in value):
            for index, item in enumerate(value):
                visit(f'{path}.{index}', item)
        else:
            result.append(_spec(path, value, group=path.split('.', 1)[0]))

    if mapping:
        visit('', mapping)
    return result


def apply_plan_values(plan: Mapping[str, Any], mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Copy current control values into the YAML plan without running algorithms."""
    result = copy.deepcopy(dict(plan))

    def key(container: Any, token: str) -> Any:
        token = token.replace('~1', '.').replace('~0', '~')
        if isinstance(container, list):
            return int(token)
        return next((existing for existing in container if str(existing) == token), token)

    for path, value in mapping.items():
        current = result
        parts = path.split('.')
        for token in parts[:-1]:
            current = current[key(current, token)]
        current[key(current, parts[-1])] = copy.deepcopy(value)
    return result
