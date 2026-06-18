# -*- coding: utf-8 -*-
"""工具 Schema 生成"""
from typing import Any, get_type_hints, get_origin

TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def python_type_to_json_type(py_type: Any) -> str:
    origin = get_origin(py_type)
    py_type = origin or py_type
    return TYPE_MAP.get(py_type, "string")


def generate_tool_schema(name: str, description: str, params: dict[str, tuple[type, str]] | None = None) -> dict:
    if params is None:
        params = {}
    properties = {}
    required = []
    for param_name, (param_type, param_desc) in params.items():
        json_type = python_type_to_json_type(param_type)
        properties[param_name] = {
            "type": json_type,
            "description": param_desc,
        }
        required.append(param_name)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
