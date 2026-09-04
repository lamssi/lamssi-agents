"""Decorator-based tool authoring: the ``@tool`` decorator and ``Param``/``Float``/``Int``/... helpers."""

from __future__ import annotations

import inspect
import logging
import re
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from lamssi_tools.models import (
    ApprovalName,
    Expose,
    GuardRoleName,
    JSONType,
    ToolDefinition,
    ToolParameter,
)

log = logging.getLogger(__name__)


class Param:
    """Constraint metadata for advanced ``typing.Annotated`` parameters.

    Most tools can use ordinary Python annotations and the friendly parameter
    helpers through ``@tool(parameters={...})``. Use ``Param`` directly when a
    reusable ``Annotated`` type alias is useful.

    Args:
        description: Meaning of the argument shown in the model-facing schema.
        enum: Allowed literal values.
        examples: Representative valid values included in the schema.
        ge: Inclusive numeric minimum.
        le: Inclusive numeric maximum.
        gt: Exclusive numeric minimum.
        lt: Exclusive numeric maximum.
        multiple_of: Required numeric step.
        min_length: Minimum string length.
        max_length: Maximum string length.
        pattern: Regular expression required for string values.
        min_items: Minimum array length.
        max_items: Maximum array length.
        unique_items: Whether array elements must be unique.

    Example:
        Describe and validate a tool parameter::

            from typing import Annotated

            position: Annotated[float, Param("Target in mm", ge=0, le=50)]
    """

    __slots__ = (
        "description",
        "enum",
        "examples",
        "ge",
        "le",
        "gt",
        "lt",
        "multiple_of",
        "min_length",
        "max_length",
        "pattern",
        "min_items",
        "max_items",
        "unique_items",
    )

    def __init__(
        self,
        description: str = "",
        *,
        enum: Optional[List[Any]] = None,
        examples: Optional[List[Any]] = None,
        ge: Optional[float] = None,
        le: Optional[float] = None,
        gt: Optional[float] = None,
        lt: Optional[float] = None,
        multiple_of: Optional[float] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        min_items: Optional[int] = None,
        max_items: Optional[int] = None,
        unique_items: Optional[bool] = None,
    ) -> None:
        self.description = description
        self.enum = enum
        self.examples = examples
        self.ge = ge
        self.le = le
        self.gt = gt
        self.lt = lt
        self.multiple_of = multiple_of
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = pattern
        self.min_items = min_items
        self.max_items = max_items
        self.unique_items = unique_items

    def __repr__(self) -> str:
        parts = [f"description={self.description!r}"]
        for slot in self.__slots__:
            if slot == "description":
                continue
            val = getattr(self, slot)
            if val is not None:
                parts.append(f"{slot}={val!r}")
        return f"Param({', '.join(parts)})"


class _ParameterSpec:
    """Typed metadata consumed by ``@tool(parameters={...})``."""

    __slots__ = ("python_type", "metadata")

    def __init__(
        self,
        python_type: type,
        description: str = "",
        **kwargs: Any,
    ) -> None:
        self.python_type = python_type
        self.metadata = Param(description, **kwargs)

    def __repr__(self) -> str:
        name = {
            str: "Str",
            int: "Int",
            float: "Float",
            bool: "Bool",
            list: "Array",
            dict: "Object",
        }.get(self.python_type, "Parameter")
        return f"{name}({self.metadata!r})"


def Float(description: str = "", **kwargs: Any) -> _ParameterSpec:
    """Describe a ``float`` argument in ``@tool(parameters={...})``."""
    return _ParameterSpec(float, description, **kwargs)


def Int(description: str = "", **kwargs: Any) -> _ParameterSpec:
    """Describe an ``int`` argument in ``@tool(parameters={...})``."""
    return _ParameterSpec(int, description, **kwargs)


def Str(description: str = "", **kwargs: Any) -> _ParameterSpec:
    """Describe a ``str`` argument in ``@tool(parameters={...})``."""
    return _ParameterSpec(str, description, **kwargs)


def Bool(description: str = "", **kwargs: Any) -> _ParameterSpec:
    """Describe a ``bool`` argument in ``@tool(parameters={...})``."""
    return _ParameterSpec(bool, description, **kwargs)


def Array(
    description: str = "",
    **kwargs: Any,
) -> _ParameterSpec:
    """Describe a ``list[T]`` argument in ``@tool(parameters={...})``."""
    return _ParameterSpec(list, description, **kwargs)


def Object(description: str = "", **kwargs: Any) -> _ParameterSpec:
    """Describe a ``dict`` argument in ``@tool(parameters={...})``."""
    return _ParameterSpec(dict, description, **kwargs)


_PYTHON_TO_JSON: Dict[type, JSONType] = {
    str: JSONType.STRING,
    int: JSONType.INTEGER,
    float: JSONType.NUMBER,
    bool: JSONType.BOOLEAN,
    list: JSONType.ARRAY,
    dict: JSONType.OBJECT,
}


#: Param constraint attrs that map to the same JSON-schema key.
_CONSTRAINT_NAMES = (
    "enum",
    "examples",
    "multiple_of",
    "min_length",
    "max_length",
    "pattern",
    "min_items",
    "max_items",
    "unique_items",
)
#: Comparison operators, renamed to their JSON-schema keys.
_CONSTRAINT_RENAMES = {
    "ge": "minimum",
    "le": "maximum",
    "gt": "exclusive_minimum",
    "lt": "exclusive_maximum",
}


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Unwrap either spelling of ``Optional[X]``."""
    import types as _types

    is_union = get_origin(annotation) is Union or isinstance(
        annotation, _types.UnionType
    )
    if is_union:
        args = get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and len(args) == 2:
            return non_none[0], True
    return annotation, False


def _resolve_json_type(annotation: Any) -> JSONType:
    """Map a Python annotation to its JSON schema type."""
    if annotation is Any:
        return JSONType.ANY
    if annotation in _PYTHON_TO_JSON:
        return _PYTHON_TO_JSON[annotation]

    origin = get_origin(annotation)
    if origin is list:
        return JSONType.ARRAY
    if origin is dict:
        return JSONType.OBJECT

    inner, was_optional = _unwrap_optional(annotation)
    if was_optional:
        return _resolve_json_type(inner)

    return JSONType.STRING


def _infer_from_default(default: Any) -> type:
    """Infer an unannotated parameter type from a concrete default."""
    if default is inspect.Parameter.empty or default is None:
        return str
    return type(default) if type(default) in _PYTHON_TO_JSON else str


def _resolve_annotation(
    annotation: Any,
) -> tuple[JSONType, Optional[JSONType], Optional[Param]]:
    """Resolve a full annotation into ``(json_type, items_type, Param)``."""
    param_meta: Optional[Param] = None

    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        base = args[0]
        for extra in args[1:]:
            if isinstance(extra, Param):
                param_meta = extra
                break
        annotation = base

    annotation, _ = _unwrap_optional(annotation)
    json_type = _resolve_json_type(annotation)

    items_type: Optional[JSONType] = None
    if json_type == JSONType.ARRAY:
        inner_args = get_args(annotation)
        if inner_args:
            items_type = _resolve_json_type(inner_args[0])

    return json_type, items_type, param_meta


def _parse_docstring(func: Callable) -> tuple[str, Dict[str, str]]:
    """Extract description and per-parameter docs from a Google-style docstring."""
    doc = inspect.getdoc(func) or ""
    if not doc:
        return "", {}

    lines = doc.split("\n")
    desc_lines: List[str] = []
    param_docs: Dict[str, str] = {}

    in_args = False
    desc_done = False
    current_param: Optional[str] = None

    _SECTION_HEADERS = frozenset(
        {
            "args:",
            "arguments:",
            "parameters:",
            "params:",
        }
    )
    _END_HEADERS = frozenset(
        {
            "returns:",
            "return:",
            "raises:",
            "yields:",
            "note:",
            "notes:",
            "example:",
            "examples:",
            "see also:",
        }
    )

    for line in lines:
        stripped = line.strip().lower()

        # description = summary + pre-Args prose only; a recognised section header ends it.
        if stripped in _SECTION_HEADERS:
            in_args = True
            desc_done = True
            current_param = None
            continue
        if stripped in _END_HEADERS:
            in_args = False
            desc_done = True
            current_param = None
            continue

        if in_args:
            match = re.match(r"(\w+)(?:\s*\([^)]*\))?\s*:\s*(.*)", line.strip())
            if match:
                current_param = match.group(1)
                param_docs[current_param] = match.group(2).strip()
            elif current_param and line.strip():
                param_docs[current_param] += " " + line.strip()
        elif not desc_done:
            desc_lines.append(line.strip())

    description = " ".join(l for l in desc_lines if l).strip()
    return description, param_docs


def _get_type_hints_with_extras(func: Callable) -> Dict[str, Any]:
    """Resolve type hints with metadata, falling back per annotation."""
    try:
        return get_type_hints(func, include_extras=True)
    except (TypeError, SyntaxError):
        pass

    func_name = getattr(func, "__qualname__", None) or getattr(
        func, "__name__", "<anon>"
    )
    raw = getattr(func, "__annotations__", {})
    resolved: Dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, str):
            try:
                resolved[k] = eval(v, getattr(func, "__globals__", {}))  # noqa: S307
            except Exception as exc:
                log.error(
                    "tool %s: annotation for parameter %r failed to evaluate "
                    "(%s: %s): falling back to Any. Fix the annotation.",
                    func_name,
                    k,
                    type(exc).__name__,
                    exc,
                )
                resolved[k] = Any
        else:
            resolved[k] = v
    return resolved


def _as_type_tuple(value: Any) -> tuple:
    """Normalise ``requires=`` to a tuple, accepting a bare type."""
    if value is None:
        return ()
    if isinstance(value, type):
        return (value,)
    return tuple(v for v in value if isinstance(v, type))


def _as_str_tuple(value: Any) -> tuple:
    """Normalise ``keywords=`` and split a bare string into words."""
    if not value:
        return ()
    if isinstance(value, str):
        return tuple(value.split())
    return tuple(str(v) for v in value if str(v).strip())


def _schema_parameters(
    signature_parameters: list[tuple[str, inspect.Parameter]],
    *,
    inject_context: bool,
    tool_name: str,
) -> list[tuple[str, inspect.Parameter]]:
    """Return parameters that can be supplied by a model tool call."""
    parameters = list(signature_parameters)
    if parameters and parameters[0][0] in ("self", "cls"):
        parameters.pop(0)

    if inject_context:
        if not parameters:
            raise TypeError(
                f"tool {tool_name!r} enables inject_context but has no context parameter"
            )
        context_name, context_parameter = parameters.pop(0)
        if context_parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise TypeError(
                f"tool {tool_name!r} context parameter {context_name!r} must accept "
                "a positional value"
            )

    visible = [
        (name, parameter)
        for name, parameter in parameters
        if parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    positional_only = [
        name
        for name, parameter in visible
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY
    ]
    if positional_only:
        raise TypeError(
            f"tool {tool_name!r} has positional-only model argument(s): "
            f"{', '.join(positional_only)}; tool arguments are passed by name"
        )
    return visible


def tool(
    fn: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Mapping[str, Any]] = None,
    group: Any = None,
    dispatch: Optional[str] = None,
    inject_context: bool = False,
    expose: Optional[Expose] = None,
    returns: Optional[Dict[str, Any]] = None,
    approval: ApprovalName = "always",
    guard_role: Optional[GuardRoleName] = None,
    requires: Any = None,
    keywords: Any = None,
    safe_when: Optional[Dict[str, Any]] = None,
    truncation: Optional[int] = None,
    truncation_hint: Optional[str] = None,
    truncation_side: str = "middle",
) -> Callable:
    """Declare a function as a tool.

    Introspects the function's **signature**, **type hints**, and **docstring**
    to build a :class:`ToolDefinition` automatically.

    Args:
        fn: The function to decorate (supports both bare ``@tool`` and
            ``@tool(...)`` forms).
        name: Override the tool name (defaults to ``fn.__name__``).
        description: Override the description (defaults to the parsed docstring).
        parameters: Optional argument metadata keyed by the exact function
            parameter name. Values come from :func:`Str`, :func:`Int`,
            :func:`Float`, :func:`Bool`, :func:`Array`, or :func:`Object`.
            Python annotations remain ordinary types and defaults remain real
            defaults; unknown names and helper/type mismatches are rejected.
        group: Free-form grouping tag for documentation/UI ordering; a string
            or anything with a ``.value``.
        dispatch: Opaque tag naming the execution context the body needs, e.g.
            ``"worker"``. ``None`` (default) runs inline. The host maps tags via
            a :data:`~lamssi_tools.dispatch.Dispatcher`; an unclaimed tag runs
            inline unless the host installed ``strict_dispatcher``, which raises.
        inject_context: When True, the first parameter receives a
            :class:`~lamssi_tools.context.CapabilityContext` and is excluded from the
            schema. Explicit by design: never inferred from a parameter name.
        expose: Where the tool is visible: combine ``Expose.HOST``,
            ``Expose.AGENT``, ``Expose.MCP`` with ``|``. Defaults to ``HOST``
            only: agent/MCP visibility is always an explicit opt-in.
        returns: JSON Schema for the return value (documentation and logging).
        approval: ``"always"`` (default), ``"never"``, or ``"conditional"`` -
            gated unless ``safe_when`` accepts the call's arguments.
        guard_role: How the loop guard treats repeats of this tool -
            ``"always_allowed"``, ``"recovery"`` or ``"repeatable"``.
        requires: A capability type, or tuple of them, this tool cannot work
            without. A host that registered none of them never sees the tool,
            not even in its schema.
        keywords: Words a person might use for this tool that its description
            does not: ``"save"`` for one that says "write". Only used to match
            requests against tools with a deferred schema; never sent to the
            model, so free. A string or an iterable of them.
        safe_when: Argument pattern under which a ``"conditional"`` call is safe
            and skips approval, e.g. ``{"action": ["query", "inspect"]}``.
        truncation: Per-tool cap on result characters.
        truncation_hint: Guidance appended when this tool's result is truncated.
        truncation_side: Which end to keep when clipping: "middle", "head", or "tail".

    Returns:
        The decorated callable, carrying its generated ``ToolDefinition``.

    Raises:
        TypeError: If ``parameters=`` uses a helper for the wrong Python type.
        ValueError: If the tool name or configuration is invalid.

    Example:
        Describe and constrain an application tool without changing its normal
        Python signature::

            @tool(
                expose=Expose.AGENT,
                approval="never",
                parameters={
                    "query": Str("Text to find.", min_length=1),
                    "limit": Int("Maximum matches.", ge=1, le=100),
                },
            )
            def search(query: str, limit: int = 10) -> list[str]:
                '''Search application records.'''
                return []
    """

    def decorator(func: Callable) -> Callable:
        _name = name or func.__name__
        doc_desc, doc_params = _parse_docstring(func)
        _description = description or doc_desc or f"Tool: {_name}"

        _group_value: Optional[str] = None
        if group is not None:
            _group_value = group.value if hasattr(group, "value") else str(group)

        sig = inspect.signature(func)
        hints = _get_type_hints_with_extras(func)
        sig_params = list(sig.parameters.items())
        parameter_specs = dict(parameters or {})

        e = Expose.HOST if expose is None else Expose(int(expose))
        _agent_visible = bool(e & Expose.AGENT)
        _mcp_visible = bool(e & Expose.MCP)

        schema_parameters = _schema_parameters(
            sig_params,
            inject_context=inject_context,
            tool_name=_name,
        )
        schema_names = {pname for pname, _ in schema_parameters}
        unknown_specs = sorted(set(parameter_specs) - schema_names)
        if unknown_specs:
            raise ValueError(
                f"tool {_name!r} parameters= names unknown argument(s): "
                f"{', '.join(unknown_specs)}"
            )
        invalid_specs = sorted(
            name
            for name, spec in parameter_specs.items()
            if not isinstance(spec, _ParameterSpec)
        )
        if invalid_specs:
            raise TypeError(
                f"tool {_name!r} parameters= values must come from Str(), Int(), "
                f"Float(), Bool(), Array(), or Object(); invalid: "
                f"{', '.join(invalid_specs)}"
            )

        params: List[ToolParameter] = []
        for pname, param in schema_parameters:
            annotation = hints.get(pname) or _infer_from_default(param.default)
            if isinstance(annotation, _ParameterSpec):
                raise TypeError(
                    f"tool {_name!r} argument {pname!r} uses a parameter helper "
                    "as its type annotation. Use an ordinary Python type and move "
                    f"the helper to @tool(parameters={{\"{pname}\": ...}})."
                )
            json_type, items_type, param_meta = _resolve_annotation(annotation)
            configured = parameter_specs.get(pname)
            if configured is not None:
                if param_meta is not None:
                    raise ValueError(
                        f"tool {_name!r} argument {pname!r} has metadata in both "
                        "Annotated and parameters="
                    )
                expected_type = _resolve_json_type(configured.python_type)
                if json_type != expected_type:
                    raise TypeError(
                        f"tool {_name!r} argument {pname!r} is annotated "
                        f"{json_type.value}, but its parameters= helper "
                        f"describes {expected_type.value}"
                    )
                param_meta = configured.metadata

            has_default = param.default is not inspect.Parameter.empty
            default_val = param.default if has_default else None

            pdesc = ""
            if param_meta and param_meta.description:
                pdesc = param_meta.description
            elif pname in doc_params:
                pdesc = doc_params[pname]

            tp_kwargs: Dict[str, Any] = {
                "name": pname,
                "type": json_type,
                "description": pdesc,
                "required": not has_default,
            }
            if has_default:
                tp_kwargs["default"] = default_val

            if json_type == JSONType.ARRAY and items_type is not None:
                tp_kwargs["items"] = ToolParameter(
                    name="item",
                    type=items_type,
                    description="",
                )

            # Apply Param constraints; only the comparison operators are renamed to their JSON-schema keys.
            if param_meta:
                for param_attr in _CONSTRAINT_NAMES:
                    val = getattr(param_meta, param_attr, None)
                    if val is not None:
                        tp_kwargs[param_attr] = val
                for param_attr, tp_field in _CONSTRAINT_RENAMES.items():
                    val = getattr(param_meta, param_attr, None)
                    if val is not None:
                        tp_kwargs[tp_field] = val

            params.append(ToolParameter(**tp_kwargs))

        tool_def = ToolDefinition(
            name=_name,
            description=_description,
            parameters=params,
            group=_group_value,
            returns=returns,
            expose_to_agent=_agent_visible,
            expose_to_mcp=_mcp_visible,
            dispatch=dispatch,
            approval=approval,
            guard_role=guard_role,
            requires=_as_type_tuple(requires),
            keywords=_as_str_tuple(keywords),
            safe_when=safe_when,
            truncation=truncation,
            truncation_hint=truncation_hint,
            truncation_side=truncation_side,
        )

        func._tool_definition = tool_def  # type: ignore[attr-defined]
        func._tool_inject_context = inject_context  # type: ignore[attr-defined]

        return func

    if fn is not None:
        return decorator(fn)
    return decorator
