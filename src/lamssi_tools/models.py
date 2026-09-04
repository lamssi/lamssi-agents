"""Tool data models: the single source of truth for tool schemas."""

from __future__ import annotations

import re
from enum import Enum, IntFlag
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
)

from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    ConfigDict,
    model_validator,
    field_validator,
    create_model,
)

from lamssi_tools.errors import LamssiError

#: Valid loop-guard role tags; mirrors GuardRole in lamssi_agents.tooling.guard.
GuardRoleName = Literal["normal", "always_allowed", "recovery", "repeatable"]

#: Valid per-tool approval declarations; single source for approval.DECLARATIONS.
ApprovalName = Literal[
    "never",        # never asks for approval
    "conditional",  # asks unless safe_when matches the call's arguments
    "always",       # always asks for approval
]

# ``group`` is free-form (not an enum) since access control is by exact tool name, not group.


class Expose(IntFlag):
    """Bit flags selecting the surfaces where a tool is visible.

    Attributes:
        NONE: No public surface.
        HOST: Embedding application's own command surface.
        AGENT: Model-facing Agent tool schema.
        MCP: Remote MCP clients.
        ALL: Host, Agent, and MCP surfaces combined.

    Bit values are never renumbered: a persisted bitmask outlives the code
    that wrote it, so reassigning a member would silently change stored data.

    Example:
        Expose a tool to the Agent and MCP::

            @tool(expose=Expose.AGENT | Expose.MCP)
            def inspect_status() -> dict:
                ...
    """

    NONE = 0
    HOST = 1
    AGENT = 2
    MCP = 4
    ALL = 1 | 2 | 4


class JSONType(str, Enum):
    """Canonical JSON Schema type identifiers."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"
    # ``ANY`` is our addition: the unresolvable-type fallback, omitting ``type`` so bugs surface as body errors, not misleading rejections.
    ANY = "any"


def _json_type_to_python(t: JSONType) -> type:
    return {
        JSONType.STRING: str,
        JSONType.INTEGER: int,
        JSONType.NUMBER: float,
        JSONType.BOOLEAN: bool,
        JSONType.ARRAY: list,
        JSONType.OBJECT: dict,
        JSONType.NULL: type(None),
        JSONType.ANY: Any,  # type: ignore[dict-item]
    }[t]


#: The identity rule for a tool or parameter name; compiled once, applied to both.
_SNAKE_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")


def _require_snake_case(value: str, kind: str) -> str:
    """Return *value* if it is snake_case, else raise with *kind* naming the field."""
    if not _SNAKE_NAME.fullmatch(value):
        raise ValueError(f"{kind} must be snake_case [a-z][a-z0-9_]{{0,63}}")
    return value


class _Constraint(NamedTuple):
    """One scalar constraint and the three names it is spelled by."""

    field: str  # ToolParameter attribute
    json_key: str  # JSON Schema keyword
    field_kwarg: str  # pydantic Field keyword
    types: Tuple[JSONType, ...]  # parameter types it applies to


_NUMERIC = (JSONType.INTEGER, JSONType.NUMBER)

#: Scalar constraints shared by the JSON schema and pydantic validator emitters so the two cannot disagree; ``uniqueItems`` and array ``items`` aren't plain copies and stay explicit in each.
_SCALAR_CONSTRAINTS: Tuple[_Constraint, ...] = (
    _Constraint("min_length", "minLength", "min_length", (JSONType.STRING,)),
    _Constraint("max_length", "maxLength", "max_length", (JSONType.STRING,)),
    _Constraint("pattern", "pattern", "pattern", (JSONType.STRING,)),
    _Constraint("minimum", "minimum", "ge", _NUMERIC),
    _Constraint("maximum", "maximum", "le", _NUMERIC),
    _Constraint("exclusive_minimum", "exclusiveMinimum", "gt", _NUMERIC),
    _Constraint("exclusive_maximum", "exclusiveMaximum", "lt", _NUMERIC),
    _Constraint("multiple_of", "multipleOf", "multiple_of", _NUMERIC),
    _Constraint("min_items", "minItems", "min_length", (JSONType.ARRAY,)),
    _Constraint("max_items", "maxItems", "max_length", (JSONType.ARRAY,)),
)


class ToolParameter(BaseModel):
    """A single tool argument, described in JSON Schema terms.

    Supports primitives, arrays with typed items, and free-form objects.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Parameter name (snake_case)")
    type: JSONType = Field(..., description="JSON Schema type")
    description: str = Field("", description="Parameter description")

    required: bool = Field(True, description="Whether the parameter is required")
    default: Any = Field(
        None, description="Default value if omitted (optional parameters only)"
    )

    # Common constraints
    enum: Optional[List[Any]] = Field(None, description="Allowed literal values")
    examples: Optional[List[Any]] = Field(None, description="Example values")

    # String constraints
    min_length: Optional[int] = Field(None, ge=0)
    max_length: Optional[int] = Field(None, ge=0)
    pattern: Optional[str] = Field(None, description="Regex pattern for strings")

    # Numeric constraints
    minimum: Optional[float] = Field(None)
    maximum: Optional[float] = Field(None)
    exclusive_minimum: Optional[float] = Field(None)
    exclusive_maximum: Optional[float] = Field(None)
    multiple_of: Optional[float] = Field(None, gt=0)

    # Array constraints
    min_items: Optional[int] = Field(None, ge=0)
    max_items: Optional[int] = Field(None, ge=0)
    unique_items: Optional[bool] = Field(None)
    items: Optional["ToolParameter"] = Field(
        None,
        description="Schema for array items (required when type=array for typed arrays)",
    )

    # ----- Validators -----

    @field_validator("name")
    @classmethod
    def _validate_param_name(cls, v: str) -> str:
        return _require_snake_case(v, "Parameter name")

    @model_validator(mode="after")
    def _validate_consistency(self) -> "ToolParameter":
        if self.required and self.default is not None:
            self.required = False
        return self

    # ----- Schema emitters -----

    def to_json_schema(self) -> Dict[str, Any]:
        """Emit a JSON Schema snippet for this parameter; ``JSONType.ANY`` omits ``type`` so the model can send any JSON value."""
        s: Dict[str, Any] = {}
        if self.type != JSONType.ANY:
            s["type"] = self.type.value
        if self.description:
            s["description"] = self.description
        if self.enum is not None:
            s["enum"] = self.enum
        if self.examples is not None:
            s["examples"] = self.examples

        if (not self.required) and (self.default is not None):
            s["default"] = self.default

        # Items sub-schema goes first; the scalar-constraint loop below extends the same dict, and uniqueItems is emitted last.
        if self.type == JSONType.ARRAY:
            s["items"] = self.items.to_json_schema() if self.items is not None else {}

        for c in _SCALAR_CONSTRAINTS:
            if self.type in c.types:
                value = getattr(self, c.field)
                if value is not None:
                    s[c.json_key] = value

        if self.type == JSONType.ARRAY and self.unique_items is not None:
            s["uniqueItems"] = self.unique_items

        # An object param stays a bare {"type": "object"}: never additionalProperties:false, which would tell a strict model the dict must be empty.
        return s

    def to_python_annotation(self) -> Any:
        """Best-effort conversion to Python typing annotation for runtime validation."""
        if self.enum is not None and len(self.enum) > 0:
            try:
                return Literal[tuple(self.enum)]  # type: ignore[misc]
            except Exception:
                pass

        if self.type == JSONType.ARRAY:
            if self.items is None:
                return List[Any]
            return List[self.items.to_python_annotation()]  # type: ignore[index]

        if self.type == JSONType.OBJECT:
            return Dict[str, Any]

        return _json_type_to_python(self.type)


ToolParameter.model_rebuild()


class ToolDefinition(BaseModel):
    """A tool the model can call."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Tool name (snake_case)")
    description: str = Field(..., description="What the tool does")
    parameters: List[ToolParameter] = Field(
        default_factory=list, description="Input parameters"
    )
    group: Optional[str] = Field(
        None,
        description=(
            "Free-form grouping tag for documentation and UI ordering (e.g. "
            "'system', 'files', 'code'). Not a gate: what a model may call is "
            "decided by exact tool name."
        ),
    )
    returns: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional JSON Schema for return value (for documentation/logging)",
    )
    expose_to_agent: bool = Field(
        False,
        description="If True, the tool is offered to the AI agent.",
    )
    expose_to_mcp: bool = Field(
        False,
        description=(
            "If True, the tool is offered to MCP clients. Opt-in: a tool declared "
            "for internal use should not become remotely callable by default."
        ),
    )
    dispatch: Optional[str] = Field(
        None,
        description=(
            "Opaque tag naming the execution context this body needs, resolved by "
            "the host's dispatcher (e.g. 'worker', 'gui'). None (default) means "
            "inline on the calling thread: correct for any pure function, and "
            "the only safe default for an arbitrary host."
        ),
    )
    approval: ApprovalName = Field(
        "always",
        description=(
            "Human-in-the-loop policy: 'always' = require approval (default), "
            "'never' = never require it (read-only / safe), 'conditional' = "
            "require it unless 'safe_when' accepts the call's arguments."
        ),
    )
    requires: Tuple[type, ...] = Field(
        (),
        description=(
            "Capability types required by this tool. The tool is omitted from the "
            "model schema when the host has not registered them."
        ),
    )
    keywords: Tuple[str, ...] = Field(
        (),
        description=(
            "Words a person might use for this tool that its description does not. "
            "'save' for a tool that says 'write', 'rename' for one that says 'edit'. "
            "Used when matching a request against tools whose schema was deferred, "
            "and never sent to the model: so they cost nothing per turn, and the "
            "list can be as long as it is useful. Only the tool's author knows "
            "these; no amount of matching on the description can invent them."
        ),
    )
    guard_role: Optional[GuardRoleName] = Field(
        None,
        description=(
            "How the loop guard should treat this tool: 'always_allowed' "
            "(never blocked), 'recovery' (permitted after an error streak), "
            "'repeatable' (repeats are expected). None = ordinary."
        ),
    )
    safe_when: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Argument pattern under which a 'conditional' call is safe and skips "
            "approval, e.g. {'action': ['query', 'inspect']}. Every key must match "
            "the call's argument; a list means any of these."
        ),
    )
    truncation: Optional[int] = Field(
        None,
        description="Per-tool cap on result characters before truncation; None = the runtime default.",
    )
    truncation_hint: Optional[str] = Field(
        None,
        description="Guidance appended when this tool's result is truncated (e.g. how to narrow the query).",
    )
    truncation_side: str = Field(
        "middle",
        description="Which end to keep when clipping this tool's result: 'middle', 'head', or 'tail'.",
    )

    @field_validator("approval")
    @classmethod
    def _validate_approval(cls, v: str) -> str:
        allowed = {"never", "conditional", "always"}
        if v not in allowed:
            raise ValueError(f"approval must be one of {allowed}, got {v!r}")
        return v

    @field_validator("name")
    @classmethod
    def _validate_tool_name(cls, v: str) -> str:
        return _require_snake_case(v, "Tool name")

    def input_schema(self) -> Dict[str, Any]:
        """Shared JSON Schema; adapters apply their provider-specific wrapping."""
        props: Dict[str, Any] = {}
        required: List[str] = []
        for p in self.parameters:
            props[p.name] = p.to_json_schema()
            if p.required:
                required.append(p.name)

        schema: Dict[str, Any] = {
            "type": "object",
            "properties": props,
            "additionalProperties": False,
        }
        if required:
            schema["required"] = required
        return schema

    def to_openai_tool(self) -> Dict[str, Any]:
        """OpenAI-style tool schema: ``{"type":"function","function":{...}}``."""
        fn: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema(),
        }
        if self.returns:
            fn["returns"] = self.returns
        return {"type": "function", "function": fn}

    def build_args_model(self) -> Type[BaseModel]:
        """Build a Pydantic model for runtime argument validation."""
        return _create_args_model(self.name, self.parameters, forbid_extra=True)

    def get_param_defaults(self) -> Dict[str, Any]:
        """Default values for optional parameters."""
        return {p.name: p.default for p in self.parameters if not p.required}


def _require_unique_items(v: Any) -> Any:
    """Reject a list with duplicate items (JSON Schema ``uniqueItems``); tested against a list, not a set, so unhashable JSON values compare by equality."""
    if isinstance(v, list):
        seen: List[Any] = []
        for item in v:
            if item in seen:
                raise ValueError("array items must be unique")
            seen.append(item)
    return v


def _create_args_model(
    model_name: str,
    params: Sequence["ToolParameter"],
    forbid_extra: bool = True,
) -> Type[BaseModel]:
    """Create a Pydantic v2 model at runtime with constraints from Field(...)."""
    fields: Dict[str, Tuple[Any, Any]] = {}

    for p in params:
        ann = p.to_python_annotation()

        fk: Dict[str, Any] = {}
        if p.description:
            fk["description"] = p.description

        # Same table the schema is built from, so the validator enforces exactly what it advertises (e.g. exclusiveMinimum <-> gt).
        for c in _SCALAR_CONSTRAINTS:
            if p.type in c.types:
                value = getattr(p, c.field)
                if value is not None:
                    fk[c.field_kwarg] = value

        # uniqueItems is enforced by a validator, not a Field keyword.
        if p.type == JSONType.ARRAY and p.unique_items:
            ann = Annotated[ann, AfterValidator(_require_unique_items)]

        if p.required:
            fields[p.name] = (ann, Field(..., **fk))
        else:
            default = p.default
            # Wrap in Optional when the default is None so Pydantic v2 accepts None as a valid value.
            if default is None:
                ann = Optional[ann]
            fields[p.name] = (ann, Field(default=default, **fk))

    cfg = ConfigDict(extra="forbid" if forbid_extra else "allow")
    Dynamic = create_model(f"{model_name}_Args", __config__=cfg, **fields)  # type: ignore[arg-type]
    return Dynamic


def build_tools_openai_schema(tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
    """Convert tool definitions into OpenAI-style tools list."""
    return [t.to_openai_tool() for t in tools]


class ToolExecutionError(LamssiError, RuntimeError):
    """Raised when a tool fails validation or execution."""

    pass


_PYDANTIC_URL_RE = re.compile(
    r"\s*For further information visit https://errors\.pydantic\.dev/\S*",
)


def strip_pydantic_urls(msg: str) -> str:
    """Remove Pydantic doc-link lines from an error string; public because validation errors reach the model as tool results, and the URL is pure token cost it cannot follow."""
    return _PYDANTIC_URL_RE.sub("", msg)
