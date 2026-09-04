"""``lamssi_tools`` the tool-declaration foundation: schema models, the ``@tool`` decorator, registry, context, and dispatch."""

from lamssi_tools.context import (  # noqa: F401
    CapabilityContext,
    CapabilityMissing,
)
from lamssi_tools.decorator import (  # noqa: F401
    Array,
    Bool,
    Float,
    Int,
    Object,
    Param,
    Str,
    tool,
)
from lamssi_tools.dispatch import (  # noqa: F401
    Dispatcher,
    UnknownDispatchTarget,
    inline_dispatcher,
    strict_dispatcher,
)
from lamssi_tools.errors import (  # noqa: F401
    LamssiError,
    err,
)
from lamssi_tools.models import (  # noqa: F401
    Expose,
    JSONType,
    ToolDefinition,
    ToolExecutionError,
    ToolParameter,
    build_tools_openai_schema,
    strip_pydantic_urls,
)
from lamssi_tools.registry import (  # noqa: F401
    ToolRegistry,
)
from lamssi_tools.mounted import (  # noqa: F401
    MountedRegistry,
    ToolNameConflictError,
)
from lamssi_tools.sources import (  # noqa: F401
    CallableSource,
    DirectorySource,
    InstanceSource,
    ModuleSource,
    ToolPair,
    ToolSource,
)

__all__ = [
    "LamssiError",
    "err",
    "CapabilityContext",
    "CapabilityMissing",
    "Dispatcher",
    "UnknownDispatchTarget",
    "inline_dispatcher",
    "strict_dispatcher",
    "Expose",
    "JSONType",
    "ToolParameter",
    "ToolDefinition",
    "ToolExecutionError",
    "build_tools_openai_schema",
    "strip_pydantic_urls",
    "tool",
    "Param",
    "Float",
    "Int",
    "Str",
    "Bool",
    "Array",
    "Object",
    "ToolRegistry",
    "MountedRegistry",
    "ToolNameConflictError",
    "ToolSource",
    "ToolPair",
    "CallableSource",
    "InstanceSource",
    "ModuleSource",
    "DirectorySource",
]
