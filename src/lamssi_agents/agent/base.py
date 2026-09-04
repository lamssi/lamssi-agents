"""``Agent``: state and override hooks; the behaviour lives in the sibling modules."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Sequence, TypeVar, Union

from lamssi_agents.agent import (
    conversation,
    loop,
)
from lamssi_agents import tool_runtime as tools_mod
from lamssi_agents.agent.control import RunControl
from lamssi_agents.approval import ApprovalPolicy
from lamssi_agents.events import (
    AgentAborted,
    AgentEvent,
    AgentEventCallback,
    AgentEventType,
)
from lamssi_agents.features.base import Feature
from lamssi_agents.history import (
    Compactor,
    get_compaction_strategy,
)
from lamssi_agents.model import Model, ModelInput, adapter_name, model_id, resolve_model
from lamssi_agents.interaction import InteractionHandler
from lamssi_agents.prompt.model import AssembledPrompt, PromptPosition
from lamssi_agents.prompt import SectionRegistry
from lamssi_agents.providers import Message
from lamssi_agents.result import RunResult, usage_since, usage_snapshot
from lamssi_agents.runtime.config import AgentConfig
from lamssi_agents.runtime.scope import RunScope

if TYPE_CHECKING:
    from collections.abc import Callable

    from lamssi_agents.tooling.dedupe import DedupePolicy
    from lamssi_agents.tooling.guard import GuardRole
from lamssi_tools import (
    Expose,
    CapabilityContext,
    MountedRegistry,
    ToolDefinition,
    ToolRegistry,
)

log = logging.getLogger(__name__)

#: Ties a capability to its protocol, so `get`/`provide` are typed rather than Any.
T = TypeVar("T")


def _position_label(position: PromptPosition | int) -> str:
    """Return a named prompt position while preserving custom advanced values."""
    if isinstance(position, PromptPosition):
        return position.name.lower()
    return str(position)


class Agent:
    """Embeddable synchronous model-and-tool loop for one conversation.

    An agent owns its conversation history, model, complete base instructions,
    and explicitly installed tools and features. A bare agent has no tools and
    no implicit system prompt. :meth:`run` returns a structured result;
    :meth:`chat` is the text-only convenience.

    Example:
        Create a small project assistant and run one blocking request::

            from lamssi_agents import Agent, Files, Guidance, SystemTools

            agent = Agent(
                "openai/gpt-5-mini",
                instructions="Help maintain this project.",
                features=[SystemTools(), Guidance(), Files(".")],
            )

            result = agent.run("Explain the project structure.")
            print(result.text)

    Note:
        ``run()`` and ``chat()`` block until the request finishes. Use a
        separate agent for an independent concurrent conversation.
    """

    def __init__(
        self,
        model: Optional[ModelInput] = None,
        *,
        instructions: str = "",
        tools: Sequence[Any] = (),
        capabilities: Optional[Mapping[type, Any]] = None,
        context: Sequence[Any] | SectionRegistry = (),
        features: Sequence[Feature] = (),
        approval: Optional[ApprovalPolicy] = None,
        safe_when: Optional[Mapping[str, Any]] = None,
        interaction: Optional[InteractionHandler] = None,
        tool_sources: Sequence[ToolRegistry] = (),
        only: Optional[Sequence[str]] = None,
        verbose: bool = False,
        log_dir: Union[str, Path, None] = None,
        config: Optional[AgentConfig] = None,
        max_turns: Optional[int] = None,
    ) -> None:
        """Create an agent and install its explicit contributions.

        Args:
            model: A LiteLLM model identifier or a custom ``Model`` adapter.
            instructions: The complete base system instructions. Nothing is
                loaded or appended implicitly.
            tools: ``@tool`` functions, plain functions, or modules of them.
            capabilities: Typed values available to tools through ``ctx.get()``.
            context: Named dynamic :class:`ContextBlock` objects for application
                state. Advanced composition may pass a :class:`SectionRegistry`
                directly.
            features: Explicit :class:`Feature` objects contributing tools,
                capabilities, context blocks, or lifecycle hooks.
            approval: The application's explicit tool-approval policy. The
                fail-closed default rejects gated calls when nobody can answer.
            safe_when: Advanced checks deciding when conditionally gated
                tools may run without approval. Keys are exact tool names. The
                mapping is copied, so later changes stay local to this agent.
            interaction: Typed handler for questions, guard overrides, and budget
                checkpoints.
            tool_sources: Advanced. Live external registries this agent reads
                tools from, in addition to its own. Shared by reference, so the
                host adding or removing a tool changes the next turn.
            only: Fixed allow-list of tool names shown to the model. Every schema
                is re-sent on every call, so this is the cost knob. An unnarrowed
                agent sees later additions to mounted registries automatically.
            verbose: Print each tool call and result as it happens.
            log_dir: Write an append-only JSONL transcript here.
            config: Full runtime configuration for the history budget, compaction
                strategy, tool-result cap, and turn limit. A matching keyword
                argument such as max_turns overrides the config field.
            max_turns: Hard backstop on model/tool turns in one run. Defaults to
                the config value when omitted.

        Raises:
            TypeError: If a feature, capability, context block, or model has an
                unsupported shape.
            ValueError: If configuration is contradictory or invalid.
        """
        base_config = config if config is not None else AgentConfig()

        if max_turns is not None:
            base_config = base_config.merged(max_turns=max_turns)

        self._config = base_config.normalised()
        self._run_lock = threading.Lock()

        self._capabilities = CapabilityContext()
        # Tools reach the running agent with ctx.get(RunScope).
        self._capabilities.register(RunScope, RunScope(self))

        self._registry = ToolRegistry()

        self._tools = MountedRegistry(self._registry, capabilities=self._capabilities)

        for registry in tool_sources:
            self._tools.mount(registry)

        self._prompt = (
            context if isinstance(context, SectionRegistry) else SectionRegistry()
        )

        self._before_turn: list[Any] = []
        self._before_tool: list[Any] = []
        self._after_tool: list[Any] = []
        self._approved_call_handlers: list[Any] = []
        self._feature_event_handlers: list[AgentEventCallback] = []
        self.dedupe: dict[str, Any] = {}
        self.safe_when: dict[str, Any] = dict(safe_when or {})
        self.guard_roles: dict[str, str] = {}

        self._control: RunControl = RunControl(approval=approval)
        if interaction is not None:
            self._control.interaction.handler = interaction

        # The run authority as a capability, for tools that need only it
        # (e.g. write hooks) rather than the whole running agent via RunScope.
        self._capabilities.register(RunControl, self._control)

        #: The conversation logger, subscribed to the run's EventBus. The Agent
        #: owns the subscription lifetime, not the RunControl.
        self._conv_logger: Any = None

        #: What this conversation owns and does not share. The services read the
        #: model, strategy, and config live, so a mid-run swap is seen next request.
        self._conversation = conversation.Conversation(
            conversation.HistoryServices(
                model=lambda: self._model,
                summary_model=lambda: self.summary_model,
                compactor=lambda: self.compactor,
                config=lambda: self._config,
                emit=self.emit,
                abort_event=self._control.aborted,
                fixed_overhead=self._fixed_request_overhead,
                notify_compacted=lambda demoted: self._runtime.on_history_compacted(
                    demoted
                ),
            )
        )

        #: The agent's tool behavior: surface, gates, execution, and result
        #: format, plus the dedupe cache and loop guard for this run.
        self._runtime = tools_mod.ToolRuntime(
            registry=self._tools,
            capabilities=self._capabilities,
            control=self._control,
            emit=self.emit,
            policy=tools_mod.ToolPolicy(
                dedupe=self.dedupe,
                safe_when=self.safe_when,
                guard_roles=self.guard_roles,
                before_tool=self._before_tool,
                after_tool=self._after_tool,
                approved_hooks=self._approved_call_handlers,
                max_chars=self._config.max_tool_result_chars,
            ),
        )

        #: The complete base system instructions. Empty really means empty.
        self.instructions: str = instructions

        self._model: Optional[Model] = None
        self._summary_model: Optional[Model] = None
        if model is not None:
            self._attach_model(resolve_model(model))

        #: History-compaction strategy; a feature or host may replace it.
        self.compactor: Compactor = get_compaction_strategy(self._config.compaction)

        #: Optional behaviour, in the order its hooks run.
        self.features: List[Feature] = []
        for feature in features:
            self.use(feature)

        for protocol, implementation in (capabilities or {}).items():
            self.provide(protocol, implementation)
        if tools:
            self.add_tools(*tools)
        if not isinstance(context, SectionRegistry):
            for block in context:
                self.add_context(block)

        if only is not None:
            self.tool_scope = tools_mod.resolve_scope(only, self)

        if verbose:
            from lamssi_agents.console import printer
            self.add_event_listener(printer())

        if log_dir:
            self.set_conversation_log_dir(log_dir)
            
        self._validate_tools()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} model={self.model_id!r}>"

    @property
    def rules(self):
        """Effective loop-guard rules after applying role overrides."""
        return self._runtime.rules

    @property
    def tool_scope(self) -> Optional[set]:
        """Host-set allow-list narrowing this instance; ``None`` means unnarrowed.

        Always-available tools stay in scope so the agent can recover from a dead end.
        """
        return self._runtime.tool_scope

    @tool_scope.setter
    def tool_scope(self, value: Optional[set]) -> None:
        self._runtime.tool_scope = value

    @property
    def truncator(self):
        """Truncator for one oversize tool result; a feature or host may replace it."""
        return self._runtime.truncator

    @truncator.setter
    def truncator(self, value) -> None:
        self._runtime.truncator = value

    @property
    def guard_rules(self):
        """Loop-guard rule set before role overrides; ``None`` for the default."""
        return self._runtime.guard_rules

    @guard_rules.setter
    def guard_rules(self, value) -> None:
        self._runtime.guard_rules = value

    def _reject_while_running(self, method: str) -> None:
        """Reject *method* while a run is active; compose before running."""
        if self._control.is_running:
            raise RuntimeError(
                f"{method} is not safe during a run; compose before the run "
                "(mount_tools, disable_tool, and abort stay live-safe)"
            )

    def use(self, feature: Feature) -> "Agent":
        """Install one feature after construction.

        Args:
            feature: A :class:`Feature` contributing tools, context, capabilities,
                or runtime hooks.

        Returns:
            This agent, allowing fluent composition.

        Raises:
            TypeError: If ``feature`` is not a :class:`Feature` instance.
        """
        self._reject_while_running("use")
        if not isinstance(feature, Feature):
            raise TypeError(
                f"features must inherit Feature; got {type(feature).__name__}"
            )
        feature.install(self)
        self._attach_feature(feature)
        self._validate_tools()
        return self

    def _attach_feature(self, feature: Feature) -> None:
        """Attach hooks for a feature whose contributions are already available."""
        if type(feature).before_turn is not Feature.before_turn:
            hook = feature.before_turn
            self._before_turn.append(lambda turn, h=hook: h(self, turn))

        if type(feature).before_tool is not Feature.before_tool:
            hook = feature.before_tool
            self._before_tool.append(lambda call, h=hook: h(call, self))
            
        if type(feature).after_tool is not Feature.after_tool:
            after = feature.after_tool
            self._after_tool.append(
                lambda call, result, is_error, h=after: h(call, result, is_error, self)
            )
        if type(feature).on_event is not Feature.on_event:
            self._feature_event_handlers.append(feature.on_event)
        self.features.append(feature)

    def _validate_tools(self) -> None:
        """Reject broken approval checks after composition changes."""
        from lamssi_agents.approval import validate_safe_when

        validate_safe_when(self)

    def add_tools(self, *items: Any) -> "Agent":
        """Register tools owned by this agent.

        Args:
            *items: ``@tool`` callables, plain callables, modules, or bound
                objects whose methods are tools.

        Returns:
            This agent.
        """
        self._reject_while_running("add_tools")
        tools_mod.register_tools(self, items)
        return self

    def expose_tool(self, name: str, expose: Expose) -> "Agent":
        """Set which surfaces an already-registered tool reaches.

        For deciding per-agent what a built-in is offered to: keep a tool off MCP,
        or reveal a host-only one to the model. Like ``add_tools``, this edits this
        agent's registry, never the shared ``@tool``
        definition, so other agents are unaffected. To hide a tool for one run
        instead, use ``disable_tool``.

        Args:
            name: Registered tool name.
            expose: Bit flags describing whether the agent, host, or MCP surface
                may see the tool.

        Returns:
            This agent.

        Example:
            Keep a tool model-visible but off MCP::

                agent.expose_tool("read_file", Expose.AGENT)
        """
        self._registry.set_exposure(name, expose)
        return self

    def provide(self, protocol: type[T], implementation: T) -> "Agent":
        """Provide a typed capability to this Agent's tool context.

        A tool obtains the value with ``ctx.get(protocol)`` or
        ``ctx.require(protocol)``. A bare callable is adapted to a single-method
        protocol; anything else is registered as-is.

        Args:
            protocol: Type used as the lookup key.
            implementation: Application object or compatible callable.

        Returns:
            This agent.
        """
        self._reject_while_running("provide")
        self._capabilities.register(protocol, implementation)
        return self

    def get(self, protocol: type[T]) -> Optional[T]:
        """Return the most recently registered capability for a protocol.

        Args:
            protocol: Type used when the capability was provided.

        Returns:
            The registered implementation, or ``None`` when absent.
        """
        return self._capabilities.get(protocol)

    def add_context(self, block: Any) -> "Agent":
        """Install one named dynamic context block.

        Conditional blocks return an empty string when they do not apply. Use
        :class:`lamssi_agents.prompt.ContextBlock` for application state.

        Args:
            block: A context block or supported prompt-section instance.

        Returns:
            This agent.
        """
        from lamssi_agents.prompt.section import normalize_section

        name, section = normalize_section(block)
        self._prompt.register(name, section)
        return self

    def mount_tools(self, registry: ToolRegistry) -> "Agent":
        """Mount a live, externally owned tool registry.

        Unlike :meth:`add_tools`, mounting does not copy definitions and never
        mutates *registry*. Additions and removals made by its owner are visible
        on the next model turn, and calls execute through its own dispatcher.

        Args:
            registry: Application-owned live registry to read by reference.

        Returns:
            This agent. Mounting the same registry twice is harmless.

        Raises:
            TypeError: If ``registry`` is not a :class:`ToolRegistry`.
            ValueError: If it is the agent's private registry or creates a tool
                name conflict.
        """
        if not isinstance(registry, ToolRegistry):
            raise TypeError(
                "mount_tools() requires a lamssi_tools.ToolRegistry; "
                f"got {type(registry).__name__}"
            )
        if registry is self._registry:
            raise ValueError("the Agent's private tool registry cannot be mounted")
        if not self._tools.mount(registry):
            return self

        try:
            # Reject existing ambiguity now; a conflict introduced later is caught at the next live surface resolution.
            self._tools.list_tools()
            self._validate_tools()
        except Exception:
            self._tools.unmount(registry)
            raise
        return self

    def unmount_tools(self, registry: ToolRegistry) -> bool:
        """Detach a mounted registry without modifying it.

        Args:
            registry: The exact registry instance previously mounted.

        Returns:
            ``True`` when it was mounted and removed; otherwise ``False``.
        """
        return self._tools.unmount(registry)

    def set_tool_dispatcher(self, dispatcher: Any) -> "Agent":
        """Route tool bodies through an application dispatcher.

        This is the narrow advanced seam used by GUI features; the registry and
        executor stay private. The dispatcher receives the tool definition, a
        context-bound callable, and validated keyword arguments. It must return
        the final result synchronously.

        Args:
            dispatcher: Callable deciding where the tool body executes.

        Returns:
            This agent.
        """
        self._reject_while_running("set_tool_dispatcher")
        self._registry.set_dispatcher(dispatcher)
        return self

    def on_approved_call(self, handler: Any) -> "Agent":
        """Observe a call that received approval, for a feature's install step."""
        self._approved_call_handlers.append(handler)
        return self

    def add_dedupe_policies(self, policies: Mapping[str, DedupePolicy]) -> "Agent":
        """Register per-tool dedupe policies, for a feature's install step.

        Example:
            from lamssi_agents.tooling.dedupe import DedupePolicy, arg_subset_signature

            agent.add_dedupe_policies({
                "read_config": DedupePolicy(
                    signature=arg_subset_signature("path"),
                    invalidated_by=frozenset({"write_file"}),
                    invalidation_key=lambda args: args.get("path"),
                ),
            })
        """
        self._reject_while_running("add_dedupe_policies")
        self.dedupe.update(policies)
        return self

    def add_safe_when(
        self, checks: Mapping[str, Callable[[Mapping[str, Any]], bool]]
    ) -> "Agent":
        """Let a tool skip the approval prompt when its arguments look safe.

        For each tool name, give a small function that receives the call's
        arguments and returns True to run it without asking, or False to send it
        to the approval handler. Only tools declared ``approval="conditional"``
        use these, and a function added here overrides any ``safe_when`` declared
        on the tool itself.

        Example:
            #set_power up to 100W runs on its own; higher still asks the user.
            agent.add_safe_when({
                "set_power": lambda args: args.get("watts", 0) <= 100,
            })
        """
        self._reject_while_running("add_safe_when")
        self.safe_when.update(checks)
        return self

    def add_guard_roles(self, roles: Mapping[str, GuardRole | str]) -> "Agent":
        """Register loop-guard roles by tool name, and re-sync the guard.

        Example:
            from lamssi_agents.tooling.guard import GuardRole

            agent.add_guard_roles({"read_power_meter": GuardRole.REPEATABLE})
        """
        self._reject_while_running("add_guard_roles")
        self.guard_roles.update(roles)
        self._runtime._sync_guard()
        return self

    def assemble_prompt(self) -> AssembledPrompt:
        """Assemble the current system prompt with provenance metadata.

        Returns:
            Rendered text plus the name, source, position, cacheability, and
            size of every included block.
        """
        from lamssi_agents.prompt import ContextBlock

        instance: List[Any] = []
        if self.instructions:
            instance.append(ContextBlock(
                "instructions",
                lambda: self.instructions,
                position=PromptPosition.INSTRUCTIONS,
                stable=True,
                source="Agent.instructions",
            ))
        resolved = self._runtime.surface()

        return self._prompt.assemble(
            instance_sections=instance,
            model=self.model_id,
            tools=frozenset(d.name for d in resolved.defs),
        )

    def build_system_prompt(self) -> str:
        """Return the current assembled system prompt as plain text.

        Returns:
            Exactly the text that would be sent as the system message this turn.
        """
        return self.assemble_prompt().text

    def _fixed_request_overhead(self) -> int:
        """Character size of the request's fixed part: system prompt and tool schemas."""
        return len(self.assemble_prompt().text) + tools_mod.schema_json_len(
            self._runtime.all_defs()
        )

    def explain_prompt(self) -> str:
        """Describe the source and ordering of every current prompt block.

        Returns:
            Human-readable table suitable for logs, diagnostics, or a settings UI.
        """
        prompt = self.assemble_prompt()
        if not prompt.parts:
            return "System prompt is empty."
        lines = ["name | source | position | cacheable | chars"]
        lines.extend(
            f"{part.name} | {part.source} | {_position_label(part.position)} | "
            f"{'yes' if part.cacheable else 'no'} | {part.chars}"
            for part in prompt.parts
        )
        return "\n".join(lines)

    def run(self, message: str, *, image: Any = None) -> RunResult:
        """Run one blocking user request and return its structured outcome.

        The message is appended to this conversation. Model and tool turns run
        until the model answers, aborts, errors, or reaches ``max_turns``.

        Args:
            message: User text to append and process.
            image: Optional image input supported by the configured model.

        Returns:
            Response text, run outputs, token usage, turn count, abort state,
            and any reported error.

        Raises:
            RuntimeError: If this same agent is already running.
        """
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError(
                "This Agent is already running. Use a separate Agent for an "
                "independent concurrent conversation."
            )
        try:
            return self._run_once(message, image=image)
        finally:
            self._run_lock.release()

    def _run_once(self, message: str, *, image: Any = None) -> RunResult:
        """Run one request after this Agent has acquired exclusive ownership."""
        self._validate_tools()
        before_usage = usage_snapshot(self._model)
        if self._model is None:
            text = "No model configured: pass model= when creating the Agent."
            return RunResult(text=text, error=text)
        with self._control.enter_run():
            images = None
            if image is not None:
                from lamssi_agents.vision import to_image_urls

                try:
                    images = to_image_urls(image) or None
                except Exception as exc:
                    text = f"Could not encode image input: {exc}"
                    log.warning("run(image=...) could not encode the image: %s", exc)
                    self.emit(AgentEventType.USER_MESSAGE, message)
                    self.emit(AgentEventType.ERROR, text)
                    self.emit(AgentEventType.DONE, text)
                    return RunResult(
                        text=text,
                        outputs=self._control.outputs.snapshot(),
                        usage=usage_since(before_usage, self._model),
                        turns=0,
                        error=text,
                    )

            self._conversation.sanitize()
            self._conversation.begin_request()
            self._runtime.begin_request()

            self._conversation.append(
                Message(role="user", content=message, images=images)
            )
            self.emit(AgentEventType.USER_MESSAGE, message)
            errors: list[str] = []

            def capture_error(event: Any) -> None:
                if event.type is AgentEventType.ERROR:
                    errors.append(str(event.data or "Agent run failed."))

            unsubscribe = self.add_event_listener(capture_error)
            aborted = False
            run_loop = loop.RunLoop(
                conversation=self._conversation,
                runtime=self._runtime,
                control=self._control,
                model=lambda: self._model,
                config=self._config,
                emit=self.emit,
                assemble_prompt=self.assemble_prompt,
                before_turn=self._before_turn,
            )
            try:
                text = loop.run(run_loop)
            except AgentAborted:
                text = "Generation stopped."
                aborted = True
                self.emit(AgentEventType.ABORTED, text)
                self.emit(AgentEventType.DONE, text)
            finally:
                unsubscribe()

        return RunResult(
            text=text,
            outputs=self._control.outputs.snapshot(),
            usage=usage_since(before_usage, self._model),
            turns=self._conversation.turn,
            aborted=aborted,
            error=errors[-1] if errors else None,
        )

    def chat(self, message: str, *, image: Any = None) -> str:
        """Run one blocking request and return only its response text.

        Args:
            message: User text to append and process.
            image: Optional image input supported by the configured model.

        Returns:
            The :attr:`RunResult.text` value from :meth:`run`.
        """
        return self.run(message, image=image).text

    def clear_history(self) -> None:
        """Clear conversation messages and conversation-scoped feature state."""
        self._conversation.clear()
        self._runtime.on_cleared()
        logger = self._conv_logger
        if logger is not None:
            try:
                logger.roll_session()
            except Exception as exc:
                log.debug("conversation logger roll_session() raised: %s", exc)

    def compact(
        self,
        *,
        budget_tokens: int = 1,
        keep_recent: Optional[int] = None,
        focus: Optional[str] = None,
    ) -> "conversation.CompactionResult":
        """Compact the history now, however little of it there is.

        For a host that knows something the automatic estimate can't: a task just
        finished, or a long unattended run is starting.

        Args:
            budget_tokens: Compact toward this; 1 (default) runs every pass for
                the strongest shrink.
            keep_recent: Recent messages left untouched, overriding the config
                for this call: lowering it costs the next turn's context.
            focus: Optional instruction telling the summarizer what details to
                preserve. It is guidance, not a new user request.

        Returns:
            A :class:`~lamssi_agents.agent.conversation.CompactionResult`, falsy
            when nothing could be removed.
        """
        return self._conversation.force_compaction(
            budget_tokens=budget_tokens,
            keep_recent=keep_recent,
            focus=focus or "",
        )

    def emit(self, etype: AgentEventType, data: Any = None, **meta: Any) -> None:
        """Publish an advanced application or feature event for this run.

        Args:
            etype: Event type consumed by registered listeners.
            data: Primary event payload.
            **meta: Additional event metadata.
        """
        event = AgentEvent(type=etype, data=data, metadata=meta)
        for handler in list(self._feature_event_handlers):
            try:
                handler(event)
            except Exception as exc:
                log.debug(
                    "feature event handler %r raised: %s",
                    handler,
                    exc,
                    exc_info=True,
                )
        self._control.events.emit(event)

    def add_event_listener(self, cb: AgentEventCallback):
        """Subscribe to this agent's events.

        Args:
            cb: Callable receiving one :class:`AgentEvent` at a time.

        Returns:
            A zero-argument callable that removes this subscription.
        """
        return self._control.subscribe(cb)

    def abort(self) -> None:
        """Request cancellation of the current run."""
        self._control.abort()
        log.info("Abort requested")

    @property
    def is_aborted(self) -> bool:
        """Whether cancellation has been requested for the current run."""
        return self._control.is_aborted

    def check_abort(self) -> None:
        """Raise immediately when cancellation has been requested.

        Raises:
            AgentAborted: If :attr:`is_aborted` is true.
        """
        if self._control.is_aborted:
            raise AgentAborted("Agent run aborted by user")

    def set_conversation_log_dir(
        self,
        path: Union[str, Path, None],
        *,
        remote: Optional[Any] = None,
        debug: bool = False,
    ) -> None:
        """Attach or remove the append-only conversation log.

        Args:
            path: Directory receiving JSONL logs, or ``None`` to disable logging.
            remote: Optional remote sink consumed by the conversation logger.
            debug: Include full debug records when true.
        """
        from lamssi_agents.conversation_log import ConversationLogger

        existing = self._conv_logger
        if existing is not None:
            self._control.events.unsubscribe(existing)
            try:
                existing.close()
            except Exception as exc:
                log.debug("conversation logger close() raised: %s", exc)
            self._conv_logger = None
        if path is None:
            return

        logger = ConversationLogger(
            path,
            model=self.model_id,
            adapter=self.model_adapter_name,
            remote=remote,
            debug=debug,
        )
        self._conv_logger = logger
        self._control.events.subscribe(logger)

    @property
    def conversation_logger(self):
        """The active conversation logger, or ``None`` when logging is disabled."""
        return self._conv_logger

    @property
    def model(self) -> Optional[Model]:
        """The active model adapter, or ``None`` for an unconfigured agent."""
        return self._model

    @property
    def model_id(self) -> str:
        """The adapter's model identifier for prompts, logs, and diagnostics."""
        return model_id(self._model)

    def use_model(self, value: ModelInput) -> None:
        """Replace the model without clearing conversation history.

        Args:
            value: LiteLLM model identifier or custom model adapter, using the
                same seam as construction.
        """
        self._attach_model(resolve_model(value))

    def _attach_model(self, model: Model) -> None:
        """Attach a resolved model and refresh dependent runtime metadata."""
        self._model = model
        logger = self._conv_logger
        # Keep shared logger metadata tied to the top-level run.
        if logger is not None and not self._control.is_running:
            try:
                logger.set_metadata(model=model_id(model), adapter=adapter_name(model))
            except Exception as exc:
                log.debug("conversation logger set_metadata raised: %s", exc)
        log.info("Model: %s  adapter: %s", model_id(model), adapter_name(model))

    @property
    def summary_model(self) -> Optional[Model]:
        """Model used for compaction summaries; the main model by default."""
        return self._summary_model or self._model

    @summary_model.setter
    def summary_model(self, value: Optional[ModelInput]) -> None:
        """Set an explicit summary model, or ``None`` to follow the main model."""
        self._summary_model = resolve_model(value) if value is not None else None

    def check_model_connectivity(self) -> tuple:
        """Ask the active adapter whether its endpoint is reachable.

        Returns:
            ``(ok, detail)``. Custom adapters without a connectivity method are
            treated as reachable because no generic probe exists.
        """
        if self._model is None:
            return False, "No model configured"
        check = getattr(self._model, "check_connectivity", None)
        return check() if check is not None else (True, "custom model adapter")

    @property
    def model_loaded(self) -> bool:
        """Whether this agent currently has a model adapter."""
        return self._model is not None

    @property
    def model_adapter_name(self) -> str:
        """Short adapter name for diagnostics and status UIs."""
        return adapter_name(self._model)

    @property
    def usage(self):
        """A read-only snapshot of cumulative usage for the active model."""
        return usage_snapshot(self._model)

    @property
    def model_is_local(self) -> bool:
        """Whether the active adapter identifies its endpoint as local."""
        return self._model is not None and getattr(self._model, "is_local", False)

    @property
    def context_usage(self) -> "conversation.ContextUsage":
        """How full the context window is right now: ``print(agent.context_usage)``.

        The *window* size, not cumulative spend: a long run makes many requests
        each charged in full, so spend climbs while this number can sit still.
        """
        return self._conversation.context_usage()

    @property
    def max_turns(self) -> int:
        """Maximum model/tool turns allowed in one request."""
        return self._config.max_turns

    @max_turns.setter
    def max_turns(self, value: int) -> None:
        self._config = self._config.merged(max_turns=value).normalised()

    @property
    def interaction(self) -> Optional[InteractionHandler]:
        """Typed application interaction handler for this run."""
        return self._control.interaction.handler

    @interaction.setter
    def interaction(self, value: Optional[InteractionHandler]) -> None:
        self._control.interaction.handler = value

    @property
    def approval(self) -> ApprovalPolicy:
        """The application tool-approval policy for this agent."""
        return self._control.approval

    @approval.setter
    def approval(self, value: ApprovalPolicy) -> None:
        self._control.approval = value
        log.info("Approval policy: %s", value.name)

    def disable_tool(self, name: str) -> None:
        """Disable a tool by name for this shared run authority.

        Args:
            name: Tool to hide and reject until re-enabled.
        """
        self._control.disable_tool(name)

    def enable_tool(self, name: str) -> None:
        """Re-enable a tool disabled through :meth:`disable_tool`.

        Args:
            name: Tool name to remove from the disabled set.
        """
        self._control.enable_tool(name)

    def disabled_tool_names(self) -> set:
        """Return a snapshot of tool names disabled for this run authority."""
        return set(self._control.disabled_tool_names)

    def available_tool_names(self) -> List[str]:
        """Every tool this agent may call."""
        return sorted(self._runtime.surface().names)

    def visible_tool_defs(self) -> List[ToolDefinition]:
        """Tools whose full schema goes to the model this turn."""
        return self._runtime.all_defs()

    def all_tool_defs(self) -> List[ToolDefinition]:
        """Every agent-exposed tool the host offers, ignoring narrowing."""
        return self._runtime.all_defs_unfiltered()

    @property
    def history(self) -> List[Message]:
        """This conversation's messages, for a host that renders them."""
        return list(self._conversation.history)

    def conversation_state(self, key: type[T], factory: Any) -> T:
        """Get or create feature-owned state scoped to this conversation.

        Args:
            key: Type used to identify the state slot.
            factory: Zero-argument callable used only when the slot is absent.

        Returns:
            Existing or newly created state.
        """
        return self._conversation.state(key, factory)


__all__ = ["Agent"]
