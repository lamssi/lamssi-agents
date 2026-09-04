# Security policy

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use the repository's
private vulnerability-reporting channel; if it is unavailable, contact the repository
owner privately. Include the affected version, a minimal reproduction, and the expected
impact.

## Scope

The current supported line is `0.1.x`. Tool approval records user consent. Applications
controlling hardware or sensitive services must also enforce their domain safety and
authorization checks in the tool execution path.

Lamssi removes recognized credentials from model-requested shell environments and
redacts registered secrets from logs and model-visible tool results. Applications remain
responsible for registering nonstandard secrets and for deciding which environment
variables, tools, files, and external registries an Agent may access.
