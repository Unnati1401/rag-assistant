---
id: response-model
category: response-handling
confidence: high
source: fastapi-docs
updated: 2026-07-25
---

# Response Model

Declare the response schema with the `response_model` parameter or a return type
annotation. FastAPI validates, serializes, and filters output to the declared
schema and documents it in OpenAPI. If both are given, `response_model` wins.

**Rules:**
- Fields not in the model are filtered out (e.g. hide a password via an output model).
- `response_model=None` disables response modeling.
- `response_model_exclude_unset=True` omits fields left at defaults.

**Related entries:** `request-body`