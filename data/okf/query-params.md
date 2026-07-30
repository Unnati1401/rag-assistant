---
id: query-parameters
category: routing
confidence: high
source: fastapi-docs
updated: 2026-07-25
---

# Query Parameters

Function parameters not part of the path are interpreted as query parameters
(the key-value pairs after `?` in the URL). Declaring a Python type converts and
validates the value.

**Rules:**
- A default value makes a query parameter optional; no default makes it required.
- Use `= None` to make it optional.
- `bool` parameters treat `1`, `true`, `on`, `yes` (any case) as True.

**Related entries:** `path-parameters`, `request-body`