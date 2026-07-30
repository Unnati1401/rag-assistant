---
id: path-parameters
category: routing
confidence: high
source: fastapi-docs
updated: 2026-07-25
---

# Path Parameters

Declare path parameters with curly-brace syntax in the route, e.g.
`@app.get("/items/{item_id}")`, with a matching function argument. Add a type
hint (e.g. `item_id: int`) for automatic parsing and validation; invalid values
return a clear validation error.

**Rules:**
- Declare fixed paths (e.g. `/users/me`) before variable paths (e.g. `/users/{user_id}`); routes match in order.
- Use `{param:path}` to allow a value containing slashes.
- Use a `str`-based `Enum` to restrict a parameter to predefined values.

**Related entries:** `query-parameters`, `request-body`