```markdown
# daisy-bridge Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches you the core development patterns, coding conventions, and workflows used in the `daisy-bridge` repository. The project is a Python codebase built on the Flask framework, focusing on modular bridge, loss, and serving components. You'll learn how to implement new features, update serving endpoints and UI, follow repository conventions, and write and organize tests.

## Coding Conventions

### File Naming
- Use **snake_case** for all Python files and modules.
  - Example: `bridge_factory.py`, `test_serving.py`

### Import Style
- Use **relative imports** within modules.
  - Example:
    ```python
    from .base_bridge import BaseBridge
    from ..losses.cross_entropy import CrossEntropyLoss
    ```

### Export Style
- Use **named exports** (explicitly define what is exported).
  - Example:
    ```python
    __all__ = ["BridgeFactory", "BaseBridge"]
    ```

### Commit Messages
- Use **conventional commits** with the `feat` prefix for new features.
  - Example:  
    ```
    feat: add new bridge module for RAG integration
    ```

## Workflows

### Feature Module Implementation
**Trigger:** When you want to add a new major component or capability (e.g., bridges, losses, sidecar, training, serving) to the system.  
**Command:** `/new-module`

1. **Create or update source files** under the relevant submodule directory:
    - Example: `src/bridge_rag/bridges/my_new_bridge.py`
2. **Update or add integration points** such as factories, orchestrators, or pipelines:
    - Example: Update `src/bridge_rag/pipeline/orchestrator.py` to register the new bridge.
3. **Add or update corresponding test files** in the `tests/` directory:
    - Example: `tests/test_bridges.py`
4. **Fix or update related defects** surfaced by the new tests.

**Example:**
```python
# src/bridge_rag/bridges/my_new_bridge.py
from .base_bridge import BaseBridge

class MyNewBridge(BaseBridge):
    def forward(self, input):
        # implement bridge logic
        pass
```
```python
# tests/test_bridges.py
import unittest
from src.bridge_rag.bridges.my_new_bridge import MyNewBridge

class TestMyNewBridge(unittest.TestCase):
    def test_forward(self):
        bridge = MyNewBridge()
        assert bridge.forward("test") is not None
```

### Serving Endpoint and UI Update
**Trigger:** When you want to expose new functionality via the API or update the user-facing console/UI.  
**Command:** `/new-endpoint`

1. **Update or add a Flask app or endpoint** in `src/bridge_rag/serving/app.py`.
2. **Add or update static assets** in `src/bridge_rag/serving/static/`.
3. **Update backend logic** to emit new events or data, e.g., in `pipeline/lifecycle.py` or `pipeline/orchestrator.py`.
4. **Update or add documentation** in `README.md`.
5. **Optionally, add or update test files** for serving endpoints.

**Example:**
```python
# src/bridge_rag/serving/app.py
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/new-feature', methods=['GET'])
def new_feature():
    return jsonify({"status": "ok"})
```
```html
<!-- src/bridge_rag/serving/static/new_feature.html -->
<div>
  <h2>New Feature UI</h2>
</div>
```

## Testing Patterns

- **Test files** are named with the pattern `test_<module>.py` and placed in the `tests/` directory.
  - Example: `tests/test_serving.py`, `tests/test_bridges.py`
- **Testing framework** is not explicitly detected, but Python's `unittest` or `pytest` are commonly used.
- **Test structure**:
    ```python
    import unittest
    from src.bridge_rag.serving.app import app

    class TestServing(unittest.TestCase):
        def test_endpoint(self):
            client = app.test_client()
            response = client.get('/api/new-feature')
            assert response.status_code == 200
    ```

## Commands

| Command        | Purpose                                                        |
|----------------|----------------------------------------------------------------|
| /new-module    | Start the workflow for implementing a new feature/module       |
| /new-endpoint  | Start the workflow for adding or updating a serving endpoint   |
```