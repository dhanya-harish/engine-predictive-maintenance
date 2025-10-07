
## `src/utils.py`
```python
import re
import pandas as pd

def find_target_column(df: pd.DataFrame, want="engine_condition") -> str:
    cols = [c.strip() for c in df.columns]
    norm = lambda s: re.sub(r"[\s_-]+", "", s.lower())

    want_norm = norm(want)
    for c in cols:
        if norm(c) == want_norm:
            return c

    fallbacks = [
        "engine condition", "condition", "target", "label",
        "failure", "engine_status", "engine health", "enginehealth", "status"
    ]
    for fb in fallbacks:
        fb_norm = norm(fb)
        for c in cols:
            if norm(c) == fb_norm:
                return c
    raise KeyError(f"Could not find a target column like '{want}'. Columns: {list(df.columns)}")
