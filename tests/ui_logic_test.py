"""Executes dashboard/app.py top-to-bottom against FAKE streamlit/plotly objects.

Purpose: catch Python errors (bad column names, wrong data shapes, scoring failures) in the dashboard
logic when Streamlit itself is not installed. It does NOT test rendering - run the real app for that.
    python tests/ui_logic_test.py
"""
import runpy
import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CALLS = {"plotly_chart": 0, "dataframe": 0, "markdown": 0}


class Fake:
    def __enter__(self): return self
    def __exit__(self, *a): return False

    def __getattr__(self, name):
        def f(*a, **k):
            if name in CALLS: CALLS[name] += 1
            if name == "columns":
                n = a[0] if isinstance(a[0], int) else len(a[0])
                return [Fake() for _ in range(n)]
            if name == "tabs": return [Fake() for _ in a[0]]
            if name == "radio" or name == "selectbox": return a[1][k.get("index", 0)]
            if name == "multiselect": return k.get("default", [])
            if name in ("number_input", "slider"): return a[3] if len(a) > 3 else k["value"]
            if name == "checkbox": return a[1] if len(a) > 1 else k.get("value", False)
            if name in ("stop",): raise SystemExit("st.stop() called")
            if name in ("cache_resource", "cache_data"): return (lambda fn: fn)
            return Fake()
        return f


st = Fake()
st.session_state = {}
st.sidebar = Fake()
st.cache_resource = lambda fn: fn
st.cache_data = lambda fn: fn
sys.modules["streamlit"] = st
for m in ("plotly", "plotly.express", "plotly.graph_objects"):
    sys.modules[m] = MagicMock()

for currency in ("INR", "USD"):
    st.session_state.clear()
    orig = st.radio
    st.radio = lambda label, options, index=0, **k: ("USD ($)" if (currency == "USD" and "currency" in label) else options[index])
    runpy.run_path(str(ROOT / "dashboard" / "app.py"))
    st.radio = orig
    print(currency, "run OK", CALLS)
print("UI logic test passed")
