
import time
from typing import Callable, Optional, Dict, Tuple
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Try import PyDDSBB
try:
    import PyDDSBB
except Exception:
    PyDDSBB = None

# -------------------- 1D Functions (4 existing + 2 new) --------------------
def f_sin2pix(x):
    return np.sin(2 * np.pi * x)

def f_sin_combo(x):
    return np.sin(3 * x) + 0.5 * np.sin(7 * x)

def f_poly(x):
    return x**3 - 0.5 * x**2 + 0.2 * x

def f_gauss_bump(x):
    return np.exp(-((x - 0.2) ** 2) / 0.01) - 0.7 * np.exp(-((x - 0.75) ** 2) / 0.02)

# New 1D
def f_modulated_sine(x):
    return (1.4 - 3.0 * x) * np.sin(18.0 * x)

def f_sin_sum_2pt5(x):
    return np.sin(x) + np.sin((10.0 / 4.0) * x)

# -------------------- 2D Functions --------------------
def multiGauss(x):
    x0, x1 = float(x[0]), float(x[1])
    f = (-0.5*np.exp(-100*(x0**2 + x1**2))) \
        - 1.2*np.exp(-4*((-1 + x0)**2 + x1**2)) \
        - np.exp(-3*(x0**2 + (0.5 + x1)**2)) \
        - np.exp(-2*((0.5 + x0)**2 + x1**2)) \
        - 1.2*np.exp(-4*(x0**2 + (-1 + x1)**2))
    return float(f)

def himmelblau(x):
    return float((x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2)

def rosenbrock(x, a=1.0, b=100.0):
    return float((a - x[0])**2 + b*(x[1] - x[0]**2)**2)

# New 2D
def six_hump_camel(x):
    # (4 - 2.1*x^2 + x^4/3)*x^2 + x*y + (-4 + 4*y^2)*y^2
    return float((4 - 2.1*x[0]**2 + (x[0]**4)/3.0)*x[0]**2 + x[0]*x[1] + (-4 + 4*x[1]**2)*x[1]**2)

def sphere_2d(x):
    return float(x[0]**2 + x[1]**2)

def ackley_2d(x):
    x0, x1 = float(x[0]), float(x[1])
    return float(-20.0*np.exp(-0.2*np.sqrt(0.5*(x0**2 + x1**2))) - np.exp(0.5*(np.cos(2*np.pi*x0) + np.cos(2*np.pi*x1))) + np.e + 20.0)\n
# -------------------- Functions defined below --------------------

def multiGauss(x):
    x0, x1 = float(x[0]), float(x[1])
    f = (-0.5*np.exp(-100*(x0**2 + x1**2))) \
        - 1.2*np.exp(-4*((-1 + x0)**2 + x1**2)) \
        - np.exp(-3*(x0**2 + (0.5 + x1)**2)) \
        - np.exp(-2*((0.5 + x0)**2 + x1**2)) \
        - 1.2*np.exp(-4*(x0**2 + (-1 + x1)**2))
    return float(f)

def himmelblau(x):
    # Minima near (3,2), (-2.805,3.131), (-3.779,-3.283), (3.584,-1.848)
    a = (x[0]**2 + x[1] - 11)**2
    b = (x[0] + x[1]**2 - 7)**2
    return float(a + b)

def rosenbrock(x, a=1.0, b=100.0):
    # Global min at (a, a^2) = (1,1) for a=1
    return float((a - x[0])**2 + b*(x[1] - x[0]**2)**2)

# Registry of objectives: name -> (dimension, callable, default_bounds_per_dim)
# Registry: name -> (dimension, callable, default_bounds, latex_equation)
OBJECTIVES: Dict[str, Tuple[int, Callable, Tuple[Tuple[float, float], ...], str]] = {
    # 1D (4 existing + 2 new)
    "Sine (sin(2πx))": (1, f_sin2pix, ((-1.0, 1.0),), r"\sin(2\pi x)"),
    "Sine (sin(3x) + 0.5 sin(7x))": (1, f_sin_combo, ((-1.0, 1.0),), r"\sin(3x) + 0.5\sin(7x)"),
    "Polynomial (x^3 - 0.5x^2 + 0.2x)": (1, f_poly, ((-1.0, 1.0),), r"x^3 - 0.5x^2 + 0.2x"),
    "Gaussian bump": (1, f_gauss_bump, ((-1.0, 1.0),), r"e^{-\frac{(x-0.2)^2}{0.01}} - 0.7\, e^{-\frac{(x-0.75)^2}{0.02}}"),
    "(1.4 - 3x)\, \sin(18x)": (1, f_modulated_sine, ((-1.0, 1.0),), r"(1.4 - 3x)\sin(18x)"),
    "\sin(x) + \sin(2.5x)": (1, f_sin_sum_2pt5, ((-2.0, 2.0),), r"\sin(x) + \sin(2.5x)"),

    # 2D
    "Multi-Gaussian (2D)": (2, multiGauss, ((-1.0, 1.0), (-1.0, 1.0)), r"-0.5 e^{-100(x^2+y^2)} - 1.2 e^{-4((-1+x)^2+y^2)} - e^{-3(x^2+(0.5+y)^2)} - e^{-2((0.5+x)^2+y^2)} - 1.2 e^{-4(x^2+(-1+y)^2)}"),
    "Himmelblau (2D)": (2, himmelblau, ((-5.0, 5.0), (-5.0, 5.0)), r"(x^2 + y - 11)^2 + (x + y^2 - 7)^2"),
    "Rosenbrock (2D)": (2, rosenbrock, ((-2.0, 2.0), (-1.0, 3.0)), r"(1-x)^2 + 100(y-x^2)^2"),
    "Six-Hump Camel (2D)": (2, six_hump_camel, ((-2.0, 2.0), (-1.2, 1.2)), r"(4 - 2.1x^2 + \tfrac{x^4}{3})x^2 + xy + (-4 + 4y^2)y^2"),
    "Sphere (2D)": (2, sphere_2d, ((-3.0, 3.0), (-3.0, 3.0)), r"x^2 + y^2"),
    "Ackley (2D)": (2, ackley_2d, ((-3.0, 3.0), (-3.0, 3.0)), r"-20 e^{-0.2 \sqrt{0.5(x^2+y^2)}} - e^{0.5(\cos 2\pi x + \cos 2\pi y)} + e + 20"),
}

# -------------------- Helpers --------------------
def grid_eval_2d(f2d: Callable, x1_bounds: Tuple[float, float], x2_bounds: Tuple[float, float], n: int = 120):
    x = np.linspace(x1_bounds[0], x1_bounds[1], n)
    y = np.linspace(x2_bounds[0], x2_bounds[1], n)
    X, Y = np.meshgrid(x, y)
    Z = np.empty_like(X, dtype=float)
    for i in range(n):
        for j in range(n):
            Z[i, j] = float(f2d([X[i, j], Y[i, j]]))
    return X, Y, Z

def make_plot_1d(xs_true, ys_true, xopt: Optional[float], yopt: Optional[float], title="Objective"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs_true, y=ys_true, mode="lines", name="f(x)"))
    if xopt is not None and yopt is not None:
        fig.add_trace(go.Scatter(x=[xopt], y=[yopt], mode="markers", name="PyDDSBB optimum", marker=dict(size=10)))
    fig.update_layout(title=title, xaxis_title="x", yaxis_title="f(x)", hovermode="x", height=520, margin=dict(l=10, r=10, t=40, b=10))
    return fig

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="PyDDSBB 1D & 2D Global Optimization", layout="wide")
st.title("🧭 PyDDSBB: 1D & 2D Global Optimization (dynamic sampling)")

if PyDDSBB is None:
    st.error("PyDDSBB not available. On Streamlit Cloud, include GLPK in packages.txt and install PyDDSBB from GitHub in requirements.txt.")
    st.stop()

with st.sidebar:
    st.header("Objective")
    choice = st.selectbox("Choose objective", list(OBJECTIVES.keys()), index=0)
    dim, f_obj, default_bounds, eq_latex = OBJECTIVES[choice]

    if dim == 1:
        (x_min, x_max) = st.slider("Domain [x_min, x_max]", -6.0, 6.0, default_bounds[0], step=0.1)
    else:
        (x1_min, x1_max) = st.slider("x₁ domain", -6.0, 6.0, default_bounds[0], step=0.1)
        (x2_min, x2_max) = st.slider("x₂ domain", -6.0, 6.0, default_bounds[1], step=0.1)

    seed = st.number_input("Random seed", 0, 999999, 42)

    # Layout option for visualizations
    layout_mode = "Vertical"
    if dim == 2:
        layout_mode = st.radio("2D layout", ["2×2 grid", "Tabs", "Vertical"], index=0,
                               help="Arrange the four figures: Surface, Contour, Bounds, Search.")
    else:
        layout_mode = st.radio("1D layout", ["2×2 grid", "Tabs", "Vertical"], index=0,
                               help="Arrange: Objective, Bounds, Search.")
    st.header("Solver (PyDDSBB)")
    n_init = st.number_input("Initial samples (n_init)", min_value=3, max_value=200, value=12, step=1)
    split_method = st.selectbox("Split method", ["equal_bisection", "golden_section"])
    variable_selection = st.selectbox("Variable selection", ["longest_side", "svr_var_select"])
    multifidelity = st.checkbox("Multifidelity", value=False)
    sense = st.selectbox("Sense", ["minimize", "maximize"], index=0)

    st.subheader("Stopping criteria")
    abs_tol = st.number_input("absolute_tolerance", value=1e-3, format="%.6f")
    rel_tol = st.number_input("relative_tolerance", value=1e-3, format="%.6f")
    min_bound = st.number_input("minimum_bound", value=0.01, format="%.4f")
    sampling_limit = st.number_input("sampling_limit", min_value=10, max_value=20000, value=800, step=10)
    time_limit = st.number_input("time_limit (s)", min_value=1.0, max_value=36000.0, value=20.0, step=1.0, format="%.1f")

    st.divider()
    auto_run = st.checkbox("Auto-run on change", value=False)
    run_btn = st.button("🚀 Run optimization")
    resume_btn = st.button("↻ Resume with more budget")


# Build objective and model

def objective(x_arr):
    if dim == 1:
        xval = float(x_arr[0])
        v = f_obj(np.array([xval]))
        return float(v[0] if isinstance(v, np.ndarray) else v)
    else:
        return float(f_obj(x_arr))

model = PyDDSBB.DDSBBModel.Problem()
model.add_objective(objective, sense=sense)
if dim == 1:
    model.add_variable(float(x_min), float(x_max))
else:
    model.add_variable(float(x1_min), float(x1_max))
    model.add_variable(float(x2_min), float(x2_max))

stop_option = {
    "absolute_tolerance": float(abs_tol),
    "relative_tolerance": float(rel_tol),
    "minimum_bound": float(min_bound),
    "sampling_limit": int(sampling_limit),
    "time_limit": float(time_limit),
}

# Cache solver across runs for resume
if "solver" not in st.session_state:
    st.session_state.solver = None
if "last_cfg" not in st.session_state:
    st.session_state.last_cfg = None
if "result" not in st.session_state:
    st.session_state.result = None

cfg = (
    choice, dim,
    (x_min, x_max) if dim == 1 else (x1_min, x1_max, x2_min, x2_max),
    int(n_init), split_method, variable_selection, bool(multifidelity), sense,
    float(abs_tol), float(rel_tol), float(min_bound), int(sampling_limit), float(time_limit),
    int(seed)
)

def new_solver():
    return PyDDSBB.DDSBB(
        int(n_init),
        split_method=split_method,
        variable_selection=variable_selection,
        multifidelity=multifidelity,
        stop_option=stop_option,
        sense=sense
    )

should_run = False
if run_btn:
    st.session_state.solver = new_solver()
    should_run = True
elif auto_run and cfg != st.session_state.last_cfg:
    st.session_state.solver = new_solver()
    should_run = True
elif resume_btn and st.session_state.solver is not None:
    extra_sampling = int(max(100, 0.5 * sampling_limit))
    new_stop = dict(stop_option)
    new_stop["sampling_limit"] = int(stop_option["sampling_limit"]) + extra_sampling
    st.session_state.solver.resume(new_stop)

if should_run and st.session_state.solver is not None:
    st.session_state.last_cfg = cfg
    with st.spinner("Optimizing with PyDDSBB..."):
        start = time.time()
        st.session_state.solver.optimize(model)
        elapsed = time.time() - start
        yopt = float(st.session_state.solver.get_optimum())
        xopt_arr = st.session_state.solver.get_optimizer()
        # Robustly coerce to 1D float array, handling shapes like (2,), (1,2), (2,1), lists, etc.
        import numpy as _np
        _xv = _np.array(xopt_arr, dtype=float).reshape(-1)
        if dim == 1:
            xopt = float(_xv[0])
        else:
            xopt = (float(_xv[0]), float(_xv[1]))
        st.session_state.result = {"xopt": xopt, "yopt": yopt, "elapsed": elapsed}

# ---- Visualization
col_main, col_side = st.columns([4, 1.4])

with col_main:
    if dim == 1:

        # 1D line plot (true vs observed) + optimum
        xs_true = np.linspace(x_min, x_max, 600)
        ys_true = f_obj(xs_true)
        xs_obs, ys_obs = xs_true, ys_true.copy()
        
        xopt = st.session_state.result["xopt"] if st.session_state.result else None
        yopt = st.session_state.result["yopt"] if st.session_state.result else None
        fig_obj = make_plot_1d(xs_true, ys_true, xopt, yopt, title=f"{choice}")

        # --- Figure B: Upper/Lower Bound Evolution (Plotly) ---
        fig_bounds = None
        if st.session_state.solver is not None and hasattr(st.session_state.solver, "_lowerbound_hist") and hasattr(st.session_state.solver, "_upperbound_hist"):
            lb = list(st.session_state.solver._lowerbound_hist)
            ub = list(st.session_state.solver._upperbound_hist)
            xlv = list(range(len(lb)))
            fig_bounds = go.Figure()
            fig_bounds.add_trace(go.Scatter(x=xlv, y=ub, mode="lines+markers", name="Upper Bound"))
            fig_bounds.add_trace(go.Scatter(x=xlv, y=lb, mode="lines+markers", name="Lower Bound"))
            fig_bounds.update_layout(title="Lower & Upper Bound Evolution", xaxis_title="Level", yaxis_title="f(x)",
                                     height=420, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))

        # --- Figure C: Search Space Branching & Sampling (1D) ---
        fig_search = None
        if st.session_state.solver is not None and hasattr(st.session_state.solver, "Tree"):
            tree = st.session_state.solver.Tree
            shapes = []
            scat_x, scat_y = [], []
            levels = sorted(list(tree.keys()))
            n_levels = max(levels) + 1 if len(levels) else 1
            for level in levels:
                shade = level / max(1, n_levels)
                fill = f"rgba(128,128,128,{0.15 + 0.5*shade})"
                for node in tree[level].values():
                    x0, x1 = node.bounds[0, 0], node.bounds[1, 0]
                    y0, y1 = level, level + 0.8
                    shapes.append(dict(
                        type="rect", xref="x", yref="y",
                        x0=x0, x1=x1, y0=y0, y1=y1,
                        line=dict(color="white", width=0.5),
                        fillcolor=fill,
                    ))
                    if hasattr(node, "x") and node.x is not None and len(node.x) > 0:
                        scat_x.extend(node.x[:, 0].tolist())
                        scat_y.extend([level + 0.4] * node.x.shape[0])

            fig_search = go.Figure()
            if scat_x:
                fig_search.add_trace(go.Scatter(x=scat_x, y=scat_y, mode="markers",
                                                name="Samples", marker=dict(size=6, line=dict(width=0.5))))
            fig_search.update_layout(title="Search Space Branching & Sampling (1D)",
                                     xaxis_title="x", yaxis_title="Level",
                                     height=420, margin=dict(l=10, r=10, t=40, b=10),
                                     shapes=shapes, yaxis=dict(autorange="reversed"))

        # --- Layout for 1D ---
        if layout_mode == "2×2 grid":
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(fig_obj, use_container_width=True)
            with c2:
                if fig_bounds is not None:
                    st.plotly_chart(fig_bounds, use_container_width=True)
            c3, c4 = st.columns(2)
            with c3:
                if fig_search is not None:
                    st.plotly_chart(fig_search, use_container_width=True)
            with c4:
                st.empty()  # placeholder
        elif layout_mode == "Tabs":
            tabs = st.tabs(["Objective", "Bounds", "Search"])
            with tabs[0]:
                st.plotly_chart(fig_obj, use_container_width=True)
            with tabs[1]:
                if fig_bounds is not None:
                    st.plotly_chart(fig_bounds, use_container_width=True)
            with tabs[2]:
                if fig_search is not None:
                    st.plotly_chart(fig_search, use_container_width=True)
        else:  # Vertical
            st.plotly_chart(fig_obj, use_container_width=True)
            if fig_bounds is not None:
                st.plotly_chart(fig_bounds, use_container_width=True)
            if fig_search is not None:
                st.plotly_chart(fig_search, use_container_width=True)

    else:

        # 2D Surface plot (Plotly)
        xopt = st.session_state.result["xopt"] if st.session_state.result else None
        yopt = st.session_state.result["yopt"] if st.session_state.result else None

        X, Y, Z = grid_eval_2d(f_obj, (x1_min, x1_max), (x2_min, x2_max), n=140)
        Z_plot = Z.copy()

        # Prepare all four figures
        surf = go.Figure(data=[go.Surface(z=Z_plot, x=X, y=Y, opacity=0.95, showscale=True, name="f(x1,x2)")])
        if xopt is not None and yopt is not None:
            surf.add_trace(go.Scatter3d(x=[xopt[0]], y=[xopt[1]], z=[yopt],
                                        mode="markers", marker=dict(size=5, symbol="diamond"),
                                        name="PyDDSBB optimum"))
        surf.update_layout(title="Objective Surface with Optimum", scene=dict(
            xaxis_title="x₁", yaxis_title="x₂", zaxis_title="f(x)"
        ), height=520, margin=dict(l=10, r=10, t=40, b=10))

        contour = go.Figure(data=[go.Contour(z=Z_plot, x=X[0], y=Y[:,0], contours=dict(showlabels=True))])
        if xopt is not None:
            contour.add_trace(go.Scatter(x=[xopt[0]], y=[xopt[1]], mode="markers",
                                         marker=dict(size=10, color="red", line=dict(width=1)),
                                         name="PyDDSBB optimum"))
        contour.update_layout(title="Contour Plot with Optimum", xaxis_title="x₁", yaxis_title="x₂",
                              height=520, margin=dict(l=10, r=10, t=40, b=10))

        fig_bounds = None
        if hasattr(st.session_state.solver, "_lowerbound_hist") and hasattr(st.session_state.solver, "_upperbound_hist"):
            lb = list(st.session_state.solver._lowerbound_hist)
            ub = list(st.session_state.solver._upperbound_hist)
            xlv = list(range(len(lb)))
            fig_bounds = go.Figure()
            fig_bounds.add_trace(go.Scatter(x=xlv, y=ub, mode="lines+markers", name="Upper Bound"))
            fig_bounds.add_trace(go.Scatter(x=xlv, y=lb, mode="lines+markers", name="Lower Bound"))
            fig_bounds.update_layout(title="Lower & Upper Bound Evolution", xaxis_title="Level", yaxis_title="f(x)",
                                     height=520, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))

        fig_search = None
        if hasattr(st.session_state.solver, "Tree"):
            tree = st.session_state.solver.Tree
            shapes = []
            scat_x, scat_y = [], []
            levels = sorted(list(tree.keys()))
            n_levels = max(levels) + 1 if len(levels) else 1
            for level in levels:
                shade = level / max(1, n_levels)
                fill = f"rgba(128,128,128,{0.15 + 0.5*shade})"
                for node in tree[level].values():
                    x0, x1 = node.bounds[0, 0], node.bounds[1, 0]
                    y0, y1 = node.bounds[0, 1], node.bounds[1, 1]
                    shapes.append(dict(
                        type="rect", xref="x", yref="y",
                        x0=x0, x1=x1, y0=y0, y1=y1,
                        line=dict(color="white", width=0.5),
                        fillcolor=fill,
                    ))
                    if hasattr(node, "x") and node.x is not None and len(node.x) > 0:
                        scat_x.extend(node.x[:, 0].tolist())
                        scat_y.extend(node.x[:, 1].tolist())

            fig_search = go.Figure()
            if scat_x:
                fig_search.add_trace(go.Scatter(x=scat_x, y=scat_y, mode="markers",
                                                name="Samples", marker=dict(size=6, line=dict(width=0.5))))
            fig_search.update_layout(title="Search Space Branching & Sampling",
                                     xaxis_title="x₁", yaxis_title="x₂", height=520,
                                     margin=dict(l=10, r=10, t=40, b=10), shapes=shapes)

        # Layout selection
        if layout_mode == "2×2 grid":
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(surf, use_container_width=True)
            with c2:
                st.plotly_chart(contour, use_container_width=True)
            c3, c4 = st.columns(2)
            with c3:
                if fig_bounds is not None:
                    st.plotly_chart(fig_bounds, use_container_width=True)
            with c4:
                if fig_search is not None:
                    st.plotly_chart(fig_search, use_container_width=True)

        elif layout_mode == "Tabs":
            tabs = st.tabs(["Surface", "Contour", "Bounds", "Search"])
            with tabs[0]:
                st.plotly_chart(surf, use_container_width=True)
            with tabs[1]:
                st.plotly_chart(contour, use_container_width=True)
            with tabs[2]:
                if fig_bounds is not None:
                    st.plotly_chart(fig_bounds, use_container_width=True)
            with tabs[3]:
                if fig_search is not None:
                    st.plotly_chart(fig_search, use_container_width=True)

        else:  # Vertical
            st.plotly_chart(surf, use_container_width=True)
            st.plotly_chart(contour, use_container_width=True)
            if fig_bounds is not None:
                st.plotly_chart(fig_bounds, use_container_width=True)
            if fig_search is not None:
                st.plotly_chart(fig_search, use_container_width=True)

with col_side:
    st.subheader("Result")
    if st.session_state.result is None:
        st.info("Set options and click **Run optimization**.")
    else:
        xopt = st.session_state.result["xopt"]
        yopt = st.session_state.result["yopt"]
        if isinstance(xopt, tuple):
            st.metric("x*", f"({xopt[0]:.6g}, {xopt[1]:.6g})")
        else:
            st.metric("x*", f"{xopt:.6g}")
        st.metric("f(x*)", f"{yopt:.6g}")
        st.caption(f"Elapsed: {st.session_state.result['elapsed']:.2f} s")
        st.caption(f"n_init: {n_init} · split: {split_method} · var sel: {variable_selection}")
        
st.caption("PyDDSBB solves a black-box global optimization by sampling, building underestimators, and branching the domain. This demo supports 1D and 2D objectives.")
