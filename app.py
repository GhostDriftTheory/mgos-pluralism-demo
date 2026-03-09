import math
from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="MG-OS Demo",
    page_icon="🌀",
    layout="wide",
)


# ============================================================
# Utilities
# ============================================================
def softmax(z: np.ndarray) -> np.ndarray:
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((len(y), num_classes), dtype=float)
    out[np.arange(len(y)), y] = 1.0
    return out


def floor3(x: float) -> float:
    return math.floor(x * 1000) / 1000


def ceil3(x: float) -> float:
    return math.ceil(x * 1000) / 1000


def sigmoid_like_floor(margin: float, num_classes: int) -> float:
    # phi_C(m) = 1 / (1 + (C-1)e^{-m})
    return 1.0 / (1.0 + (num_classes - 1) * math.exp(-margin))


def train_test_split_stratified(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        idx = idx.copy()
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_ratio)))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    train_idx = np.array(train_idx, dtype=int)
    test_idx = np.array(test_idx, dtype=int)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ============================================================
# Demo A: Long-tail classification with minority barrier
# ============================================================
@dataclass
class LinearSoftmaxModel:
    W: np.ndarray
    b: np.ndarray

    def logits(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W + self.b

    def probs(self, X: np.ndarray) -> np.ndarray:
        return softmax(self.logits(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.probs(X), axis=1)


def generate_long_tail_data(
    counts: Tuple[int, int, int],
    seed: int,
    spread: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    rng = np.random.default_rng(seed)
    means = {
        0: np.array([-2.2, -0.6]),
        1: np.array([2.1, 0.3]),
        2: np.array([0.3, 2.5]),
    }
    covs = {
        0: np.array([[spread, 0.20], [0.20, spread + 0.15]]),
        1: np.array([[spread + 0.10, -0.18], [-0.18, spread]]),
        2: np.array([[spread * 0.80, 0.0], [0.0, spread * 0.80]]),
    }

    xs = []
    ys = []
    for cls, n in enumerate(counts):
        pts = rng.multivariate_normal(mean=means[cls], cov=covs[cls], size=n)
        xs.append(pts)
        ys.append(np.full(n, cls, dtype=int))
    X = np.vstack(xs)
    y = np.concatenate(ys)

    perm = rng.permutation(len(y))
    return X[perm], y[perm], means


def minority_margin_stats(
    logits: np.ndarray,
    y: np.ndarray,
    minority_class: int,
    margin_threshold: float,
) -> Dict[str, float]:
    true_logits = logits[np.arange(len(y)), y]
    masked = logits.copy()
    masked[np.arange(len(y)), y] = -np.inf
    second_best = np.max(masked, axis=1)
    margins = true_logits - second_best

    minority_mask = y == minority_class
    minority_margins = margins[minority_mask]

    if minority_margins.size == 0:
        min_margin = 0.0
    else:
        min_margin = float(np.min(minority_margins))

    ocw_trigger = float(np.mean(margins < margin_threshold))

    return {
        "min_margin": min_margin,
        "ocw_trigger": ocw_trigger,
        "all_margins": margins,
        "minority_margins": minority_margins,
    }


def train_softmax_model(
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    *,
    lr: float,
    epochs: int,
    l2: float,
    barrier_strength: float,
    margin_threshold: float,
    minority_class: int,
    seed: int,
) -> LinearSoftmaxModel:
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W = rng.normal(0.0, 0.05, size=(d, num_classes))
    b = np.zeros(num_classes, dtype=float)
    Y = one_hot(y, num_classes)

    minority_idx = np.where(y == minority_class)[0]

    for _ in range(epochs):
        logits = X @ W + b
        probs = softmax(logits)

        # Cross-entropy gradient
        grad_logits = (probs - Y) / n

        # Barrier term on minority samples only:
        # hinge(max(0, m0 - margin)), margin = z_y - max_{j != y} z_j
        if barrier_strength > 0 and minority_idx.size > 0:
            z_m = logits[minority_idx]
            y_m = y[minority_idx]
            true_z = z_m[np.arange(len(y_m)), y_m]
            masked = z_m.copy()
            masked[np.arange(len(y_m)), y_m] = -np.inf
            j_star = np.argmax(masked, axis=1)
            second = z_m[np.arange(len(y_m)), j_star]
            margins = true_z - second
            active = margins < margin_threshold

            if np.any(active):
                scale = barrier_strength / max(1, minority_idx.size)
                active_rows = minority_idx[active]
                active_y = y_m[active]
                active_j = j_star[active]
                grad_logits[active_rows, active_y] += -scale
                grad_logits[active_rows, active_j] += scale

        grad_W = X.T @ grad_logits + l2 * W
        grad_b = np.sum(grad_logits, axis=0)

        W -= lr * grad_W
        b -= lr * grad_b

    return LinearSoftmaxModel(W=W, b=b)


def classification_metrics(
    model: LinearSoftmaxModel,
    X: np.ndarray,
    y: np.ndarray,
    minority_class: int,
    margin_threshold: float,
) -> Dict[str, float]:
    logits = model.logits(X)
    preds = np.argmax(logits, axis=1)

    acc = float(np.mean(preds == y))

    minority_mask = y == minority_class
    if np.any(minority_mask):
        rec_min = float(np.mean(preds[minority_mask] == minority_class))
    else:
        rec_min = 0.0

    margin_info = minority_margin_stats(
        logits=logits,
        y=y,
        minority_class=minority_class,
        margin_threshold=margin_threshold,
    )
    eta_star = sigmoid_like_floor(margin_info["min_margin"], num_classes=len(np.unique(y)))

    return {
        "acc": acc,
        "rec_min": rec_min,
        "ocw_trigger": margin_info["ocw_trigger"],
        "eta_star": eta_star,
        "logits": logits,
        "preds": preds,
        "margins": margin_info["all_margins"],
        "minority_margins": margin_info["minority_margins"],
        "min_margin": margin_info["min_margin"],
    }


def make_decision_grid(
    model: LinearSoftmaxModel,
    X: np.ndarray,
    points: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, points),
        np.linspace(y_min, y_max, points),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).reshape(xx.shape)
    return xx, yy, zz


def plot_classification_panel(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    baseline_model: LinearSoftmaxModel,
    barrier_model: LinearSoftmaxModel,
    minority_class: int,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cmap_bg = plt.get_cmap("Pastel1")
    cmap_pts = plt.get_cmap("tab10")

    for ax, model, title in zip(
        axes,
        [baseline_model, barrier_model],
        ["Baseline softmax", "Barrier classifier (MG-OS style)"],
    ):
        xx, yy, zz = make_decision_grid(model, np.vstack([X_train, X_test]))
        ax.contourf(xx, yy, zz, alpha=0.35, cmap=cmap_bg)

        for cls in np.unique(y_train):
            train_mask = y_train == cls
            test_mask = y_test == cls
            ax.scatter(
                X_train[train_mask, 0],
                X_train[train_mask, 1],
                s=18,
                alpha=0.65,
                label=f"train class {cls}",
                color=cmap_pts(cls),
            )
            ax.scatter(
                X_test[test_mask, 0],
                X_test[test_mask, 1],
                s=36,
                marker="x",
                alpha=0.9,
                label=f"test class {cls}",
                color=cmap_pts(cls),
            )

        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=3, frameon=False)
    fig.suptitle(f"Minority class = {minority_class}", y=1.02)
    fig.tight_layout()
    return fig


def plot_margin_histograms(
    baseline_metrics: Dict[str, float],
    barrier_metrics: Dict[str, float],
    margin_threshold: float,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(baseline_metrics["minority_margins"], bins=25, alpha=0.8)
    axes[0].axvline(margin_threshold, linestyle="--")
    axes[0].set_title("Baseline minority margins")
    axes[0].set_xlabel("margin")
    axes[0].set_ylabel("count")
    axes[0].grid(alpha=0.2)

    axes[1].hist(barrier_metrics["minority_margins"], bins=25, alpha=0.8)
    axes[1].axvline(margin_threshold, linestyle="--")
    axes[1].set_title("Barrier minority margins")
    axes[1].set_xlabel("margin")
    axes[1].set_ylabel("count")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    return fig


# ============================================================
# Demo B: Double-well Langevin
# ============================================================
def simulate_langevin(
    gamma: float,
    temperature: float,
    dt: float,
    horizon: int,
    runs: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    sqrt_term = math.sqrt(2.0 * temperature * dt)
    trajectories = np.zeros((runs, horizon + 1), dtype=float)
    trajectories[:, 0] = -1.0

    crossings = np.zeros(runs, dtype=int)
    ocw_hits = np.zeros(runs, dtype=int)

    for r in range(runs):
        x = -1.0
        signs = [np.sign(x)]
        for t in range(1, horizon + 1):
            grad = gamma * x * (x * x - 1.0)
            noise = sqrt_term * rng.normal()
            x = x - grad * dt + noise
            trajectories[r, t] = x

            if abs(x) < 0.2:
                ocw_hits[r] += 1

            current_sign = np.sign(x) if x != 0 else 0.0
            prev_sign = signs[-1]
            if prev_sign < 0 and current_sign > 0:
                crossings[r] += 1
            signs.append(current_sign)

    kesc_per_run = crossings / max(1, horizon)
    ocw_trigger_per_run = ocw_hits / max(1, horizon)

    return {
        "trajectories": trajectories,
        "kesc_per_run": kesc_per_run,
        "kesc_mean": float(np.mean(kesc_per_run)),
        "kesc_std": float(np.std(kesc_per_run)),
        "ocw_trigger": float(np.mean(ocw_trigger_per_run)),
    }


def barrier_potential(x: np.ndarray, gamma: float) -> np.ndarray:
    return gamma * ((x * x - 1.0) ** 2) / 4.0


def plot_potential_and_trajectories(
    gamma: float,
    sim: Dict[str, np.ndarray],
    max_runs_to_plot: int = 12,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    xs = np.linspace(-2.0, 2.0, 500)
    U = barrier_potential(xs, gamma)
    axes[0].plot(xs, U)
    axes[0].axvline(0.0, linestyle="--", alpha=0.7)
    axes[0].axvspan(-0.2, 0.2, alpha=0.15)
    axes[0].set_title(f"Double-well potential UΓ(x), Γ={gamma:.2f}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("UΓ(x)")
    axes[0].grid(alpha=0.2)

    traj = sim["trajectories"]
    t = np.arange(traj.shape[1])
    for i in range(min(max_runs_to_plot, traj.shape[0])):
        axes[1].plot(t, traj[i], alpha=0.75)
    axes[1].axhline(0.0, linestyle="--", alpha=0.7)
    axes[1].axhspan(-0.2, 0.2, alpha=0.15)
    axes[1].set_title("Langevin trajectories")
    axes[1].set_xlabel("time step")
    axes[1].set_ylabel("x_t")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    return fig


def sweep_gamma_table(
    gammas: np.ndarray,
    temperature: float,
    dt: float,
    horizon: int,
    runs: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for i, gamma in enumerate(gammas):
        sim = simulate_langevin(
            gamma=float(gamma),
            temperature=temperature,
            dt=dt,
            horizon=horizon,
            runs=runs,
            seed=seed + 1000 + i,
        )
        rows.append(
            {
                "Gamma": round(float(gamma), 3),
                "kesc": floor3(sim["kesc_mean"]),
                "std": floor3(sim["kesc_std"]),
                "OCW Trigger": ceil3(sim["ocw_trigger"]),
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# Sidebar controls
# ============================================================
st.title("Meaning-Generation OS Demo")
st.caption(
    "Minimal theory-side demo: minority-mode preservation, barrier gap Γ, probability floor η*, and double-well barrier dynamics."
)

with st.sidebar:
    st.header("Global settings")
    seed = st.slider("Seed", min_value=1, max_value=9999, value=184, step=1)
    st.markdown("---")
    st.header("Exp-A: Long-tail classification")
    n_major = st.slider("Class 0 count", 100, 2000, 900, 10)
    n_mid = st.slider("Class 1 count", 20, 500, 90, 5)
    n_min = st.slider("Class 2 count", 10, 300, 30, 5)
    spread = st.slider("Cluster spread", 0.20, 1.20, 0.55, 0.05)
    margin_threshold = st.slider("Margin threshold m0", 0.05, 1.00, 0.30, 0.05)
    barrier_strength = st.slider("Barrier strength", 0.0, 5.0, 1.5, 0.1)
    lr = st.slider("Learning rate", 0.01, 1.0, 0.15, 0.01)
    epochs = st.slider("Training epochs", 50, 1200, 350, 25)
    l2 = st.slider("L2 regularization", 0.0, 0.1, 0.005, 0.001)

    st.markdown("---")
    st.header("Exp-B: Double-well Langevin")
    gamma = st.slider("Barrier scale Γ", 0.1, 3.0, 1.5, 0.1)
    temperature = st.slider("Temperature T", 0.01, 0.25, 0.05, 0.01)
    dt = st.slider("Time step Δt", 0.001, 0.05, 0.01, 0.001)
    horizon = st.slider("Horizon", 50, 1000, 200, 10)
    runs = st.slider("Runs", 4, 64, 12, 1)


# ============================================================
# Data and training
# ============================================================
X, y, means = generate_long_tail_data(
    counts=(n_major, n_mid, n_min),
    seed=seed,
    spread=spread,
)
minority_class = int(np.argmin(np.bincount(y)))
X_train, X_test, y_train, y_test = train_test_split_stratified(
    X=X,
    y=y,
    test_ratio=0.2,
    seed=seed,
)

baseline_model = train_softmax_model(
    X_train,
    y_train,
    num_classes=3,
    lr=lr,
    epochs=epochs,
    l2=l2,
    barrier_strength=0.0,
    margin_threshold=margin_threshold,
    minority_class=minority_class,
    seed=seed,
)

barrier_model = train_softmax_model(
    X_train,
    y_train,
    num_classes=3,
    lr=lr,
    epochs=epochs,
    l2=l2,
    barrier_strength=barrier_strength,
    margin_threshold=margin_threshold,
    minority_class=minority_class,
    seed=seed,
)

baseline_train = classification_metrics(
    baseline_model, X_train, y_train, minority_class, margin_threshold
)
baseline_test = classification_metrics(
    baseline_model, X_test, y_test, minority_class, margin_threshold
)
barrier_train = classification_metrics(
    barrier_model, X_train, y_train, minority_class, margin_threshold
)
barrier_test = classification_metrics(
    barrier_model, X_test, y_test, minority_class, margin_threshold
)

sim = simulate_langevin(
    gamma=gamma,
    temperature=temperature,
    dt=dt,
    horizon=horizon,
    runs=runs,
    seed=seed,
)

# Paper-aligned demo decomposition:
#   Gamma = B_star - C0 * f_hat(pdel, rho, delta)
# This app keeps the paper symbols visible while using observable demo-side proxies.
# It is still a demo visualization, not the full outward-rounded Sigma1 certificate.
L_phi = 1.006958
L_V = 0.800001
mu_sigma = 0.799999
eta_cert = 1.0 / 20.0
c_del = 1.0
C0_demo = 1.0

pdel_proxy = min(0.95, 1.0 - barrier_test["rec_min"])
rho_proxy = min(0.95, abs(baseline_test["eta_star"] - barrier_test["eta_star"]))
delta_proxy = min(0.95, barrier_test["ocw_trigger"])

f_hat_demo = (
    L_phi * rho_proxy
    + L_V * delta_proxy / (eta_cert * mu_sigma)
    + c_del * math.log(1.0 / max(1e-8, 1.0 - pdel_proxy))
)
B_star_demo = max(0.0, gamma)
Gamma_demo = B_star_demo - C0_demo * f_hat_demo

gamma_demo_df = pd.DataFrame(
    {
        "term": ["B*", "C0", "f_hat", "pdel (proxy)", "rho (proxy)", "delta (proxy)", "Gamma_demo"],
        "value": [
            floor3(B_star_demo),
            floor3(C0_demo),
            floor3(f_hat_demo),
            ceil3(pdel_proxy),
            ceil3(rho_proxy),
            ceil3(delta_proxy),
            floor3(Gamma_demo),
        ],
    }
)


# ============================================================
# Layout
# ============================================================
summary_cols = st.columns(6)
summary_cols[0].metric("Minority class", minority_class)
summary_cols[1].metric("B*", f"{B_star_demo:.3f}")
summary_cols[2].metric("f_hat (demo)", f"{f_hat_demo:.3f}")
summary_cols[3].metric("Demo Γ = B* - C0 f_hat", f"{Gamma_demo:.3f}")
summary_cols[4].metric("Barrier model η* (test)", f"{floor3(barrier_test['eta_star']):.3f}")
summary_cols[5].metric("Barrier model Recmin (test)", f"{floor3(barrier_test['rec_min']):.3f}")

with st.expander("What this demo shows", expanded=False):
    st.markdown(
        """
- **Exp-A** shows minority-mode preservation on a long-tail 3-class problem.
- **Exp-B** shows that increasing the barrier scale suppresses escape across a double-well landscape.
- **Γ** is displayed in the paper notation **Γ = B* - C0 f_hat(pdel, ρ, δ)**.
- Here **B*** is tied to the visible barrier scale, while **f_hat** is built from observable demo-side proxies.
- This remains a demo-side visualization, not the full outward-rounded Σ1 certificate.
- **η*** is the softmax probability floor computed from the minority minimum margin.
        """
    )

tab1, tab2 = st.tabs(["Exp-A: Minority preservation", "Exp-B: Barrier dynamics"])

with tab1:
    st.subheader("Long-tail classification with a barrier-style minority margin")

    col_plot, col_metrics = st.columns([1.7, 1.0])
    with col_plot:
        fig_cls = plot_classification_panel(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            baseline_model=baseline_model,
            barrier_model=barrier_model,
            minority_class=minority_class,
        )
        st.pyplot(fig_cls, use_container_width=True)

        fig_margin = plot_margin_histograms(
            baseline_metrics=baseline_test,
            barrier_metrics=barrier_test,
            margin_threshold=margin_threshold,
        )
        st.pyplot(fig_margin, use_container_width=True)

    with col_metrics:
        st.markdown("### Test metrics")
        metric_df = pd.DataFrame(
            {
                "Model": ["Baseline", "Barrier"],
                "Accuracy": [floor3(baseline_test["acc"]), floor3(barrier_test["acc"])],
                "Recmin": [floor3(baseline_test["rec_min"]), floor3(barrier_test["rec_min"])],
                "OCW Trigger": [ceil3(baseline_test["ocw_trigger"]), ceil3(barrier_test["ocw_trigger"])],
                "eta*": [floor3(baseline_test["eta_star"]), floor3(barrier_test["eta_star"])],
                "min margin": [floor3(baseline_test["min_margin"]), floor3(barrier_test["min_margin"])],
            }
        )
        st.dataframe(metric_df, use_container_width=True, hide_index=True)

        st.markdown("### Paper-aligned Γ display")
        st.dataframe(gamma_demo_df, use_container_width=True, hide_index=True)

        st.markdown("### Train metrics")
        train_df = pd.DataFrame(
            {
                "Model": ["Baseline", "Barrier"],
                "Accuracy": [floor3(baseline_train["acc"]), floor3(barrier_train["acc"])],
                "Recmin": [floor3(baseline_train["rec_min"]), floor3(barrier_train["rec_min"])],
                "OCW Trigger": [ceil3(baseline_train["ocw_trigger"]), ceil3(barrier_train["ocw_trigger"])],
                "eta*": [floor3(baseline_train["eta_star"]), floor3(barrier_train["eta_star"])],
            }
        )
        st.dataframe(train_df, use_container_width=True, hide_index=True)

        st.markdown("### Interpretation")
        st.markdown(
            """
- **Recmin**: minority recall.
- **OCW Trigger**: fraction of samples below the target margin.
- **eta***: probability floor from the minority minimum margin.
- **Gamma_demo**: visible demo-side decomposition of the paper quantity **Γ = B* - C0 f_hat**.
- The barrier model is intended to trade some average efficiency for stronger minority preservation.
            """
        )

with tab2:
    st.subheader("Double-well Langevin dynamics")

    col_plot, col_metrics = st.columns([1.7, 1.0])
    with col_plot:
        fig_dyn = plot_potential_and_trajectories(gamma=gamma, sim=sim)
        st.pyplot(fig_dyn, use_container_width=True)

    with col_metrics:
        st.metric("kesc", f"{floor3(sim['kesc_mean']):.3f}")
        st.metric("std", f"{floor3(sim['kesc_std']):.3f}")
        st.metric("OCW Trigger", f"{ceil3(sim['ocw_trigger']):.3f}")
        st.metric("e^(-Γ/T)", f"{math.exp(-gamma / max(1e-8, temperature)):.6f}")

        st.markdown("### Γ sweep")
        gamma_grid = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        sweep_df = sweep_gamma_table(
            gammas=gamma_grid,
            temperature=temperature,
            dt=dt,
            horizon=horizon,
            runs=runs,
            seed=seed,
        )
        st.dataframe(sweep_df, use_container_width=True, hide_index=True)

        st.markdown("### Interpretation")
        st.markdown(
            """
- Larger **Γ** raises the barrier.
- Higher barriers suppress sign-crossing events.
- The shaded region around 0 corresponds to a challenge-window style neighborhood near the saddle.
            """
        )

st.markdown("---")
st.markdown(
    "**Note**: This is a minimal theory demo. It visualizes the long-tail / probability-floor side and the barrier / escape-suppression side in a single app."
)
