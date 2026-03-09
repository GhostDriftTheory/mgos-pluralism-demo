# MG-OS Demo

<p align="center">
  <img src="docs_mgos_framework.png.png" width="720">
</p>

Repository:[https://github.com/GhostDriftTheory/mgos-pluralism-demo](https://github.com/GhostDriftTheory/mgos-pluralism-demo)

Paper (preprint):[https://zenodo.org/records/17712891](https://zenodo.org/records/17712891)

Project page:[https://www.ghostdriftresearch.com/](https://www.ghostdriftresearch.com/)

Minimal theory-side demonstration of **Meaning-Generation OS (MG-OS)**.

This repository visualizes two core ideas from the MG-OS preprint:

1. **Minority-mode preservation** in long-tail classification
2. **Barrier-driven escape suppression** in a double-well Langevin system

The goal is not benchmark performance.
The goal is to make the theory visible in a small, self-contained form.

---

## What this demo shows

### Exp-A: Long-tail classification

A 3-class long-tail toy problem compares:

* a baseline softmax classifier
* a barrier-style classifier inspired by the MG-OS formulation

The app displays:

* **Accuracy**
* **Recmin** (minority recall)
* **OCW Trigger**
* **eta*** (probability floor from the minority minimum margin)
* **Demo Gamma = B* - C0 f_hat** (paper-aligned display)

This is the visual side of **minority-mode preservation**.

### Exp-B: Double-well Langevin dynamics

A 1D double-well system visualizes how increasing the barrier scale **Gamma** suppresses escape events.
In the paper notation, the certified margin is:

**Gamma := B* - C0 f_hat(pdel, rho, delta) > 0**

In this demo, **B\*** is tied to the visible barrier scale and **f_hat** is represented by observable toy-side proxy terms.

The app displays:

* the double-well potential
* sample trajectories
* **kesc** (escape-rate proxy)
* **OCW Trigger**
* a Gamma sweep table

This is the visual side of **barrier-certified stability**.

---

## Paper-aligned certificate view

For visual alignment with the MG-OS preprint, the app exposes a demo-side quantity in the same notation as the theoretical certificate:

**Gamma_demo = B* - C0 * f_hat(pdel, rho, delta)**

This is a theory-aligned visualization layer.
It is **not** the full outward-rounded rational / Sigma1 certificate from the paper.

---

## Repository structure

* `app.py` — Streamlit app with both demos
* `requirements.txt` — minimal dependencies

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Interpretation

This repository is a **minimal theory demo**, not a production implementation.

It is designed to make the following points visible:

* minority modes can be made harder to suppress
* a barrier margin can be treated as an operational safety quantity
* theoretical quantities can be linked to observable toy metrics
* the paper notation for **Gamma** can be read against the toy observables shown in the app

---

## Scope

This demo is intentionally small.

It does **not** include:

* a full MG-OS implementation
* a full attention block or MoE stack
* large-scale training
* real-world deployment claims
* the full outward-rounded certificate package from the paper

It is a compact visualization layer for the theory.


