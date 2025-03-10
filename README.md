# Hierarchical Refinement (HiRef)

This is the repository for the paper **["Hierarchical Refinement: Optimal Transport to Infinity and Beyond,"](https://arxiv.org/pdf/2503.03025)** which scales optimal transport **linearly in space** and **log-linearly in time** by using a hierarchical strategy that constructs multiscale partitions from low-rank optimal transport.

<img src="images/fig1-2.png" alt="Hierarchical Refinement Schematic" width="50%">

*Figure 1: Hierarchical Refinement algorithm: low-rank optimal transport is used to progressively refine partitions at the previous scale, with the coarsest scale partitions denoted $X^{(1)}, Y^{(1)} $, and the finest scale partitions $X^{(\kappa)}, Y^{(\kappa)}$ corresponding to the individual points in the datasets.*

---

## **Usage**

**Hierarchical Refinement (HiRef)** only requires two **n×d** dimensional point clouds `X` and `Y` (**torch tensors**) as input.

Before running HiRef, call the **rank-annealing scheduler** to find a sequence of ranks that **minimizes the number of calls** to the low-rank optimal transport subroutine while remaining under a machine-specific maximal rank.

### **Rank Scheduler Parameters**
- `n` : The size of the dataset
- `hierarchy_depth (κ)` : The depth of the hierarchy of levels used in the refinement strategy
- `max_Q` : The maximal terminal rank at the base case
- `max_rank` : The maximal rank of the intermediate sub-problems

---

## **Getting Started**

### **1. Compute the Optimal Rank Schedule**
Import the **rank annealing** module and compute the rank schedule:

```python
import rank_annealing

rank_schedule = rank_annealing.optimal_rank_schedule(
    n=n, hierarchy_depth=hierarchy_depth, max_Q=max_Q, max_rank=max_rank
)
```

### **2. Initialize Hierarchical Refinement**
Import HR_OT and initialize the class using only the point clouds (you can additionally input the cost `C` if desired) along with any relevant parameters (e.g., **sq_Euclidean**) for your problem.
```python
import HR_OT
hrot = HR_OT.HierarchicalRefinementOT.init_from_point_clouds(
    X, Y, rank_schedule, base_rank=1, device=device
)
```
### **3. Run Hierarchical Refinement**
Run and return paired tuples from `X` and `Y`:

```python
F_lr = hrot.run(return_as_coupling=False)
```
### **4. Compute the OT primal cost**
To print the Optimal Transport (OT) cost, simply call:
```python
cost_hr_ot = hrot.compute_OT_cost()
print(f"Refinement Cost: {cost_hr_ot.item()}")
```

## Contact

For questions, discussions, or collaboration inquiries, feel free to reach out at [ph3641@princeton.edu](mailto:ph3641@princeton.edu) or [jg7090@princeton.edu](mailto:jg7090@princeton.edu).
