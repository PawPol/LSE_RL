# Phase VII Adaptive-β — Planner Clarifications

## 1. Operator Import Surface

Choose **option (b)**.

Extract the stateless TAB / safe weighted log-sum-exp kernel into:

```
src/lse_rl/operator/tab_operator.py
```

Both Phase III–VI and Phase VII must import the same kernel.

**Requirements:**
- No duplication of operator logic
- Full backward compatibility
- Add regression/property tests verifying numerical equivalence with:
  - `SafeWeightedCommon.compute_safe_target`
- Refactor must be transparent to existing phases

---

## 2. Environment Base Class

Use **MushroomRL-compatible environments**.

- Follow Phase I–VI precedent
- Subclass `Environment`
- Preserve compatibility with:
  - existing runner
  - logging stack

**Implementation notes:**
- Handle `(state, info)` outputs correctly
- Use `int(np.asarray(x).flat[0])` normalization where required

Plain Python classes are allowed only as internal helpers, not as the primary API.

---

## 3. Wrong-Sign Definition

Keep wrong-sign only where the canonical sign is well-defined:

- **Delayed Chain**:
  - canonical: +β
  - wrong-sign: −β

- **Gridworld Hazards**:
  - canonical: −β
  - wrong-sign: +β

- **RPS / Switching Bandit / Self-play**:
  - do NOT define wrong-sign
  - use:
    - fixed-positive
    - fixed-negative

---

## 4. Self-Play RPS

Defer to **Stage B**.

- Exclude from Stage A
- Add only if:
  - adaptive-β shows stable signal in Stage A
  - variance is controlled

---

## 5. Bandit Mechanism Metrics

- Keep bandit for **performance benchmarking**
- Exclude from **primary mechanism validation**

Reason:
- Horizon = 1 ⇒ `v_next = 0`
- Alignment and effective discount are trivial

**Use bandit for:**
- regret
- recovery after shifts
- AUC

**Do NOT use bandit for:**
- alignment-rate claims
- effective-discount analysis

---

## Summary

- Operator: shared kernel (no duplication)
- Envs: MushroomRL integration
- Wrong-sign: selective
- Self-play: Stage B only
- Bandit: performance only, not mechanism evidence
