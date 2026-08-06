# Calibration Deployment Contract

## Status

The current leakage-free calibration stack is **experimental**. CAL42 improves yja/E02
pose and action reconstruction by adding guarded physical-phase evidence to
CAL27, while CAL33 improves danger-oriented retrieval
on yja/E02. Neither passes the multi-subject unseen risk audit. Production code
must reject the calibration rather than silently treating it as READY.

## Physical Setup

The receiver/transmitter direction contract is fixed for training and runtime:

| Board | Direction |
|---|---|
| RX | North |
| TX1 | South |
| TX2 | West |
| TX3 | East |

The model input order is always `[TX1, TX2, TX3]`. Statistical permutation is
allowed only as an offline diagnostic view. It must never rewrite the physical
pose or risk link order.

## Calibration Prompt

CAL27 uses 64 labeled safe trials:

| Action | Repeats |
|---|---:|
| walking | 8 |
| standing still | 8 |
| sitting still | 8 |
| lying still | 8 |
| lie to stand | 8 |
| stand to lie | 8 |
| sit to stand | 8 |
| stand to sit | 8 |

The prompt is used to build target-local prototypes and to choose prototype
temperature/weight through two-fold repeat cross-validation. Calibration never
reads target query action/risk labels, pose GT, video, person ID, or environment
ID.

## Capability Gate

The runtime distinguishes an experiment candidate from a deployable
calibration:

| Capability | Required evidence | Current yja/E02 |
|---|---:|---:|
| Experimental action/pose | held-repeat >=35%, full support >=55% | 37.5% / 57.8% |
| Deployable action/pose | held-repeat >=60%, full support >=70% | FAIL |
| Deployable danger | target-relevant dynamic-risk evidence and held-subject audit | FAIL |

Safe support can bound safe false alarms, but it cannot prove that the danger
score points in the correct direction in a new person/environment. Therefore
`risk_ready` is always false for a safe-only prompt. The generic quality gate
can mark action/pose support as READY, but `accepted_for_normal_inference`
remains false unless independent validated risk evidence is supplied.

CAL42 remains experimental. Its pre-registered 15% phase branch improved mean
action accuracy on all eight audited yja/ajh/mhw/lmh environments, but the energy branch's
danger decision is immutable and the underlying ajh danger recall remains only
10-20%. The guard prevents regression of already-correct danger actions; it
does not create missing danger observability.

## Runtime

Normal loading rejects the current artifact:

```python
from notifi_pose.cal27_kp10 import Cal27ActionCalibrator

calibrator = Cal27ActionCalibrator.load("deployment_candidate.pt")
```

Research reproduction requires an explicit opt-in:

```python
calibrator = Cal27ActionCalibrator.load(
    "deployment_candidate.pt",
    allow_experimental=True,
)
output = calibrator(csi, link_mask)

action_logits = output["action_logits"]
assert output["risk_certified"] is False
assert output["accepted_for_normal_inference"] is False
```

CAL43's 25% weight produced better post-hoc numbers, but it was promoted after
inspecting target-query audits. It is therefore only a candidate for a new
sealed subject and must not replace CAL42 in reported final performance.

CAL42 requires independently fitted energy and physical-phase CAL27 branches.
It adds a second explicit experimental gate. The wrapper also verifies that
the branches use the expected feature modes and were fitted from the identical
support-row list, so calibration artifacts from different users cannot mix:

```python
from notifi_pose.cal27_kp10 import Cal27ActionCalibrator
from notifi_pose.cal42_kp10 import Cal42GuardedCalibrator

energy = Cal27ActionCalibrator.load(
    "energy_support_candidate.pt", allow_experimental=True
)
phase = Cal27ActionCalibrator.load(
    "physical_phase_support_candidate.pt", allow_experimental=True
)
calibrator = Cal42GuardedCalibrator(
    energy, phase, phase_weight=0.15, allow_experimental=True
)
output = calibrator(csi, link_mask)
assert output["calibration_status"] == "EXPERIMENTAL"
assert output["risk_certified"] is False
assert output["accepted_for_normal_inference"] is False
```

An application must label the pose as a CSI-conditioned simulation, not a
measured injury or collision sequence. It must not emit a certified danger
decision from `risk_logits_experimental`.

Runtime input validation also rejects an empty trial, a missing physical link,
less than 50% valid frame coverage on any link, or NaN/infinity marked as valid.
Masked packet loss remains supported; invalid samples must not be marked valid.

## Path To READY

1. Add independent source subjects, not additional repeats from the same four
   subjects. Current source-site transforms fail on a held-out person even when
   they succeed across rooms.
2. Define a safe, controlled dynamic-risk calibration maneuver. It must be
   ethically executable without asking a user to fall and must be validated as
   a predictor of actual falls before it can certify danger.
3. Lock model and calibration settings using source-only/validation protocols,
   then run a new sealed multi-person LOSO audit once.
4. Require both capability gates and runtime link/coverage checks. A failed gate
   returns REJECT; it does not fall back to an arbitrary pose.

The complete fixed metrics are in
[`results/cal23_cal34_calibration_audit.json`](results/cal23_cal34_calibration_audit.json).
