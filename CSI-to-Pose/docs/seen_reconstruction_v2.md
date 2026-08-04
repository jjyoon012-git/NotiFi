# Seen Reconstruction V2

`SeenReconstructionV2Net`은 기존 calibrated seen cascade를 identity 기준으로 유지하면서
데이터 품질, 시간 조건, kinematics, root flow, 부상 관련 출력을 추가한다. 입력에는 CSI와
유효 link mask만 사용하며 action, phase, contact GT는 추론 입력으로 사용하지 않는다.

## Pipeline

```text
timestamp-aligned CSI
  -> site baseline subtraction + per-link normalization
  -> validated GraphFormer/action/root cascade (coarse pose and root)
  -> external motion-first raw/delta encoder
  -> gated temporal feature fusion
  -> predicted action + frame speed/moving/phase/impact conditioning
  -> keyframe 6D bone-rotation residual
  -> bounded high-frequency Cartesian residual
  -> anchor + integrated keyframe root-step residual
  -> pose, root, contact, first-contact, impact-speed, floor outputs
```

## Requested changes 1-7

1. **Quality weighting**: `quality.py` combines recorded timestamp availability, valid link count,
   zero-lag CSI/GT motion correlation, and audit status. The train loader uses a class-balanced
   weighted sampler and the loss uses the same reliability score. No timestamp is automatically
   shifted.
2. **Root trajectory factorization**: the current root is decomposed into the first-frame anchor and
   per-frame step. A 4-frame keyframe residual corrects the step and is integrated over time.
3. **Phase-aware conditioning**: speed, moving probability, four fall-phase probabilities, impact
   probability, predicted action, and risk are projected into every temporal frame.
4. **Rotation-based decoding**: the low-frequency branch predicts 6D rotations for the 21 SMPL-22
   bones. Forward kinematics reconstructs joints while preserving the coarse model's bone lengths.
5. **Low/high separation**: large motion is represented by 4-frame rotation keyframes. A separate
   Cartesian branch is bounded to 2cm and calibrated independently.
6. **Injury heads**: the model predicts feet contact, eight injury-joint contacts, first-contact joint,
   per-joint impact speed, and floor height. Targets are derived from GVHMR GT only during training.
7. **Partial fine-tuning**: heads are trained first. The last GraphFormer temporal block and the last
   motion-first temporal block are then opened at one tenth of the head learning rate with feature
   distillation to the frozen pre-finetune teacher.

## Validation calibration

The unconstrained epoch-18 model reached 18.11cm test MPJPE but its pose-speed ratio was 2.088, so it
is rejected. Component diagnosis showed that rotation-only increased validation speed ratio from
1.117 to 1.971; the 2cm high-frequency branch only increased it to 2.025.

Validation compared rotation, high-pose, and root strengths. A hard pose-speed gate of 0.8-1.2 was
applied before test was opened. The selected configuration is:

```text
rotation_strength = 0.10
high_pose_strength = 0.00
root_strength = 0.50
```

## Seen test result

| Metric | Previous seen | V2 calibrated |
|---|---:|---:|
| MPJPE | 21.68cm | **21.29cm** |
| Dynamic MPJPE | 21.22cm | **20.90cm** |
| Distal MPJPE | 32.16cm | **31.53cm** |
| Impact MPJPE | 55.27cm | **54.72cm** |
| Root error | 32.36cm | **32.33cm** |
| Pose-speed ratio | 1.141 | **1.167** |

Auxiliary test results are injury-contact F1 0.354, feet-contact F1 0.708,
first-contact accuracy 0.378 over 90 danger trials, impact-joint speed MAE 0.553m/s,
and floor-height MAE 2.72cm. These heads are initial baselines and are not yet suitable for clinical
injury conclusions.

## Run

```powershell
python -m notifi_pose.tools.train_seen_v2 `
  --head-epochs 12 --finetune-epochs 6 --patience 4 --batch-size 8

python -m notifi_pose.tools.diagnose_seen_v2_components --dataset val
python -m notifi_pose.tools.calibrate_seen_v2
```

Local artifacts:

```text
work_v2/runs/seen_reconstruction_v2/best_model.pt
work_v2/runs/seen_reconstruction_v2/calibration.json
work_v2/runs/seen_reconstruction_v2/calibrated_model.pt
```
