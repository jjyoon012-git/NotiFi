# 9안: Fall trajectory reconstruction

## 목표

CSI만으로 낙상 전후의 전체 3D 자세 궤적을 복원한다. `어느 관절이 먼저 바닥에
닿았는가`를 휴리스틱 정답으로 강제하지 않는다. 대신 사람이 어떤 방향과 자세로
넘어졌는지, pelvis가 얼마나 이동하고 내려갔는지, 몸통과 말단 관절이 어떤 경로를
그렸는지를 직접 학습한다.

평가는 `single_split`의 ajh/lmh/mhw, E01/E02/E03 trial split을 사용한다.
train/validation/test pose trial은 각각 1,556/405/405개다. yja와 LOSO는 이 단계에
사용하지 않는다.

## 현재 파이프라인

```text
CSI [B,T,links,subcarriers,real/imag]
  -> frozen motion-first CSI encoder
  -> frozen calibrated 7안 pose/root reconstruction
  -> V9 trajectory feature assembly
       temporal feature + CSI multi-scale motion
       root velocity + body-group speed + risk probability
  -> dilated temporal blocks (1/2/4/8) + Transformer
  -> bone-length-preserving 6D rotation residual
  -> low-frequency root anchor/step residual
  -> validation-selected residual scale
  -> temporal denoising motion prior
  -> CSI-only SMPL-22 trajectory
```

구현은 `notifi_pose/seen_v4.py`, `notifi_pose/motion_prior_v9.py`에 있다.

## 손실과 시간 정렬

9안의 새 trajectory loss는 다음 항을 사용한다.

- 전체 frame pose와 root 위치
- 5-frame pose/root displacement
- 낙상 전후 root drop
- torso/shoulder orientation
- 마지막 구간 endpoint pose
- 선택적인 bounded piecewise alignment

bounded alignment는 8개 연속 구간마다 최대 15-frame offset 후보를 만들고, dynamic
programming으로 순서가 뒤집히지 않는 경로만 허용한다. offset 크기와 인접 구간 offset
변화를 벌점으로 둔다. timestamp를 대체하거나 GT를 영구 이동하지 않으며, frame-aligned
loss도 항상 함께 유지한다.

가속도 최대 프레임, 최초 충돌 관절, 바닥 근접 기반 impact score는 이 새 loss에 없다.
다만 V9의 base가 과거 7안 checkpoint이므로 과거 contact/impact supervision의 영향까지
완전히 사라진 것은 아니다.

## 실험 결과

| 모델 | MPJPE | Dynamic | Root | Danger | Danger distal | Danger endpoint | Speed ratio | 결정 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 7안 base | 21.29cm | 20.90cm | 31.81cm | - | - | - | 1.167 | 비교 기준 |
| 9A trajectory | **20.60cm** | 20.40cm | **31.61cm** | 51.15cm | 55.72cm | 69.72cm | 1.217 | 위치 개선, 속도 과장 |
| 9B + bounded alignment | 21.29cm | 20.90cm | 31.94cm | 51.95cm | 56.93cm | 71.23cm | 1.167 | pose scale 0 선택, 기각 |
| 9C + motion prior | 20.68cm | **20.41cm** | **31.61cm** | **51.14cm** | **55.64cm** | **69.66cm** | **1.163** | 현재 권장 |

9A는 validation에서 pose residual 0.15, root residual 0.5가 선택됐다. 위치 오차는
낮아졌지만 test speed ratio가 1.217로 물리 gate 1.2를 조금 넘었다.

9B는 alignment loss weight 0.15로 학습했다. validation calibration이 pose residual
strength 0을 선택했고 test도 base보다 좋아지지 않았다. 현재 correlation-based offset은
정확한 동작 correspondence를 제공하지 못하므로 정렬 loss는 공식 모델에서 끈다.

9C prior는 train GT만 noise/frame-mask/joint-mask로 오염시켜 원래 trajectory를 복구하도록
학습했다. noisy validation pose의 MPJPE를 4.55cm에서 3.70cm로 줄였다. validation에서
strength 1.0이 선택됐고 9A test speed ratio를 1.217에서 1.163으로 낮췄다. 전체 MPJPE는
0.07cm 악화됐지만 danger distal과 endpoint는 각각 0.07cm, 0.06cm 개선됐다.

## 외부 낙상 데이터 판단

`3D Skeletons - UP-Fall` 공개 데이터는 5명, 두 카메라의 MediaPipe BlazePose 33-joint
CSV다. 샘플 좌표는 이미지 정규화 x/y와 카메라 상대 z이며 timestamp, CSI, metric world
root, GVHMR/SMPL pose parameter가 없다. 파일명 규칙도 카메라 간 완전히 일치하지 않아
동일 trial의 view가 train/validation에 나뉘는 leakage 위험이 있다.

따라서 원본 33-joint 좌표를 현재 SMPL-22 metric prior에 직접 섞지 않았다. 사용하려면
먼저 동일 trial의 두 view를 묶고, MediaPipe-to-SMPL joint mapping, pelvis 중심화, bone-scale
retargeting, view canonicalization을 거친 **relative-pose prior 전용** 데이터로 만들어야 한다.
root trajectory나 CSI encoder 학습에는 사용할 수 없다.

현재 9C는 공개 데이터 대신 좌표계가 정확한 내부 train GT에 시간/관절 누락과 Gaussian
noise를 주는 self-supervised denoising augmentation을 사용한다. 외부 데이터 혼합보다
작지만 정답 좌표계를 훼손하지 않는 선택이다.

## 남은 문제와 다음 순서

1. Danger absolute MPJPE 51.14cm와 endpoint 69.66cm가 여전히 가장 큰 병목이다.
2. Bounded alignment가 실패했으므로 영상으로 trial 시작 anchor를 소량 검증하고, 이후
   monotonic soft-DTW나 CTC-style correspondence를 별도 ablation해야 한다.
3. 9C는 속도 안정화에는 효과가 있지만 danger speed correlation은 0.352로 낮다. 다음
   모델은 danger oversampling만이 아니라 fall direction/root displacement 조건을 encoder에
   명시적으로 학습해야 한다.
4. 7안 base를 impact/contact proxy 없이 full-sequence objective로 처음부터 재학습해야
   휴리스틱 의존성을 완전히 제거할 수 있다.
5. seen MPJPE 20cm, danger MPJPE 45cm, root 25cm gate에 접근한 뒤 LOSO calibration을
   재개한다.

## 재현

```powershell
python -m notifi_pose.tools.train_seen_v4_trajectory `
  --epochs 12 --batch-size 8 --alignment-weight 0 `
  --run-dir work_v2/runs/seen_v4_v9a_no_impact

python -m notifi_pose.tools.train_seen_v4_trajectory `
  --epochs 12 --batch-size 8 --alignment-weight 0.15 `
  --run-dir work_v2/runs/seen_v4_v9b_bounded_alignment

python -m notifi_pose.tools.train_motion_prior_v9 `
  --epochs 10 --batch-size 12 `
  --run-dir work_v2/priors/temporal_denoiser_v9

python -m notifi_pose.tools.calibrate_motion_prior_v9 `
  --trajectory-run work_v2/runs/seen_v4_v9a_no_impact `
  --prior-run work_v2/priors/temporal_denoiser_v9 `
  --run-dir work_v2/runs/seen_v4_v9c_motion_prior
```

9A/9B/9C의 `results.json`과 checkpoint는 위 run directory에 있다. Git에는 재현 코드와
요약 결과만 포함하고 대용량 checkpoint는 포함하지 않는다.
