# NotiFi CSI-to-Pose

## 2026-08-04 최종 감사: 새 학습 전 P0 수리

현재 **승격 가능한 모델은 없다.** 보정된 lmh GT 295개가 실제 D 드라이브 학습 데이터와
cache에 한 건도 반영되지 않았고, 그중 50개가 반복 평가한 `seen_dev_test`에 들어 있다.
old→corrected target 변화는 local 평균 `46.05cm`, absolute 평균 `68.10cm`이므로 지금까지의
V9A/V9C와 prototype 수치는 모두 `pre-GT-repair historical`로만 보존한다.

또한 `amplitude + sanitized phase`를 I/Q처럼 회전한 RF augmentation 때문에 frozen motion
encoder도 재사용할 수 없다. 64개 train trial 감사에서 증강 phase 표준편차는 원래의 31.4배였고,
55.0%가 `|phase| > pi`였다. cache는 GT/CSI/timestamp/split hash를 저장하지 않아 원본이 바뀌어도
stale 배열을 최신으로 판정한다.

V9C prior는 V9A보다 local MPJPE를 평균 `0.072cm` 악화했고 bootstrap 95% CI 전체가 음수라
기각한다. 수정 전 비교 모델은 V9A지만, 이것도 새 baseline으로 재학습해야 한다. V9A danger의
local 동작 크기는 GT의 약 23%뿐이며, 같은 site/action 안의 trial별 local residual cosine은
`0.050`이라 사용자가 본 정지 자세 collapse를 수치로 확인했다.

기존 robust 학습의 class-balanced replacement sampler도 epoch마다 고유 trial을 약 `60%`만 보여주고
draw의 약 `40%`를 중복했다. raw danger `17.6%`는 sampled `29.4%`, inverse-risk CE 질량은
`51~53%`가 되었는데 GroupDRO까지 동시에 적용됐다. 따라서 robust 수치는 도메인 기법 하나의 성능으로
해석하지 않으며, V10은 기본 순회 sampler에서 balance/CE/DRO를 각각 독립 검증한다.

V9A 실행 때는 validation speed gate 상한이 `1.20`이라 strength `0.15`가 선택됐지만 V9B와 현재
source는 `1.15`를 쓴다. 현재 코드로 V9A 후보를 재선택하면 strength `0.0`이므로 V9A 점수는 코드
재현값이 아닌 historical artifact다. V10은 선택식·threshold·smoother·metric version과 source tree
hash를 checkpoint에 함께 고정한다.

현재 index에는 physical installation ID/geometry가 없어 `subject_environment` 9개 domain만 있다.
따라서 기존 LOSO는 순수 사람 일반화가 아니라 participant와 그 사람의 설치를 함께 바꾸는 joint-shift
진단이다. 사람/환경 효과를 분리하려면 여러 사람이 같은 설치를 공유하는 factorial 추가 수집이 필요하다.

다음 **V10**은 GT 복구와 cache v4를 먼저 끝낸 뒤 `global action motion bank + 10초 empty-room
calibration adapter + monotonic progress + CSI-specific bone-direction/root residual`을 하나의
end-to-end 모델로 처음부터 학습한다. `site×action` prototype은 강한 seen 진단 기준선으로만 쓰고
배포 decoder에 site ID를 하드코딩하지 않는다.

단, 현재 ajh/lmh/mhw 2,366개 GT는 joint position만 있고 yja 263개에만 SMPL rotation이 있다.
따라서 V10 local head는 네 사람의 SMPL body pose를 같은 버전으로 재추출할 때만 SO(3)를 쓰며,
그 전에는 `unit bone direction + canonical FK`로 구현한다. joint position에서 식별할 수 없는 bone
twist를 임의 정답으로 만들지 않는다.

추가 감사에서 같은 사람의 GVHMR bone length도 trial마다 중앙 `2.85~3.17%` 흔들렸고, danger의
동작 시작 progress는 p05~p95 `15~135 frame`에 퍼졌다. nonlinear time-warp oracle의 danger
absolute 이득은 `2.40cm`로 유효하지만 주 병목을 해결할 크기는 아니었다. 반면 subcarrier를 고정
permutation하면 V9A local/root가 `+7.54/+17.12cm` 악화해 주파수 구조는 확실히 유효했다. 따라서
V10은 raw/canonical skeleton을 분리하고 dynamic-only progress를 쓰며 frequency token pooling을
늦춘다. diffusion과 자유로운 DTW는 CSI residual gate 이후로 미룬다.

- 최종 코드 감사·복구 순서·V10 구조·실험 gate:
  [`docs/final_code_audit_and_v10_execution_plan.md`](docs/final_code_audit_and_v10_execution_plan.md)
- 파일별 cache/model/checkpoint API와 학습 ladder:
  [`docs/v10_file_level_implementation_spec.md`](docs/v10_file_level_implementation_spec.md)
- 선행 종합 진단과 최신 연구 연결:
  [`docs/comprehensive_diagnosis_and_plan_v10.md`](docs/comprehensive_diagnosis_and_plan_v10.md)
- 기계 판독 결과 요약:
  [`docs/results/comprehensive_audit_v10.json`](docs/results/comprehensive_audit_v10.json)
- 최신순 실험 기록: [`docs/experiment_log.md`](docs/experiment_log.md)

당장 하지 않는 경로는 V9 checkpoint warm-start, V9C prior, impact/injury heuristic, danger weight 추가,
empty-room `PerLinkNorm` refit, site ID 기반 production prototype, smoothed-only 보고, seen gate 이전 LOSO/TTA다.
이 항목들은 이미 실패했거나 현재 데이터로 식별할 수 없거나 원인 분리를 막는다.

아래 9C 설명과 1~9안 기록은 수정 전 재현을 위한 historical baseline이며 현재 권장 실행 경로가 아니다.

NotiFi CSI-to-Pose는 WiFi CSI 신호만으로 사람의 자세와 움직임 흐름을 **3D skeleton proxy**로 복원하는 실험 기능입니다.

![Unstable walking CSI-to-Pose reconstruction](assets/unstable-walking-csi-to-pose-human-readable.gif)

![CSI-to-Pose overview](assets/csi-to-pose-overview.png)

## Historical seen-first development status

현재 개발은 `work_v2/splits/experiments.json`의 `single_split`을 사용한다. 이 protocol은
`ajh/lmh/mhw`와 `E01/E02/E03`을 train, validation, test에 모두 포함하고 trial만 분리한
**seen-subject + seen-environment + unseen-trial** 평가다. pose trial 수는
train 1,556 / validation 405 / test 405이며 split 간 trial 중복은 없다.

아래 pipeline은 **수정 전 9안 Stage C 재현 경로**다. GT와 RF augmentation 감사 전에는 권장
후보였지만 지금은 실행 기준이 아니다. 낙상 순간의 최대 가속도나 최초 충돌 관절을 정답으로
강제하지 않고 전체 자세 궤적을 복원하려던 구조적 변천을 보존한다.

```text
GraphFormer baseline
  + raw/delta dual-branch motion-first CSI encoder
  + predicted-action-conditioned pose residual
  + low-frequency keyframe root residual
  + quality-weighted phase-aware 6D rotation refinement
  + validation-selected V2 branch calibration
  + contact-guided root anchor / velocity refinement
  + validation-selected root strength 0.5
  + full-sequence trajectory residual (9A, pose 0.15 / root 0.5)
  + GT-only temporal denoising motion prior (9C, strength 1.0)
```

수정 전 seen dev_test에서 기존 7안 대비 MPJPE는 `21.29cm -> 20.68cm`, dynamic MPJPE는
`20.90cm -> 20.41cm`, root error는 `31.81cm -> 31.61cm`, pose-speed ratio는
`1.167 -> 1.163`이다. danger MPJPE는 `51.14cm`, danger distal MPJPE는
`55.64cm`로 컸다. 이후 bootstrap에서 9C prior가 9A보다 유의하게 나빠 기각됐고, 두 모델 모두
오염된 GT/cache와 잘못된 RF augmentation을 사용했으므로 corrected baseline 재학습 전에는 비교
모델로만 남긴다. 구간별 시간 이동을 허용한 9B도 validation에서 pose branch가 `0`으로 선택돼
기각했다.

8안의 contact 출력은 분석용으로 남아 있지만 9안의 새 학습 loss는 `최대 가속도 프레임`,
`최초 충돌 관절`, 휴리스틱 impact score를 사용하지 않는다. 동결된 7안 base에는 과거
학습의 영향이 남아 있으므로 다음 base 재학습 때 이 의존성도 완전히 제거할 예정이다.

번호, 시각, 목적, 방법, 결과, 채택 여부를 포함한 누적 기록은
[`docs/experiment_log.md`](docs/experiment_log.md)에 있다.
개선안 1-7의 코드 대응과 실행 방법은
[`docs/seen_reconstruction_v2.md`](docs/seen_reconstruction_v2.md)와
[`docs/seen_reconstruction_v3.md`](docs/seen_reconstruction_v3.md),
[`docs/impact_event_v8.md`](docs/impact_event_v8.md),
[`docs/fall_trajectory_v9.md`](docs/fall_trajectory_v9.md)에 있다. 1안부터의
protocol별 지표 이력은 [`docs/results/model_plan_history.json`](docs/results/model_plan_history.json)에 있다.

## Historical GVHMR v2 baseline execution

The historical code path uses 3 CSI links, exact/partial recorded timestamps, and
GVHMR SMPL-22 targets. Do not use these commands for a new reported result until the
P0 GT repair, cache v4 rebuild, and augmentation fix are complete. The MediaPipe
13-point workflow later in this README is legacy.

```text
CSI [B,304,3,114,2]
  -> per-link calibration normalization
  -> shared subcarrier Conv encoder
  -> masked link-attention fusion
  -> local temporal convolution + Transformer
  -> joint queries + SMPL-22 graph blocks
  -> hybrid direct/tree pose decoder
  -> separate root, motion, class, and risk heads
```

Prepare the index and cache:

```powershell
python -m notifi_pose.tools.build_index --no-verify-files
python -m notifi_pose.tools.link_quality --workers 8
python -m notifi_pose.tools.build_splits
python -m notifi_pose.tools.build_cache --workers 8 --rebuild
python -m notifi_pose.tools.build_site_baseline
```

Reproduce the historical GraphFormer baseline only:

```powershell
python -m notifi_pose.tools.train --exp single_split --arch graphformer `
  --decoder hybrid --hidden 128 --temporal-layers 3 --heads 4 `
  --graph-blocks 2 --epochs 50 --patience 10 --batch-size 16 `
  --lr 0.0005 --lambda-velocity 0.1 --lambda-motion 0.05 `
  --motion-weight 3.0 --baseline sub --link-dropout 0.25 `
  --tag graphformer_hybrid_dynamic_v1
```

Run the historical LOSO-subsampled protocol used by `feature/goal1/work_v2/splits`:

```powershell
$folds = "test_ajh", "test_lmh", "test_mhw"
foreach ($fold in $folds) {
  python -m notifi_pose.tools.train --exp loso --fold $fold `
    --arch graphformer --decoder hybrid --hidden 128 `
    --temporal-layers 3 --heads 4 --graph-blocks 2 `
    --epochs 50 --patience 10 --batch-size 16 --lr 0.0005 `
    --lambda-velocity 0.1 --lambda-motion 0.05 --motion-weight 3.0 `
    --baseline sub --link-dropout 0.25 `
    --tag "loso_graphformer_hybrid_$fold"
}
```

Each fold trains and validates on the other two source subjects and evaluates only
the fixed test portion, about 17.1%, of the held-out source subject. This is not full
LOSO and must be renamed before new experiments. `yja/E02` has already been inspected
repeatedly and is an `unseen_dev_test`, not a sealed final holdout.

Reproduce the historical CSI-only yja E02 evaluation with the validation-selected
5-frame offline filter:

```powershell
python -m notifi_pose.tools.evaluate_sealed `
  work_v2/runs/graphformer_hybrid_dynamic_v1/best_model.pt `
  --dataset sealed --smooth-window 5
```

See `docs/graphformer_gvhmr_v2_experiment.md` for the data contract, ablation, LOSO
results, and known limitations.

## Historical domain-robust protocol

The final robust experiment keeps the GraphFormer hybrid pose backbone and adds RF
augmentation, cross-domain balanced batches, GroupDRO, domain-adversarial and
supervised-contrastive heads, fall phase/contact supervision, dynamics losses, and
validation-selected parameter averaging.

```powershell
python -m notifi_pose.tools.run_robust_protocols --only yja_e02
python -m notifi_pose.tools.run_robust_protocols --only loso
python -m notifi_pose.tools.summarize_robust_runs
```

See `docs/robust_graphformer_experiment.md` and
`work_v2/reports/robust_protocol_results.json` for exact split counts, metrics,
comparison with the previous GraphFormer, and remaining temporal-coherence limits.

> [!WARNING]
> 아래 `1. 기능 요약`부터의 13-point/MediaPipe 수집·학습 절차는 초기 legacy 문서다.
> 현재 GVHMR SMPL-22 V10 데이터/모델 경로에 사용하지 않는다. 현재 실행 기준은 이 README 최상단과
> [`docs/final_code_audit_and_v10_execution_plan.md`](docs/final_code_audit_and_v10_execution_plan.md)다.

## 1. 기능 요약

카메라 기반 낙상 감지는 노인의 얼굴, 신체, 생활 공간이 노출되어 사생활 침해 우려가 큽니다.
단순 safe/alert 분류만으로는 왜 위험하다고 판단했는지 보호자와 사용자가 이해하기 어렵습니다.
낙상, 불안정 보행, 무활동은 순간적인 결과보다 몸의 움직임 흐름을 함께 봐야 정확한 해석이 가능합니다.
CSI-to-Pose는 WiFi CSI 신호만으로 사람의 자세와 움직임을 3D skeleton proxy로 복원하는 실험 기능입니다.
학습 단계에서는 CSI와 동기화된 영상에서 pose teacher GT를 추출하고, 실제 사용 단계에서는 CSI만 입력받습니다.
예측 대상은 머리, 어깨, 팔꿈치, 손목, 골반, 무릎, 발목으로 구성된 13-point skeleton입니다.
시간에 따른 skeleton sequence를 통해 자세 변화, 낙상 시작/종료 구간, 움직임 후 정지 여부를 추정합니다.
첫 설정 단계에서 body template 또는 SMPL-X 기반 체형 정보를 선택적으로 생성해 개인 체형 차이를 보정합니다.
CSI-to-Pose는 safe/alert 모델의 판단 결과에 “어떤 움직임 때문에 위험했는지”를 설명하는 보조 기능으로 활용됩니다.
이를 통해 카메라 없이도 낙상/보행/무활동 흐름을 해석하여 개인정보 보호와 위험 감지 설명 가능성을 동시에 높입니다.

## 2. 폴더 구성

```text
NotiFi-CSI-to-Pose/
  README.md
  requirements.txt
  scripts/
    save_csi_raw.py
    collect_csi_video.py
    preview_pose_camera.py
    extract_pose_features.py
    build_body_shape_template.py
    train_csi_to_pose.py
    render_pose_comparison.py
  docs/
    feature_spec_and_dataset_manual_2026-06-30.md
  assets/
    csi-to-pose-overview.png
    unstable-walking-csi-to-pose-human-readable.gif
```

생성 데이터는 기본적으로 아래에 저장됩니다.

```text
csi_to_pose/
  data/
  videos/
  pose_gt/
  pose_overlays/
  pose_plots/
  body_templates/
  models/
  collection_logs/
```

`csi_to_pose/` 아래 산출물은 용량과 개인정보 이슈가 있어 git에 올리지 않습니다.

## 3. 환경 준비

각자 로컬에서 레포를 받은 뒤 `CSI-to-Pose` 폴더로 이동합니다.

```bash
cd PATH_TO_REPO/NotiFi-CSI-to-Pose
python -m pip install -r requirements.txt
```

macOS에서 matplotlib 캐시 권한 문제가 있으면 아래처럼 실행합니다.

```bash
export MPLCONFIGDIR=/private/tmp/mplconfig
```

Windows PowerShell 예시:

```powershell
cd PATH_TO_REPO\NotiFi-CSI-to-Pose
python -m pip install -r requirements.txt
```

## 4. 포트와 카메라 확인

### macOS 포트 확인

```bash
find /dev -maxdepth 1 \( -name 'cu.usbmodem*' -o -name 'cu.usbserial*' \) -print
```

예:

```text
/dev/cu.usbmodem101
```

### Windows 포트 확인

장치 관리자에서 `USB Serial Device (COMx)`를 확인하거나 PowerShell에서 확인합니다.

```powershell
[System.IO.Ports.SerialPort]::GetPortNames()
```

예:

```text
COM4
```

### 카메라 확인

먼저 skeleton이 잘 잡히는지 확인합니다.

```bash
python scripts/preview_pose_camera.py --camera 0
```

카메라가 여러 개면 `--camera 1`, `--camera 2`를 시도합니다.
창이 뜨면 전신이 최대한 보이게 서고, `POSE DETECTED`가 안정적으로 나오는지 확인합니다.
종료는 preview 창에서 `q`를 누릅니다.

## 5. 수집 전 기기 세팅

| 항목 | 기준 |
| --- | --- |
| 보드 | Seeed Studio XIAO ESP32-C6 기반 sender 3개(TX1/TX2/TX3), receiver 1개 |
| 펌웨어 | ESP-CSI `csi_send`, `csi_recv` |
| 송수신기 거리 | 각 TX-RX 경로 1.2m-2.0m, 좌표를 installation manifest에 기록 |
| 허용 거리 | 1.2m-2.0m, 단 한 세션 중 위치 고정 |
| 보드 높이 | 70-100cm, 4대 모두 높이 기록 |
| 안테나 방향 | TX1/TX2/TX3/RX 방향을 고정하고 기록 |
| 사람 위치 | 세 TX-RX 경로가 행동 영역을 서로 다른 방향에서 통과하도록 배치 |
| 카메라 위치 | 정면 또는 대각 정면 |
| 영상 범위 | 머리부터 발목까지 최대한 포함 |
| 조명 | MediaPipe가 관절을 잡을 수 있을 정도로 밝게 |

권장 배치:

```text
TX1 ---------\
TX2 ---------- 행동 영역 ---------- RX
TX3 ---------/
```

## 6. CSI + Video 동시 수집

기본 명령 형식:

```bash
python scripts/collect_csi_video.py \
  --port "COM_PORT" \
  --subject "SUBJECT" \
  --label "LABEL" \
  --trial t001 \
  --duration DURATION_SECONDS \
  --repeat REPEAT_COUNT \
  --ambient quiet \
  --delay 3 \
  --break_sec 1.5 \
  --camera 0 \
  --preview_pose \
  --note "csi_to_pose_collection"
```

macOS 예시:

```bash
python scripts/collect_csi_video.py \
  --port "/dev/cu.usbmodem101" \
  --subject "yja" \
  --label unstable_walking \
  --trial t001 \
  --duration 20 \
  --repeat 25 \
  --ambient quiet \
  --delay 3 \
  --break_sec 1.5 \
  --camera 0 \
  --preview_pose \
  --note "camera_front, 1.5m, unstable_walking"
```

Windows PowerShell 예시:

```powershell
python scripts\collect_csi_video.py `
  --port "COM4" `
  --subject "mhw" `
  --label unstable_walking `
  --trial t001 `
  --duration 20 `
  --repeat 25 `
  --ambient quiet `
  --delay 3 `
  --break_sec 1.5 `
  --camera 0 `
  --preview_pose `
  --no_sound `
  --note "camera_front, 1.5m, unstable_walking"
```

수집 중에는 시작 직후 손을 크게 들거나 박수 1회를 해서 CSI와 video sync 지점을 남깁니다.

## 7. Pose GT 추출

수집이 끝나면 각 MP4에서 13-point skeleton proxy와 derived feature를 추출합니다.

```bash
python scripts/extract_pose_features.py \
  --video csi_to_pose/videos/warning/gait/unstable_walking/SUBJECT/SUBJECT_unstable_walking_t001.mp4 \
  --label unstable_walking \
  --subject SUBJECT \
  --trial t001
```

예:

```bash
python scripts/extract_pose_features.py \
  --video csi_to_pose/videos/warning/gait/unstable_walking/yja/yja_unstable_walking_t001.mp4 \
  --label unstable_walking \
  --subject yja \
  --trial t001
```

생성 파일:

```text
csi_to_pose/pose_gt/full_33_landmarks/{label}/{subject}/...
csi_to_pose/pose_gt/proxy_13/{label}/{subject}/...
csi_to_pose/pose_overlays/{label}/{subject}/...
csi_to_pose/pose_plots/{label}/{subject}/...
```

성공 기준:

```text
pose detected rate >= 90%
head / shoulder / hip / knee가 안정적으로 잡힘
overlay 영상에서 skeleton이 심하게 튀지 않음
```

## 8. Body Template 생성

여러 trial의 proxy13 결과를 기반으로 사람 형태가 읽히는 body shape proxy를 만들 수 있습니다.

```bash
python scripts/build_body_shape_template.py \
  --label unstable_walking \
  --subject SUBJECT \
  --trials t001 t002 t003 t004 t005 t006 t007 t008 t009 t010
```

출력:

```text
csi_to_pose/body_templates/{subject}/{subject}_body_shape_template.json
```

이 파일은 `render_pose_comparison.py`에서 skeleton 아래에 사람 형태 proxy를 덧그릴 때 사용됩니다.

## 9. Pilot 학습 및 복원 확인

`train_csi_to_pose.py`는 수집된 CSI와 proxy13 GT로 작은 TCN pilot 모델을 학습합니다.
먼저 하나의 label에서 복원이 되는지 확인하는 용도입니다.

```bash
python scripts/train_csi_to_pose.py \
  --label unstable_walking \
  --subject SUBJECT \
  --trials t001 t002 t003 t004 t005 t006 t007 t008 t009 t010 \
  --train_trials t001 t002 t003 t004 t005 t006 t007 t008 \
  --val_trials t009 \
  --test_trials t010 \
  --epochs 80 \
  --log_every 10
```

예측 결과를 사람이 보기 좋게 렌더링합니다.

```bash
python scripts/render_pose_comparison.py \
  --prediction csi_to_pose/models/unstable_walking/SUBJECT/predictions/SUBJECT_unstable_walking_t010_prediction.csv
```

출력:

```text
csi_to_pose/models/{label}/{subject}/human_readable/
```

## 10. 수집 라벨 및 개수

라벨당 10회는 복원 성능 확인에 부족했습니다.
시간이 빠듯한 v1 기준으로 아래처럼 수집합니다.

### Priority A: 필수 수집

총 275 trials.

| label | 초/trial | 횟수 | 목적 |
| --- | ---: | ---: | --- |
| `standing_still` | 20 | 25 | 서 있는 기본 skeleton |
| `sitting_still` | 20 | 25 | 앉은 자세 skeleton |
| `lying_still` | 20 | 25 | 누운 자세 skeleton |
| `walking` | 20 | 25 | 정상 보행 |
| `unstable_walking` | 20 | 25 | 불안정 보행 |
| `sit_to_stand` | 10 | 25 | 앉기에서 서기 |
| `stand_to_sit` | 10 | 25 | 서기에서 앉기 |
| `lie_to_stand` | 10 | 25 | 누운 상태에서 일어나기 |
| `stand_to_lie_normal` | 10 | 25 | 선 상태에서 눕기 |
| `bed_exit_failed` | 10 | 25 | 침대에서 일어나려다 실패 |
| `post_fall_inactive` | 20 | 25 | 낙상 후 무활동 |

### Priority B: 성능 개선용

총 105 trials.

| label | 초/trial | 횟수 | 목적 |
| --- | ---: | ---: | --- |
| `hand_move` | 20 | 15 | 손/팔 움직임 |
| `bed_sitting_to_stand_fall` | 10 | 15 | 침대 앉은 상태에서 일어서다 낙상 |
| `bed_lying_to_stand_fall` | 10 | 15 | 침대 누운 상태에서 일어나려다 낙상 |
| `chair_sitting_to_stand_fall` | 10 | 15 | 의자에서 일어서다 낙상 |
| `chair_stand_to_sit_fall` | 10 | 15 | 의자에 앉으려다 낙상 |
| `lying_convulsive_like_movement` | 10 | 15 | 경련 의심 움직임 |
| `normal_breathing_visible` | 20 | 15 | 정상 호흡 참고 |

### Priority C: 시간이 남을 때

| label | 초/trial | 횟수 | 목적 |
| --- | ---: | ---: | --- |
| `walking_trip_fall` | 10 | 10 | 보행 중 발 걸림 낙상 |
| `walking_turn_fall` | 10 | 10 | 방향 전환 중 낙상 |
| `lying_fast_breath` | 20 | 10 | 빠른 호흡 |
| `lying_slow_breath` | 20 | 10 | 느린 호흡 |
| `lying_irregular_breath` | 20 | 10 | 불규칙 호흡 |

## 11. 품질 확인 체크리스트

각 trial 이후 확인합니다.

```text
CSI_DATA frame이 0개가 아닌가?
MP4 영상이 정상 재생되는가?
사람 전신 또는 최소 머리/어깨/골반/무릎이 보이는가?
pose detected rate가 90% 이상인가?
overlay에서 skeleton이 심하게 튀거나 뒤집히지 않는가?
label과 다른 행동이 섞이지 않았는가?
```

실패한 trial은 삭제하고 같은 trial 번호로 다시 수집합니다.

## 12. 안전 수칙

낙상 라벨은 실제로 세게 넘어지지 않습니다.

```text
매트, 침대, 이불 등 충격 완화 공간에서만 수행
빠르게 쓰러지지 말고 천천히 무너지는 방식으로 수집
혼자 수집할 경우 보행 낙상은 Priority C로 미룸
어지러움, 통증, 호흡 불편이 있으면 즉시 중단
```

## 13. 팀원별 기본 실행 예시

각자 아래 값만 바꿔서 사용합니다.

```text
SUBJECT: 본인 이니셜 또는 짧은 이름
PORT: 본인 receiver 포트
CAMERA: 본인 노트북 카메라 index
LABEL: 현재 수집 라벨
```

macOS:

```bash
export SUBJECT="yja"
export PORT="/dev/cu.usbmodem101"
export CAMERA="0"

python scripts/collect_csi_video.py \
  --port "$PORT" \
  --subject "$SUBJECT" \
  --label standing_still \
  --trial t001 \
  --duration 20 \
  --repeat 25 \
  --ambient quiet \
  --delay 3 \
  --break_sec 1.5 \
  --camera "$CAMERA" \
  --preview_pose \
  --note "csi_to_pose_v1, 1.5m, camera_front"
```

Windows PowerShell:

```powershell
$SUBJECT="mhw"
$PORT="COM4"
$CAMERA="0"

python scripts\collect_csi_video.py `
  --port "$PORT" `
  --subject "$SUBJECT" `
  --label standing_still `
  --trial t001 `
  --duration 20 `
  --repeat 25 `
  --ambient quiet `
  --delay 3 `
  --break_sec 1.5 `
  --camera "$CAMERA" `
  --preview_pose `
  --no_sound `
  --note "csi_to_pose_v1, 1.5m, camera_front"
```

## 14. 개인정보 원칙

CSI-to-Pose는 영상 복원 기능이 아닙니다.
영상은 학습용 pose GT를 만들기 위한 도구이며, 실제 배포 단계에서는 사용하지 않습니다.

```text
Training only: CSI + Video
Deployment: CSI only
```

팀 공유 시에는 원본 영상보다 `proxy13.csv`, `overlay.mp4`, `derived_features.png`, 학습 결과를 우선 공유합니다.
