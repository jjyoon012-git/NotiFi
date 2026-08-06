"""
few_shot_calib.py

lmh로 학습한 root-relative 모델을, 배포 대상(mhw)의 소량 데이터로 fine-tune(캘리브레이션)한다.
"학습 subject는 안 늘리고, 대상자에게 기준 동작 몇 개만 받아 적응" 시나리오.

  - 베이스 모델 : experiments/cross_trainlmh_testmhw_rootrel/ALL/best_model.pt (lmh 학습)
  - 캘리브레이션: mhw train split에서 라벨당 N개  (test와 안 겹침 → 누수 없음)
  - fine-tune val: mhw val split (early stop용, test 아님)
  - 평가        : mhw test split (t007/t013/t019/t025) → 캘리브레이션 전 0.128과 비교

실행:
    python scripts/few_shot_calib.py            # 라벨당 1개
"""

import argparse
import copy
import math
import random
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import train_csi_to_pose as T


def build_seqs(rows, predict_root=False, vis_mask=False, soft_root=False, motion_weight=0.0, vel_spec=False,
               state_refs_map=None, phase=False):
    seqs = []
    for r in rows:
        seq = T.load_sequence(T.ROOT / r["csi_path"], T.ROOT / r["proxy13_path"], r["trial"],
                              vel_spec=vel_spec, phase=phase)
        seq["y"], seq["root"] = T.to_root_relative(seq["y"])   # 베이스와 동일하게 root-relative
        if predict_root:
            seq["y"] = np.concatenate([seq["y"], seq["root"]], axis=1)   # 42차원 (base ckpt와 정합)
        if vis_mask and r.get("full33_path"):
            T.attach_vis_weights(seq, T.ROOT / r["full33_path"], predict_root=predict_root,
                                 soft_root=soft_root)
        if motion_weight > 0:
            T.attach_motion_weights(seq, motion_weight)
        if state_refs_map is not None:
            T.attach_state_labels(seq, state_refs_map[r["subject"]])
        seq["label"] = r["label"]
        seq["subject"] = r["subject"]
        seqs.append(seq)
    return seqs


def pick_calib(train_rows, shots, calib_labels=None):
    """라벨당 앞에서 shots개 (trial 정렬 기준, 결정적).
    calib_labels가 주어지면 그 라벨만 캘리브레이션에 사용.
    (배포 시 위험 동작(낙상 등)은 당사자에게 시킬 수 없으므로 안전 클래스만 수집하는 시나리오.)
    평가는 항상 전체 11개 라벨로 한다."""
    bylab = defaultdict(list)
    for r in train_rows:
        if calib_labels and r["label"] not in calib_labels:
            continue
        bylab[r["label"]].append(r)
    picked = []
    for lab in sorted(bylab):
        rows = sorted(bylab[lab], key=lambda r: r["trial"])
        picked += rows[:shots]
    return picked


def per_label_mae(model, ckpt, device, seqs):
    out = defaultdict(list)
    for seq in seqs:
        pred = T.predict_sequence(model, ckpt, device, seq)
        mae, _, _ = T.evaluate(seq["y"], pred)
        out[seq["label"]].append(mae)
    return {k: float(np.mean(v)) for k, v in out.items()}


def make_model(ckpt, in_dim, out_dim, device):
    dil = tuple(ckpt.get("dilations", (1, 2, 4)))
    sc = 3 if ckpt.get("state_head", False) else 0
    model = T.build_model(ckpt.get("arch", "tcn"), in_dim, out_dim,
                          hidden=ckpt.get("hidden", 96), dilations=dil, state_classes=sc).to(device)
    model.load_state_dict(ckpt["model"])
    return model


def finetune(base_ckpt, calib_seqs, val_seqs, args, device):
    stats = (base_ckpt["x_mean"], base_ckpt["x_std"], base_ckpt["y_mean"], base_ckpt["y_std"])
    train_ds = T.WindowDataset(calib_seqs, args.window, args.step, *stats)
    val_ds = T.WindowDataset(val_seqs, args.window, args.step, *stats)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    in_dim = calib_seqs[0]["x"].shape[1]
    out_dim = calib_seqs[0]["y"].shape[1]
    model = make_model(base_ckpt, in_dim, out_dim, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(reduction="none")
    state_on = bool(base_ckpt.get("state_head", False))
    state_lambda = float(base_ckpt.get("state_lambda", 0.3))
    ce_fn = None
    if state_on:
        cnt = np.zeros(3)
        for sq in calib_seqs:
            m = sq["state_w"] > 0
            for c in range(3):
                cnt[c] += int((sq["state"][m] == c).sum())
        w = torch.tensor(cnt.sum() / (3.0 * np.maximum(cnt, 1)), dtype=torch.float32).to(device)
        ce_fn = nn.CrossEntropyLoss(weight=w, reduction="none")

    def masked_loss(pred, yb, wb):
        l = loss_fn(pred, yb) * wb
        return l.sum() / wb.sum().clamp(min=1.0)

    def state_loss(slog, sb, swb):
        l = ce_fn(slog.reshape(-1, 3), sb.reshape(-1)) * swb.reshape(-1)
        return l.sum() / swb.sum().clamp(min=1.0)

    # base 모델이 bone loss로 학습됐다면 fine-tune에도 동일 제약 유지 (레시피 일관성)
    bone_lambda = float(base_ckpt.get("bone_lambda", 0.0))
    bone_fn = (T.make_bone_loss(base_ckpt["y_mean"], base_ckpt["y_std"], device)
               if bone_lambda > 0 else None)

    best_val = math.inf
    best_state = copy.deepcopy(model.state_dict())
    patience = 0
    stop_ep = args.epochs
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb, wb, sb, swb in train_loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad()
            if state_on:
                pred, slog = model(xb, return_state=True)
                loss = masked_loss(pred, yb, wb) + state_lambda * state_loss(slog, sb.to(device), swb.to(device))
            else:
                pred = model(xb)
                loss = masked_loss(pred, yb, wb)
            if bone_fn is not None:
                loss = loss + bone_lambda * bone_fn(pred, yb, wb)
            loss.backward()
            opt.step()
        model.eval()
        vs = []
        with torch.no_grad():
            for xb, yb, wb, sb, swb in val_loader:
                xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                if state_on:
                    pred, slog = model(xb, return_state=True)
                    vl = masked_loss(pred, yb, wb) + state_lambda * state_loss(slog, sb.to(device), swb.to(device))
                else:
                    pred = model(xb)
                    vl = masked_loss(pred, yb, wb)
                if bone_fn is not None:
                    vl = vl + bone_lambda * bone_fn(pred, yb, wb)
                vs.append(float(vl.item()))
        vl = float(np.mean(vs)) if vs else math.inf
        if vl < best_val - 1e-5:
            best_val = vl
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                stop_ep = epoch
                break
    model.load_state_dict(best_state)
    new_ckpt = dict(base_ckpt)
    new_ckpt["model"] = best_state
    return model, new_ckpt, stop_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/cross_trainlmh_testmhw_rootrel/ALL/best_model.pt")
    ap.add_argument("--source", default="lmh", help="베이스 모델 학습 subject (저장 폴더명용)")
    ap.add_argument("--target", default="mhw")
    ap.add_argument("--shots", type=int, nargs="+", default=[1])
    ap.add_argument("--calib_labels", nargs="+", default=None,
                    help="캘리브레이션에 쓸 라벨만 지정 (예: 안전 클래스만). 평가는 항상 전체 라벨")
    ap.add_argument("--replay_per_label", type=int, default=0,
                    help="catastrophic forgetting 방지: 소스 subject들의 train 데이터를 "
                         "라벨당·사람당 N개 섞어서 fine-tune (0=끔). 위험 클래스 포함 전체 라벨 복습")
    ap.add_argument("--tag", default="", help="저장 폴더명 접미사 (예: safe8)")
    ap.add_argument("--save", action="store_true", help="fine-tune 모델 + 예측 + metrics 저장")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--manifest", default=None,
                    help="사용할 manifest 파일명 (world 캘리브레이션: manifest_full_world.csv)")
    args = ap.parse_args()

    if args.manifest:
        from pathlib import Path as _P
        mp = _P(args.manifest)
        T.MANIFEST_PATH = mp if mp.is_absolute() else T.ROOT / "split_manifest" / args.manifest
        print(f"[manifest] {T.MANIFEST_PATH}")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = T.ROOT / args.ckpt
    base_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # base 체크포인트에서 root-head(42차원)·vis_mask 여부 자동 감지 → base와 동일 구성
    predict_root = base_ckpt["y_mean"].shape[1] > len(T.JOINTS) * 3
    vis_mask = bool(base_ckpt.get("vis_mask", False))
    soft_root = bool(base_ckpt.get("vis_mask_v2", False))
    motion_weight = float(base_ckpt.get("motion_weight", 0.0))
    vel_spec = bool(base_ckpt.get("vel_spec", False))
    phase = bool(base_ckpt.get("phase", False))
    state_on = bool(base_ckpt.get("state_head", False))
    if "window" in base_ckpt and base_ckpt["window"] != args.window:
        print(f"base 모델 window={base_ckpt['window']} → fine-tune window 동일 적용")
        args.window = int(base_ckpt["window"])
    if predict_root:
        print("base 모델이 root-head(42차원) → 캘리브레이션 타깃도 42차원으로 구성")
    if vis_mask:
        print(f"base 모델이 vis-mask 학습 → fine-tune도 동일 마스킹 적용 (soft_root={soft_root})")
    if motion_weight > 0:
        print(f"base 모델이 motion-weight α={motion_weight:g} → fine-tune도 동일 적용")
    if vel_spec:
        print("base 모델이 vel-spec(148차원) → 캘리브레이션도 _v2 캐시 사용")
    refs_map = None
    if state_on:
        print("base 모델이 state-head → 상태 라벨 부착 (subject별 자기 기준)")
        subs = set()
        with open(T.MANIFEST_PATH, newline="", encoding="utf-8") as f:
            import csv as _csv
            subs = {r["subject"] for r in _csv.DictReader(f) if r["valid"] == "True"}
        refs_map = {sub: T.build_state_refs(sub) for sub in sorted(subs)}

    train_rows, val_rows, test_rows = T.load_manifest_rows(label="ALL", subject=args.target)
    print(f"[few-shot calibration]  target={args.target}  base={args.ckpt}")
    print(f"mhw train={len(train_rows)}  val={len(val_rows)}  test={len(test_rows)}")

    if phase:
        print("base 모델이 phase(404차원) → 캘리브레이션도 _v3 캐시 사용")
    val_seqs = build_seqs(val_rows, predict_root, vis_mask, soft_root, motion_weight, vel_spec, refs_map, phase)
    test_seqs = build_seqs(test_rows, predict_root, vis_mask, soft_root, motion_weight, vel_spec, refs_map, phase)

    in_dim = test_seqs[0]["x"].shape[1]; out_dim = test_seqs[0]["y"].shape[1]
    base_model = make_model(base_ckpt, in_dim, out_dim, device)
    base_mae = per_label_mae(base_model, base_ckpt, device, test_seqs)

    if args.calib_labels:
        print(f"캘리브레이션 라벨 제한: {args.calib_labels}")
        # 배포 시나리오 정합성: early-stop용 val도 안전 클래스만 (위험 동작 GT는 존재할 수 없음)
        val_seqs = [s for s in val_seqs if s["label"] in args.calib_labels]
        print(f"val도 동일 제한 → {len(val_seqs)}개")

    # replay: 소스 subject(=target 제외 전원)의 train 데이터에서 라벨당·사람당 N개
    replay_seqs = []
    if args.replay_per_label > 0:
        src_train, _, _ = T.load_manifest_rows(label="ALL", test_subject=args.target)
        bykey = defaultdict(list)
        for r in src_train:
            bykey[(r["subject"], r["label"])].append(r)
        replay_rows = []
        for key in sorted(bykey):
            rows = sorted(bykey[key], key=lambda r: r["trial"])
            replay_rows += rows[:args.replay_per_label]
        replay_seqs = build_seqs(replay_rows, predict_root, vis_mask, soft_root, motion_weight, vel_spec, refs_map, phase)
        print(f"replay: 소스 {sorted({r['subject'] for r in replay_rows})} "
              f"라벨당 {args.replay_per_label}개 → {len(replay_seqs)}개 trial (전체 11라벨 포함)")

    results = {"baseline(0-shot)": base_mae}
    for shots in args.shots:
        calib_rows = pick_calib(train_rows, shots, args.calib_labels)
        calib_seqs = build_seqs(calib_rows, predict_root, vis_mask, soft_root, motion_weight, vel_spec, refs_map, phase)
        print(f"  캘리브레이션 trial ({shots}/label):", sorted({r['label'] for r in calib_rows}))

        ft_seqs = calib_seqs
        if replay_seqs:
            # 윈도우 수 균형: replay가 calib를 묻지 않게 calib를 반복해 비율 ~1:1로
            n_cal = sum(len(s["x"]) for s in calib_seqs)
            n_rep = sum(len(s["x"]) for s in replay_seqs)
            k = max(1, round(n_rep / max(n_cal, 1)))
            ft_seqs = calib_seqs * k + replay_seqs
            print(f"  fine-tune 구성: calib {len(calib_seqs)}개 x{k}반복 + replay {len(replay_seqs)}개")

        model, new_ckpt, ep = finetune(base_ckpt, ft_seqs, val_seqs, args, device)
        mae = per_label_mae(model, new_ckpt, device, test_seqs)
        results[f"{shots}-shot"] = mae
        print(f"  [{shots}-shot] calib trials={len(calib_rows)}  stopped ep={ep}  "
              f"test MAE={np.mean(list(mae.values())):.4f}")

        if args.save:
            suffix = f"_{args.tag}" if args.tag else ""
            out_dir = T.ROOT / "experiments" / f"cross_train{args.source}_test{args.target}_rootrel_calib{shots}{suffix}" / "ALL"
            (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
            torch.save(new_ckpt, out_dir / "best_model.pt")
            summary_rows = []
            for seq in test_seqs:
                pred = T.predict_sequence(model, new_ckpt, device, seq)
                m, per_joint, extra = T.evaluate(seq["y"], pred)
                summary_rows.append({
                    "split": "test", "subject": seq["subject"], "label": seq["label"],
                    "trial": seq["trial"], "mae_all": m, **extra,
                    **{f"mae_{k}": v for k, v in per_joint.items()},
                })
                pred_csv = out_dir / "predictions" / f"{seq['subject']}_{seq['label']}_{seq['trial']}_prediction.csv"
                T.save_prediction_csv(seq, pred, pred_csv)
            import pandas as pd
            pd.DataFrame(summary_rows).to_csv(out_dir / "metrics_summary.csv", index=False)
            print(f"  [저장] {out_dir}")

    if state_on:
        import numpy as _np
        print("\n=== 캘리브레이션 후 상태 분류 (test) ===")
        accs, lrs = {}, {}
        for seq in test_seqs:
            pred, st = T.predict_sequence(model, new_ckpt, device, seq, return_state=True)
            m = seq["state_w"] > 0
            if m.any():
                accs.setdefault(seq["label"], []).append(float((st[m] == seq["state"][m]).mean()))
            lm = m & (seq["state"] == T.STATE_NAMES.index("lying"))
            if lm.any():
                lrs.setdefault(seq["label"], []).append(float((st[lm] == T.STATE_NAMES.index("lying")).mean()))
        for lab in sorted(accs):
            lr = f"  lying_recall={_np.mean(lrs[lab]):.2f}" if lab in lrs else ""
            print(f"  {lab:22s} state_acc={_np.mean(accs[lab]):.2f}{lr}")

    import pandas as pd
    df = pd.DataFrame(results).sort_values("baseline(0-shot)", ascending=False)
    print("\n=== test per-label MAE: 캘리브레이션 전 vs 후 ===")
    print(df.round(4).to_string())
    print("\n전체평균:")
    for k in results:
        print(f"  {k:18s} {np.mean(list(results[k].values())):.4f}")


if __name__ == "__main__":
    main()
