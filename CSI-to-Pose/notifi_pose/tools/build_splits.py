"""split 생성 — 사이트 고정 분할 + LOSO 3-fold + yja 봉인.

설계(확정):

  개발셋 = ajh + mhw + lmh, 9개 사이트(subject x environment)
    각 사이트를 70/15/15 로 **한 번만** 나누고 그 역할을 영구 고정한다.
    → 모든 trial 이 영구적인 역할(train/val/test)을 갖는다.
    → test 몫은 어떤 설정에서도 학습에 들어가지 않는다.
    → 같은 test trial 을 '그 사람을 본 모델'과 '안 본 모델'로 각각 평가할 수 있어
      일반화 갭이 순수하게 측정된다. (전역 랜덤 분할로는 불가능한 비교)

  실험 1 — 단일 split (in-domain)
    학습: 세 사람의 train + val
    평가: test (사람별·환경별 분해 가능)

  실험 2 — LOSO 3-fold (unseen subject)
    fold(S): 나머지 두 사람의 train + val 로 학습 → S 의 **test 몫만** 평가.
    held-out 인 S 의 train/val 은 쓰지 않는다. 실험 1과 동일한 trial 로 평가되므로
    두 숫자가 직접 비교된다.

  봉인
    yja E02  최종 unseen subject (3링크 온전)
    yja E01  제외 — 진폭 6 수준에 프레임간 상관 0.52. 잡음이지 신호가 아니다.
    yja E03  제외 — 3링크 전부 값이 0.

분할 규칙:
  - 사이트 안에서 **클래스별로** 나눈다(계층화). 17클래스가 전 split 에 들어간다.
  - trial 번호 순서대로 **연속 블록**으로 자른다: [train ... | val | test]
    랜덤하게 흩뿌리면 t009/t010 처럼 수초 간격으로 찍힌 인접 trial 이 train 과 test 로
    갈라져 사실상 같은 데이터를 시험하게 된다.
    val 을 train 과 test 사이에 두어 완충 구간으로 삼는다.

실행:
    python -m notifi_pose.tools.build_splits
"""

from __future__ import annotations

import argparse
import json
import re

import pandas as pd

from .. import contract as C
from .. import linkqc
from ..dataio import index as idx

DEV_SUBJECTS = ("ajh", "mhw", "lmh")
SEALED_SUBJECT = "yja"
SEALED_ENV = "E02"
SEALED_EXCLUDED = {
    "E01": "진폭 6 수준 + 프레임간 상관 0.52 = 채널이 아니라 잡음",
    "E03": "3링크 전부 값이 0",
}

#: 개발셋에서 제외하는 사이트. 제외 사유는 CSI 가 아니라 **GT** 다.
#:
#: 촬영 영상이 90도 돌아간 채로 GVHMR 을 돌려서 world 좌표계가 깨졌다. 서 있는
#: 사람이 누운 자세로 복원된다(직립 동작에서 몸통이 수직에서 벗어난 각: 정상
#: 사이트 2.4~4.4도 vs lmh E02 64.8도). 전 시나리오 폴더를 육안 전수 확인했다.
#:
#:   lmh E02  12/17 시나리오 파손 (D05 S01 S02 S03 S05 S06 S07 S08 S09 W01 W02 W03)
#:   lmh E03   6/17 시나리오 파손 (D02 D03 D04 D05 W01 W02)
#:
#: 시나리오 단위로 걸러내면 절반 이상을 살릴 수 있지만(파손은 pose trial 의 11.2%),
#: 재촬영/재생성 데이터가 들어올 예정이라 사이트 단위로 통째 제외한다. 부분적으로
#: 살린 사이트는 클래스 분포가 심하게 치우쳐(E02 는 17종 중 5종만 남는다) 사이트별
#: 성능 비교를 왜곡한다.
#:
#: **CSI 와 class/risk 라벨은 멀쩡하다.** 나중에 분류 전용으로 되살릴 수 있다.
# lmh E02/E03의 회전 영상과 GVHMR GT는 2026-08-02에 재생성되어 복구됐다.
# 이 목록을 남겨두면 수정된 550 trial이 조용히 학습에서 빠진다.
EXCLUDED_SITES: dict[tuple[str, str], str] = {}

#: 제외 trial 에 붙이는 role. dev_index.csv 에 남겨두어 캐시 행 순서를 보존하고,
#: experiments.json 의 roles 목록에 없으므로 학습·평가에서 자동으로 빠진다.
ROLE_EXCLUDED = "excluded"


def _trial_number(trial_id: str) -> int:
    """`lmh_E03_S05_t012` → 12. 연속 블록 분할의 정렬 키."""
    m = re.search(r"_t(\d+)$", trial_id)
    return int(m.group(1)) if m else 0


def assign_roles(df: pd.DataFrame, val_ratio: float, test_ratio: float) -> pd.Series:
    """사이트 x 클래스 안에서 연속 블록으로 train/val/test 역할을 배정한다.

    trial 번호 오름차순으로 [train ... | val | test] 순서.
    val 이 train 과 test 사이에 놓여 인접 trial 누수의 완충 역할을 한다.
    """
    role = pd.Series(index=df.index, dtype=object)
    for _, g in df.groupby(["subject", "environment", "class_id"], sort=False):
        order = (g.assign(_n=g.trial_id.map(_trial_number))
                  .sort_values(["_n", "trial_id"]).index)
        n = len(order)
        if n == 1:
            role.loc[order] = "train"
            continue
        if n == 2:
            role.loc[order[:1]] = "train"
            role.loc[order[1:]] = "test"
            continue
        n_test = max(1, int(n * test_ratio + 0.5))
        n_val = max(1, int(n * val_ratio + 0.5))
        if n_test + n_val >= n:                      # 아주 작은 클래스 보호
            n_test, n_val = 1, 1
        n_train = n - n_val - n_test
        role.loc[order[:n_train]] = "train"
        role.loc[order[n_train:n_train + n_val]] = "val"
        role.loc[order[n_train + n_val:]] = "test"
    return role


def verify(dev: pd.DataFrame, sealed: pd.DataFrame) -> dict:
    """dev 는 제외분(role=excluded)까지 포함한 전체. 검사는 활성분만 대상으로 한다."""
    report: dict = {"checks": []}

    def chk(name, ok, detail):
        report["checks"].append({"name": name, "ok": bool(ok), "detail": detail})

    active = dev[dev.role != ROLE_EXCLUDED]
    n_sites = active.groupby(["subject", "environment"]).ngroups

    ids = {r: set(dev.loc[dev.role == r, "trial_id"]) for r in ("train", "val", "test")}
    chk("role_partition_exclusive",
        not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"])
        and not (ids["val"] & ids["test"]),
        {r: len(v) for r, v in ids.items()})
    chk("role_covers_all", sum(len(v) for v in ids.values()) == len(active),
        {"assigned": sum(len(v) for v in ids.values()), "active": len(active),
         "excluded": int((dev.role == ROLE_EXCLUDED).sum())})
    chk("sealed_disjoint", not (set(sealed.trial_id) & set(dev.trial_id)),
        {"sealed": len(sealed), "dev": len(dev)})
    chk("sealed_subject_absent_from_dev", SEALED_SUBJECT not in set(dev.subject),
        {"dev_subjects": sorted(set(dev.subject))})

    # 제외 사이트가 활성 역할에 하나도 안 남아있는지
    leaked = [f"{s}_{e}" for (s, e) in EXCLUDED_SITES
              if len(active[(active.subject == s) & (active.environment == e)])]
    chk("excluded_sites_absent", not leaked, {"leaked": leaked, "n_active_sites": n_sites})

    for r in ("train", "val", "test"):
        sub = dev[dev.role == r]
        chk(f"{r}_all_sites", sub.groupby(["subject", "environment"]).ngroups == n_sites,
            {"n_sites": int(sub.groupby(["subject", "environment"]).ngroups),
             "expected": int(n_sites)})
        chk(f"{r}_all_classes", sub.class_id.nunique() == C.N_CLASSES,
            {"n_classes": int(sub.class_id.nunique())})

    for s in DEV_SUBJECTS:
        tr = dev[(dev.subject != s) & (dev.role.isin(["train", "val"]))]
        te = dev[(dev.subject == s) & (dev.role == "test")]
        chk(f"loso_{s}_no_leak", s not in set(tr.subject),
            {"train_subjects": sorted(set(tr.subject)), "n_train": len(tr), "n_test": len(te)})
        chk(f"loso_{s}_test_all_classes", te.class_id.nunique() == C.N_CLASSES,
            {"n_classes": int(te.class_id.nunique())})

    # 인접 trial 누수: 같은 (사이트, 클래스) 안에서 train 과 test 가 맞붙지 않았는가
    adjacent = []
    for key, g in dev.groupby(["subject", "environment", "class_id"], sort=False):
        roles = (g.assign(_n=g.trial_id.map(_trial_number))
                  .sort_values("_n").role.tolist())
        if any({a, b} == {"train", "test"} for a, b in zip(roles, roles[1:])):
            adjacent.append(str(key))
    chk("no_train_test_adjacency", len(adjacent) == 0,
        {"n_groups": len(adjacent), "examples": adjacent[:5]})

    report["passed"] = all(c["ok"] for c in report["checks"])
    report["n_failed"] = sum(1 for c in report["checks"] if not c["ok"])
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    args = ap.parse_args()

    ix = idx.load_all_index(resolve=False)
    ix = ix.join(linkqc.alive_links_per_trial(), on="trial_id")
    ix["n_alive"] = ix["n_alive"].fillna(0).astype(int)

    # 살아있는 링크가 하나도 없으면 입력 신호가 없는 것이므로 제외
    dev = ix[ix.subject.isin(DEV_SUBJECTS) & (ix.n_alive > 0)].copy()
    dev["role"] = assign_roles(dev, args.val_ratio, args.test_ratio)

    # 제외 사이트는 dev_index 에 남기되 role 을 바꾼다. 행을 지우면 캐시 배열의
    # 행 번호가 전부 밀려 1.14GB 를 다시 구워야 한다(build_cache --reindex 참조).
    for (s, e) in EXCLUDED_SITES:
        dev.loc[(dev.subject == s) & (dev.environment == e), "role"] = ROLE_EXCLUDED

    sealed = ix[(ix.subject == SEALED_SUBJECT) & (ix.environment == SEALED_ENV)].copy()

    C.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    keep = ["trial_id", "subject", "environment", "task", "class_id", "scenario_id",
            "detail_label", "risk", "risk_id", "gt_schema", "n_alive",
            "time_method", "csi", "gt_pose", "original_video"]
    dev[keep + ["role"]].to_csv(C.SPLIT_DIR / "dev_index.csv", index=False, encoding="utf-8")
    sealed[keep].to_csv(C.SPLIT_DIR / "sealed_index.csv", index=False, encoding="utf-8")

    experiments = {
        "single_split": {
            "description": "in-domain. 세 사람의 train+val 로 학습, test 로 평가",
            "train": {"subjects": list(DEV_SUBJECTS), "roles": ["train"]},
            "val": {"subjects": list(DEV_SUBJECTS), "roles": ["val"]},
            "test": {"subjects": list(DEV_SUBJECTS), "roles": ["test"]},
        },
        "yja_holdout": {
            "description": (
                "Protocol A: use every source trial for train/validation and keep "
                "yja E02 pose trials as the unseen test set"
            ),
            "train": {
                "subjects": list(DEV_SUBJECTS),
                "roles": ["train", "test"],
            },
            "val": {"subjects": list(DEV_SUBJECTS), "roles": ["val"]},
            "test": {
                "subject": SEALED_SUBJECT,
                "environment": SEALED_ENV,
                "task": C.TASK_POSE,
            },
        },
        "loso": {
            f"test_{s}": {
                "description": f"{s} 미학습. 나머지 둘의 train+val 로 학습, "
                               f"{s} 의 test 몫만 평가(단일 split 과 동일 trial)",
                "train": {"subjects": [x for x in DEV_SUBJECTS if x != s], "roles": ["train"]},
                "val": {"subjects": [x for x in DEV_SUBJECTS if x != s], "roles": ["val"]},
                "test": {"subjects": [s], "roles": ["test"]},
            } for s in DEV_SUBJECTS
        },
        "sealed": {
            "yja_E02": {"subject": SEALED_SUBJECT, "environment": SEALED_ENV,
                        "note": "최종 unseen subject. 마일스톤에서만 연다."},
        },
        "excluded": {f"{SEALED_SUBJECT}_{e}": r for e, r in SEALED_EXCLUDED.items()},
        "link_qc": {
            "min_frame_corr": linkqc.MIN_FRAME_CORR,
            "min_pkt_max": linkqc.MIN_PKT_MAX,
            "max_allzero_ratio": linkqc.MAX_ALLZERO_RATIO,
        },
    }
    (C.SPLIT_DIR / "experiments.json").write_text(
        json.dumps(experiments, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------- 출력 ----------------
    n_excl = int((dev.role == ROLE_EXCLUDED).sum())
    print(f"[build_splits] 개발셋 {len(dev) - n_excl:,} (제외 {n_excl:,})  봉인 {len(sealed):,}")

    if EXCLUDED_SITES:
        print("\n=== 제외 사이트 ===")
        for (s, e), why in EXCLUDED_SITES.items():
            n = int(((dev.subject == s) & (dev.environment == e)).sum())
            print(f"  {s} {e}: {n} trial  ({why})")

    print("\n=== 사이트별 역할 배정 ===")
    piv = dev.pivot_table(index=["subject", "environment"], columns="role",
                          values="trial_id", aggfunc="count").fillna(0).astype(int)
    for c in ("train", "val", "test", ROLE_EXCLUDED):
        if c not in piv.columns:
            piv[c] = 0
    piv = piv[["train", "val", "test", ROLE_EXCLUDED]]
    piv["계"] = piv.sum(axis=1)
    print(piv.to_string())
    print(f"  합계: train {piv.train.sum()}  val {piv.val.sum()}  test {piv.test.sum()}"
          f"  (제외 {piv[ROLE_EXCLUDED].sum()})")

    print("\n=== LOSO fold 규모 ===")
    for s in DEV_SUBJECTS:
        tr = dev[(dev.subject != s) & (dev.role == "train")]
        va = dev[(dev.subject != s) & (dev.role == "val")]
        te = dev[(dev.subject == s) & (dev.role == "test")]
        print(f"  test_{s}: train {len(tr)}  val {len(va)}  test {len(te)}  "
              f"{te.groupby('environment').size().to_dict()}")

    print("\n=== test 몫의 클래스별 trial 수 ===")
    for s in DEV_SUBJECTS:
        te = dev[(dev.subject == s) & (dev.role == "test")]
        vc = te.class_id.value_counts()
        dg = te[te.risk == "danger"].class_id.value_counts()
        print(f"  {s}: 클래스당 {vc.min()}~{vc.max()} | "
              f"danger 클래스당 {dg.min()}~{dg.max()} (총 {dg.sum()})")

    print("\n=== 링크 결손 trial (n_alive < 3) 이 어디로 갔나 ===")
    deg = dev[dev.n_alive < 3]
    if len(deg):
        print(deg.pivot_table(index=["subject", "environment"], columns="role",
                              values="trial_id", aggfunc="count")
                 .fillna(0).astype(int).to_string())
        print(f"  총 {len(deg)} (1링크 {int((deg.n_alive==1).sum())}, "
              f"2링크 {int((deg.n_alive==2).sum())})")
    else:
        print("  없음")

    print("\n=== 봉인 / 제외 ===")
    print(f"  [봉인] {SEALED_SUBJECT} {SEALED_ENV}: {len(sealed)}  "
          f"n_alive {sealed.n_alive.value_counts().sort_index().to_dict()}")
    for env, reason in SEALED_EXCLUDED.items():
        n = int(((ix.subject == SEALED_SUBJECT) & (ix.environment == env)).sum())
        print(f"  [제외] {SEALED_SUBJECT} {env}: {n}  ({reason})")
    dropped = int((ix.subject.isin(DEV_SUBJECTS) & (ix.n_alive == 0)).sum())
    print(f"  [제외] 개발셋 중 살아있는 링크 0개: {dropped}")

    print("\n[build_splits] 검증 ...")
    report = verify(dev, sealed)
    (C.REPORT_DIR / "split_verify.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if report["passed"]:
        print(f"[build_splits] OK - {len(report['checks'])} checks passed")
    else:
        print(f"[build_splits] FAILED - {report['n_failed']}/{len(report['checks'])}")
        for c in report["checks"]:
            if not c["ok"]:
                print(f"    - {c['name']}: {c['detail']}")

    print(f"\n  wrote {C.SPLIT_DIR / 'dev_index.csv'}")
    print(f"  wrote {C.SPLIT_DIR / 'sealed_index.csv'}")
    print(f"  wrote {C.SPLIT_DIR / 'experiments.json'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
