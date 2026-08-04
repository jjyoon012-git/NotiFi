"""학습 실행.

    python -m notifi_pose.tools.train                                # 단일 split, TCN
    python -m notifi_pose.tools.train --arch mamba
    python -m notifi_pose.tools.train --exp loso --fold test_lmh
    python -m notifi_pose.tools.train --epochs 5 --tag smoke         # 빠른 점검

결과는 work_v2/runs/<tag>/ 에 저장된다 (best_model.pt, history.csv, result.json).
"""

from __future__ import annotations

import argparse
import json

from .. import contract as C
from ..dataio import dataset as D
from ..trainer import TrainConfig, train


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--exp", default="single_split",
        choices=["single_split", "single_split_lmh_e01", "yja_holdout", "loso"],
    )
    ap.add_argument("--fold", default=None, help="exp=loso 일 때 test_ajh 등")
    ap.add_argument(
        "--arch", default="tcn",
        choices=[
            "tcn", "mamba", "graphformer", "robust_graphformer",
            "impact_graphformer", "latent_flow", "v3",
        ],
    )
    ap.add_argument("--fusion", default="gate", choices=["gate", "concat"],
                    help="3링크 합치는 방식. concat 은 링크 간 차이를 보존한다")
    ap.add_argument("--film", action="store_true",
                    help="LinkEncoder 에서 링크별 곱셈 변조. 링크마다 다른 비선형 특징")
    ap.add_argument("--tag", default=None, help="결과 폴더 이름 (기본: 자동 생성)")

    # 상한을 넉넉히 두고 early stopping 에 맡긴다. 200 으로는 부족했다
    # (concat7_tcn 이 best epoch 199/200 으로 상한에 걸려 끝났다).
    # 주의: LR 스케줄이 CosineAnnealingLR(T_max=epochs) 라 이 값을 바꾸면
    # 학습률 감쇠 속도도 같이 바뀐다. 이전 실행과 완전한 동일 조건 비교는 아니다.
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--temporal-layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--graph-blocks", type=int, default=2)
    ap.add_argument("--decoder", choices=["tree", "hybrid"], default="tree")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--lambda-root", type=float, default=1.0)
    ap.add_argument("--lambda-bone", type=float, default=0.1)
    ap.add_argument("--lambda-cls", type=float, default=0.2)
    ap.add_argument("--lambda-risk", type=float, default=0.1)
    ap.add_argument("--lambda-velocity", type=float, default=0.0)
    ap.add_argument("--lambda-motion", type=float, default=0.0)
    ap.add_argument("--lambda-acceleration", type=float, default=0.0)
    ap.add_argument("--lambda-jerk", type=float, default=0.0)
    ap.add_argument("--lambda-impact", type=float, default=0.0)
    ap.add_argument("--lambda-coarse", type=float, default=0.0)
    ap.add_argument("--lambda-displacement", type=float, default=0.0)
    ap.add_argument("--lambda-flow", type=float, default=0.0)
    ap.add_argument("--lambda-contact", type=float, default=0.0)
    ap.add_argument("--lambda-phase", type=float, default=0.0)
    ap.add_argument("--lambda-foot-slide", type=float, default=0.0)
    ap.add_argument("--lambda-floor", type=float, default=0.0)
    ap.add_argument("--lambda-domain", type=float, default=0.0)
    ap.add_argument("--lambda-supcon", type=float, default=0.0)
    ap.add_argument("--lambda-latent", type=float, default=0.0)
    ap.add_argument("--motion-weight", type=float, default=0.0,
                    help="GT body speed가 큰 프레임의 pose/root 가중치")

    ap.add_argument("--baseline", default="none", choices=["none", "sub", "sub_z"],
                    help="사이트별 빈방 기준선 제거 (Phase 2). sub=빼기, sub_z=빼고 잔차정규화")
    ap.add_argument("--link-dropout", type=float, default=0.25,
                    help="링크를 꺼보는 확률. 0 이면 끔")
    ap.add_argument("--group-dro-eta", type=float, default=0.0)
    ap.add_argument("--balanced-batches", action="store_true")
    ap.add_argument("--rf-augment", action="store_true")
    ap.add_argument("--frequency-tokens", type=int, default=12)
    ap.add_argument("--geometry-path", default=None)
    ap.add_argument("--motion-prior", default=None)
    ap.add_argument(
        "--init-checkpoint", default=None,
        help="warm-start model weights; impact_graphformer accepts robust checkpoints",
    )
    ap.add_argument("--backbone-lr-scale", type=float, default=1.0)
    ap.add_argument("--refiner-warmup-epochs", type=int, default=0)
    ap.add_argument("--flow-steps", type=int, default=4)
    ap.add_argument("--flow-noise", type=float, default=0.25)
    ap.add_argument("--domain-grl", type=float, default=0.2)
    ap.add_argument(
        "--weight-average-start", type=int, default=10,
        help="epoch to start dense checkpoint averaging; <=0 disables it",
    )

    args = ap.parse_args()

    cfg = TrainConfig(
        arch=args.arch, fusion=args.fusion, film=args.film,
        hidden=args.hidden, dilations=tuple(args.dilations),
        n_blocks=args.temporal_layers, heads=args.heads,
        graph_blocks=args.graph_blocks, decoder=args.decoder,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        patience=args.patience, num_workers=args.workers, seed=args.seed,
        lambda_root=args.lambda_root, lambda_bone=args.lambda_bone,
        lambda_cls=args.lambda_cls, lambda_risk=args.lambda_risk,
        lambda_velocity=args.lambda_velocity, lambda_motion=args.lambda_motion,
        lambda_acceleration=args.lambda_acceleration,
        lambda_jerk=args.lambda_jerk, lambda_impact=args.lambda_impact,
        lambda_coarse=args.lambda_coarse,
        lambda_displacement=args.lambda_displacement,
        lambda_flow=args.lambda_flow,
        lambda_contact=args.lambda_contact, lambda_phase=args.lambda_phase,
        lambda_foot_slide=args.lambda_foot_slide, lambda_floor=args.lambda_floor,
        lambda_domain=args.lambda_domain, lambda_supcon=args.lambda_supcon,
        lambda_latent=args.lambda_latent,
        motion_weight=args.motion_weight,
        group_dro_eta=args.group_dro_eta,
        balanced_batches=args.balanced_batches,
        rf_augment=args.rf_augment,
        frequency_tokens=args.frequency_tokens,
        geometry_path=args.geometry_path,
        motion_prior_path=args.motion_prior,
        init_checkpoint=args.init_checkpoint,
        backbone_lr_scale=args.backbone_lr_scale,
        refiner_warmup_epochs=args.refiner_warmup_epochs,
        flow_steps=args.flow_steps,
        flow_noise=args.flow_noise,
        domain_grl=args.domain_grl,
        weight_average_start=args.weight_average_start,
        baseline=args.baseline,
    )

    datasets = D.build_datasets(
        exp=args.exp, fold=args.fold,
        dropout=D.DropoutConfig(
            p=args.link_dropout, rf_augment=args.rf_augment
        ), seed=args.seed,
        baseline=args.baseline)

    tag = args.tag or f"{args.exp}{'_' + args.fold if args.fold else ''}_{args.arch}"
    out_dir = C.WORK_ROOT / "runs" / tag

    print(f"[train] exp={args.exp} fold={args.fold} -> {out_dir}")
    for k, v in datasets.items():
        print(f"  {k:5s} {v.describe()}")

    result = train(datasets, cfg, out_dir)
    (out_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
