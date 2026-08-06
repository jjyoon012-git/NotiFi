"""
render_world_state_video.py

state_head(world) 모델을 3분할 비디오로 시각화 + 예측 자세 라벨 오버레이.
  [ 위: 원본 영상 + "GT: {state}"  "Pred: {state}" 배너 (일치=초록/불일치=빨강) ]
  [ 아래: GT 골격 | 예측 골격 (정면뷰) ]

실행:
    python scripts/render_world_state_video.py --labels standing_still lying_still post_fall_inactive walking sitting_still --trials t007
"""
import argparse, sys
from pathlib import Path
import cv2, numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import train_csi_to_pose as T

CKPT = ROOT / "experiments/loso_test_mhw_rootrel_vismask_vel_rf16_st_world/ALL/best_model.pt"
JOINTS = T.JOINTS
EDGES = [("head","left_shoulder"),("head","right_shoulder"),("left_shoulder","right_shoulder"),
         ("left_shoulder","left_elbow"),("left_elbow","left_wrist"),
         ("right_shoulder","right_elbow"),("right_elbow","right_wrist"),
         ("left_shoulder","left_hip"),("right_shoulder","right_hip"),("left_hip","right_hip"),
         ("left_hip","left_knee"),("left_knee","left_ankle"),
         ("right_hip","right_knee"),("right_knee","right_ankle")]
GT_C, PR_C = (0,200,0),(0,60,230)
STATE = T.STATE_NAMES
LVP = {"standing_still":"safe/posture/standing_still","sitting_still":"safe/posture/sitting_still",
       "lying_still":"safe/posture/lying_still","walking":"safe/motion/walking",
       "sit_to_stand":"safe/transition/sit_to_stand","stand_to_sit":"safe/transition/stand_to_sit",
       "lie_to_stand":"safe/transition/lie_to_stand","stand_to_lie_normal":"safe/transition/stand_to_lie_normal",
       "bed_exit_failed":"warning/bed_exit/bed_exit_failed","unstable_walking":"warning/gait/unstable_walking",
       "post_fall_inactive":"danger/fall/post_fall_inactive"}
PW,PH = 400,540
VW = PW*2; VH = int(VW*9/16); CH = VH+PH


def draw(row, pfx, color):
    c = np.full((PH,PW,3),255,np.uint8); pts={}
    xs=[row[f"{pfx}{j}_x"] for j in JOINTS]; ys=[row[f"{pfx}{j}_y"] for j in JOINTS]
    cx,cy=(min(xs)+max(xs))/2,(min(ys)+max(ys))/2; span=max(max(xs)-min(xs),max(ys)-min(ys),0.5)*1.3
    for j in JOINTS:
        pts[j]=(int((row[f"{pfx}{j}_x"]-cx)/span*PW+PW/2), int((row[f"{pfx}{j}_y"]-cy)/span*PH+PH/2))
    for a,b in EDGES: cv2.line(c,pts[a],pts[b],color,3,cv2.LINE_AA)
    for p in pts.values(): cv2.circle(c,p,5,color,-1,cv2.LINE_AA)
    return c


def render(label, trial, model, ckpt, out_dir):
    rows_all = None
    # seq 구성 (world, vel_spec, root_relative, state 라벨)
    import csv
    with open(T.MANIFEST_PATH, newline="", encoding="utf-8") as f:
        row = next((r for r in csv.DictReader(f)
                    if r["subject"]=="mhw" and r["label"]==label and r["trial"]==trial
                    and r["valid"]=="True"), None)
    if row is None:
        print(f"  [SKIP] manifest 없음 {label} {trial}"); return
    seq = T.load_sequence(ROOT/row["csi_path"], ROOT/row["proxy13_path"], trial, vel_spec=True)
    seq["y"], seq["root"] = T.to_root_relative(seq["y"])
    refs = render.refs
    T.attach_state_labels(seq, refs)
    pose_pred, st_pred = T.predict_sequence(model, ckpt, "cpu", seq, return_state=True)
    gt = seq["y"]; t = seq["time"]; gt_state = seq["state"]

    vdir = ROOT/f"csi_to_pose_mhw/videos/{LVP[label]}/mhw"
    vpath = vdir/f"mhw_{label}_{trial}.avi"; tspath = vdir/f"mhw_{label}_{trial}_frame_timestamps.csv"
    if not vpath.exists(): print(f"  [SKIP] 영상없음 {label}"); return
    import pandas as pd
    fts = pd.read_csv(tspath)["timestamp_sec"].to_numpy()
    idx = np.clip(np.searchsorted(t, fts),0,len(gt)-1)

    out_dir.mkdir(parents=True,exist_ok=True)
    out = out_dir/f"mhw_{label}_{trial}_state.mp4"
    wr = cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*"mp4v"),15,(VW,CH))
    cap = cv2.VideoCapture(str(vpath)); fi=0
    cols = [f"{p}{j}_{ax}" for p in ("gt_","pred_") for j in JOINTS for ax in ("x","y","z")]
    while True:
        ok,frame=cap.read()
        if not ok: break
        k = idx[fi] if fi<len(idx) else len(gt)-1
        rd = {}
        for j in JOINTS:
            for ax,a in zip(("x","y"),(0,1)):
                rd[f"gt_{j}_{ax}"]=gt[k].reshape(len(JOINTS),3)[JOINTS.index(j),a]
                rd[f"pred_{j}_{ax}"]=pose_pred[k].reshape(len(JOINTS),3)[JOINTS.index(j),a]
        canvas=np.full((CH,VW,3),255,np.uint8)
        canvas[0:VH,0:VW]=cv2.resize(frame,(VW,VH))
        gs, ps = STATE[gt_state[k]], STATE[st_pred[k]]
        match = gt_state[k]==st_pred[k]
        cv2.putText(canvas,f"{label} {trial}",(12,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2,cv2.LINE_AA)
        cv2.putText(canvas,f"GT: {gs}",(12,VH-50),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,220,0),3,cv2.LINE_AA)
        cv2.putText(canvas,f"Pred: {ps}",(12,VH-12),cv2.FONT_HERSHEY_SIMPLEX,1.0,
                    (0,200,0) if match else (0,50,240),3,cv2.LINE_AA)
        canvas[VH:VH+PH,0:PW]=draw(rd,"gt_",GT_C)
        canvas[VH:VH+PH,PW:PW*2]=draw(rd,"pred_",PR_C)
        cv2.putText(canvas,"GT skeleton",(12,VH+28),cv2.FONT_HERSHEY_SIMPLEX,0.7,GT_C,2)
        cv2.putText(canvas,"Pred skeleton",(PW+12,VH+28),cv2.FONT_HERSHEY_SIMPLEX,0.7,PR_C,2)
        wr.write(canvas); fi+=1
    cap.release(); wr.release()
    print(f"  [OK] {out.name} ({fi} frames)")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--labels",nargs="+",default=["standing_still","sitting_still","lying_still","walking","post_fall_inactive"])
    ap.add_argument("--trials",nargs="+",default=["t007"])
    args=ap.parse_args()
    T.MANIFEST_PATH = ROOT/"split_manifest"/"manifest_full_world.csv"
    ckpt=torch.load(CKPT,map_location="cpu",weights_only=False)
    model=T.TinyTCN(ckpt["x_mean"].shape[1],ckpt["y_mean"].shape[1],96,
                    dilations=tuple(ckpt["dilations"]),state_classes=3)
    model.load_state_dict(ckpt["model"]); model.eval()
    render.refs = T.build_state_refs("mhw")
    out_dir = CKPT.parent/"state_video"
    for lab in args.labels:
        for tr in args.trials:
            render(lab,tr,model,ckpt,out_dir)
    print(f"\n완료 → {out_dir}")


if __name__=="__main__":
    main()
