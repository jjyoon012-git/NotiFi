import argparse
import time

import cv2
import mediapipe as mp


def main():
    parser = argparse.ArgumentParser(
        description="Preview camera feed with MediaPipe pose overlay."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    window_name = f"NotiFi Pose Preview - camera {args.camera} (press q to quit)"

    print(f"[PREVIEW] camera={args.camera}")
    print("Press q in the preview window to quit.")

    frame_count = 0
    detected_count = 0
    start = time.time()
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame_count += 1
            shown = frame.copy()
            rgb = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if result.pose_landmarks:
                detected_count += 1
                mp_draw.draw_landmarks(
                    shown,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )
                status = "POSE DETECTED"
                color = (0, 220, 0)
            else:
                status = "NO POSE"
                color = (0, 0, 255)

            elapsed = max(time.time() - start, 1e-6)
            cv2.putText(
                shown,
                f"{status} | camera={args.camera} | fps={frame_count / elapsed:.1f}",
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, shown)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    ratio = detected_count / frame_count * 100 if frame_count else 0
    print(f"[DONE] pose detected: {detected_count}/{frame_count} ({ratio:.1f}%)")


if __name__ == "__main__":
    main()
