"""3링크 비동기 CSI 파서.

한 trial 의 `csi.csv` 에는 TX1/TX2/TX3 패킷이 **시간순으로 섞여** 들어 있다.
legacy 파이프라인은 sender_id 를 무시했기 때문에 세 링크의 차이를 사람의 움직임으로
오인할 수 있었다. 이 모듈은 링크를 분리해서 공통 시간축으로 재표본화하고, 각 시점에
실측 패킷이 있었는지를 mask 로 남긴다.

출력 계약:
    times     float32 [T]              trial 상대 시각(초)
    iq        float32 [T, 3, 128, 2]   (I, Q)
    link_mask bool    [T, 3]           해당 시점에 유효 패킷이 있었는가
    meta      dict                     링크별 패킷수/Hz, 드롭 행수, 전처리 버전

설계 근거(표본 실측):
  - 링크별 패킷율이 6.7~63 Hz 로 크게 다르다(중앙 25.7 Hz). "3링크가 다 살아있다"는
    가정을 쓸 수 없어 mask 가 필수다.
  - 시리얼 로그가 잘려 CSV 필드가 밀린 파손 행이 존재한다(mhw 0.76%, ajh 0.29%).
    원본 라인 자체가 truncate 되어 복구 불가 → 드롭한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import contract as C


@dataclass
class CsiTrial:
    times: np.ndarray       # [T] float32
    iq: np.ndarray          # [T, L, S, 2] float32
    link_mask: np.ndarray   # [T, L] bool
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.times.shape[0])


class CsiParseError(RuntimeError):
    pass


# ------------------------------------------------------------------ 파싱


def _payload_token_count(payloads: pd.Series) -> pd.Series:
    """대괄호를 벗긴 payload 의 콤마 구분 토큰 수. 벡터화되어 있어 필터링용으로 싸다."""
    return payloads.str.strip().str.strip("[]").str.count(",") + 1


def _parse_payload_block(payloads: pd.Series) -> np.ndarray:
    """`"[0,0,9,38,...]"` 문자열 시리즈 → [N, 256] float32.

    행마다 ast.literal_eval 을 부르면 trial 당 수백 ms 가 나가므로, 대괄호를 벗기고
    한 번에 이어붙여 단일 파싱으로 처리한다.

    np.fromstring 은 쓰지 않는다 — 숫자가 아닌 토큰을 만나면 예외 없이 그 지점에서
    잘라낸 배열을 돌려주므로, 파손 데이터가 조용히 통과할 수 있다. 명시적 split 은
    잘못된 토큰에서 ValueError 를 낸다.

    호출 전에 토큰 수가 검증되었다고 가정하되, 최종 크기를 다시 확인한다.
    """
    if len(payloads) == 0:
        return np.zeros((0, C.CSI_RAW_LEN), dtype=np.float32)

    joined = ",".join(payloads.str.strip().str.strip("[]").tolist())
    try:
        values = np.array(joined.split(","), dtype=np.float32)
    except ValueError as exc:
        raise CsiParseError(f"payload 에 숫자가 아닌 토큰: {exc}") from exc

    expected = len(payloads) * C.CSI_RAW_LEN
    if values.size != expected:
        raise CsiParseError(
            f"payload 길이 불일치: {values.size} != {expected} "
            f"({len(payloads)} rows x {C.CSI_RAW_LEN})"
        )
    return values.reshape(len(payloads), C.CSI_RAW_LEN)


def read_csi_packets(csi_path: Path) -> tuple[pd.DataFrame, dict]:
    """csi.csv 를 읽어 유효 패킷만 남긴 DataFrame 과 품질 메타를 반환한다.

    모델 입력 화이트리스트(C.CSI_USECOLS) 컬럼만 로드한다. subject_id, session_id,
    risk_label 같은 누수 컬럼은 **읽지도 않는다** — 실수로 피처에 섞이는 경로를
    원천 차단하기 위해서다.
    """
    csi_path = Path(csi_path)
    if not csi_path.exists():
        raise CsiParseError(f"csi.csv not found: {csi_path}")

    df = pd.read_csv(csi_path, usecols=list(C.CSI_USECOLS))
    n_raw = len(df)

    # 파손 유형이 세 가지 관측된다. 셋 다 걸러야 한다.
    #  (1) 시리얼 라인이 잘려 CSV 필드가 밀린 행 → csi_len 이 26, -5 같은 쓰레기 값
    #  (2) 필드 밀림의 2차 징후 → sender_id 가 TX1/TX2/TX3 가 아님
    #  (3) csi_len 은 256 인데 payload 토큰이 230/351 개인 행 (드물지만 존재)
    #      → csi_len 필드만 믿으면 안 되고 payload 를 직접 세야 한다.
    csi_len = pd.to_numeric(df["csi_len"], errors="coerce")
    keep = csi_len == C.CSI_RAW_LEN
    keep &= df["sender_id"].isin(C.LINKS)

    elapsed = pd.to_numeric(df["pc_elapsed_s"], errors="coerce")
    keep &= elapsed.notna()

    #  (4) ESP32 콘솔 로그가 payload 안으로 섞여 들어간 행. 예:
    #      "E (408732) task_wdt: Task watchdog got triggered. ..."
    #      토큰 수는 우연히 맞을 수 있으므로 숫자 이외의 문자가 있는지 따로 본다.
    payload_ok = pd.Series(False, index=df.index)
    idx_len_ok = keep[keep].index
    if len(idx_len_ok):
        payload = df.loc[idx_len_ok, "csi_data"]
        payload_ok.loc[idx_len_ok] = (
            (_payload_token_count(payload) == C.CSI_RAW_LEN)
            & ~payload.str.contains(r"[^0-9,\s\[\]+-]", regex=True, na=True)
        )
    n_payload_bad = int(keep.sum() - payload_ok.sum())
    keep &= payload_ok

    df = df.loc[keep].copy()
    df["pc_elapsed_s"] = elapsed.loc[keep].astype(np.float64)
    n_dropped = n_raw - len(df)

    if len(df) == 0:
        raise CsiParseError(f"유효 패킷이 없다: {csi_path}")

    meta = {
        "n_raw_rows": int(n_raw),
        "n_dropped_rows": int(n_dropped),
        "n_dropped_payload_len": n_payload_bad,
        "bad_row_ratio": float(n_dropped / n_raw) if n_raw else 0.0,
    }
    return df, meta


# ------------------------------------------------------------------ 리샘플


def _resample_link(
    t_src: np.ndarray,
    x_src: np.ndarray,
    t_dst: np.ndarray,
    max_gap: float,
) -> tuple[np.ndarray, np.ndarray]:
    """링크 하나를 공통 격자로 선형보간한다.

    I/Q 를 실수부·허수부 각각 보간한다. 위상을 unwrap 한 뒤 보간하면 링크마다
    unwrap 기준이 달라 깨지므로, 복소수 성분 그대로 섞는 쪽이 안전하다.

    격자점 주변 max_gap 안에 실측 패킷이 없으면 mask=False, 값은 0 으로 둔다.
    (양 끝 바깥은 hold 가 아니라 masked-out 이다 — 없는 구간을 있는 척하지 않는다.)

    Args:
        t_src: [N] 오름차순
        x_src: [N, D]
        t_dst: [T]
    Returns:
        out:  [T, D] float32
        mask: [T] bool
    """
    D = x_src.shape[1]
    T = t_dst.shape[0]
    if len(t_src) == 0:
        return np.zeros((T, D), dtype=np.float32), np.zeros(T, dtype=bool)

    if len(t_src) == 1:
        out = np.repeat(x_src, T, axis=0).astype(np.float32)
        mask = np.abs(t_dst - t_src[0]) <= max_gap
        return out * mask[:, None], mask

    hi = np.clip(np.searchsorted(t_src, t_dst), 1, len(t_src) - 1)
    lo = hi - 1
    t0, t1 = t_src[lo], t_src[hi]
    span = t1 - t0
    w = np.where(span > 0, (t_dst - t0) / np.where(span > 0, span, 1.0), 0.0)
    w = np.clip(w, 0.0, 1.0)          # 구간 밖은 외삽하지 않고 끝값으로 고정
    out = x_src[lo] * (1.0 - w[:, None]) + x_src[hi] * w[:, None]

    # 가장 가까운 실측 패킷까지의 거리로 유효성 판정
    nearest = np.minimum(np.abs(t_dst - t0), np.abs(t_dst - t1))
    mask = nearest <= max_gap
    out[~mask] = 0.0
    return out.astype(np.float32), mask


def default_grid(duration_s: float, fps: float = C.TARGET_FPS) -> np.ndarray:
    n = max(1, int(round(duration_s * fps)))
    return (np.arange(n, dtype=np.float64) / fps).astype(np.float32)


# ------------------------------------------------- 학습·배포 공용 핵심

_LIVE_IDX = np.asarray(C.LIVE_SUBCARRIERS, dtype=np.int64)


def sanitize_phase(phi: np.ndarray, sc_idx: np.ndarray) -> np.ndarray:
    """패킷 내 위상에서 랜덤 성분을 제거한다 (SpotFi 방식).

    한 패킷의 위상은 이렇게 분해된다.

        phi(f) = phi_multipath(f) + 2*pi*f*tau + phi_0
                 ^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^
                 방/사람 구조       패킷마다 랜덤 (타이밍 오프셋 tau, CFO phi_0)

    뒤의 두 항은 패킷마다 값이 달라지지만 **그 패킷 안의 모든 subcarrier 에 공통**인
    선형 함수다. 그래서 subcarrier 축으로 직선을 최소자승 적합해 빼면 랜덤성이 사라지고
    다중경로 구조만 남는다.

    실측 근거 (개발셋 표본 12 trial x 3링크):
        raw 위상    인접 패킷 상관 -0.028   <- 그냥 쓰면 순수 잡음
        sanitize 후 인접 패킷 상관  0.950   <- 진폭(0.871)보다도 안정적

    Args:
        phi:    [N, S] 라디안 위상
        sc_idx: [S] 실제 subcarrier 인덱스. guard band 를 뺀 뒤라 번호가 연속이 아니므로
                (0~5, 63~65, 123~127 제거) 위치를 그대로 넘겨야 기울기가 맞다.
    """
    ph = np.unwrap(phi.astype(np.float64), axis=1)
    x = sc_idx.astype(np.float64)
    xm = x.mean()
    ym = ph.mean(axis=1, keepdims=True)
    denom = ((x - xm) ** 2).sum()
    k = ((ph - ym) * (x - xm)).sum(axis=1, keepdims=True) / max(denom, 1e-12)
    return (ph - (k * (x - xm) + ym)).astype(np.float32)


def iq_to_features(iq: np.ndarray, sc_idx: np.ndarray,
                   representation: str = C.CSI_REPRESENTATION) -> np.ndarray:
    """패킷 I/Q [N, S, 2] -> 보간해도 안전한 표현 [N, S, 2].

    **보간 전에 반드시 이 변환을 거쳐야 한다.**

    ESP32 CSI 의 위상은 패킷마다 랜덤이다(공통 위상회전 std 1.79 rad, 균등랜덤
    이론값 1.81). 그 상태의 I/Q 를 선형보간하면 방향이 제각각인 두 화살표의 중간점을
    잡는 셈이라 길이가 줄어든다. 실측 손실: 중앙 12%, 하위 10% 는 52%.
    30fps 격자점의 78% 가 패킷 사이에 떨어지므로 입력 전체에 랜덤 감쇠가 곱해진다.

    그래서 보간 전에 (진폭, 정제위상) 으로 바꾼다. 둘 다 패킷 간 연속성이 있어
    선형보간이 물리적으로 타당하다.
    """
    if representation == "iq":
        return iq
    if representation != "amp_phase":
        raise ValueError(f"unknown representation {representation!r}")
    I, Q = iq[..., 0], iq[..., 1]
    amp = np.sqrt(I * I + Q * Q)
    phi = sanitize_phase(np.arctan2(Q, I), sc_idx)
    return np.stack([amp, phi], axis=-1).astype(np.float32)


def parse_csi_line(line: str) -> tuple[str, np.ndarray] | None:
    """시리얼 한 줄 → (sender_mac, iq[256]). 형식이 어긋나면 None.

    데이터셋 CSV 의 `raw_line` 컬럼이 곧 이 시리얼 라인이다(수집 스크립트가 그대로
    저장한다). 그래서 배포의 시리얼 경로와 학습 데이터가 같은 형식을 공유한다.

    실측 43us/줄. 초당 100줄 도착 기준 CPU 0.43% 라 실시간에 부담이 없다.
    """
    if not line.startswith("CSI_DATA"):
        return None
    lb = line.rfind("[")
    rb = line.rfind("]")
    if lb < 0 or rb <= lb:
        return None
    head = line[:lb].split(",")
    if len(head) < 3:
        return None
    try:
        iq = np.array(line[lb + 1:rb].split(","), dtype=np.float32)
    except ValueError:
        return None
    if iq.size != C.CSI_RAW_LEN:
        return None
    return head[2], iq


def packets_to_grid(
    per_link_times: list[np.ndarray],
    per_link_iq: list[np.ndarray],
    grid_times: np.ndarray,
    max_gap: float = C.MAX_GAP_S,
    trim_guard: bool = True,
    representation: str = C.CSI_REPRESENTATION,
) -> tuple[np.ndarray, np.ndarray]:
    """**학습과 배포가 공유하는 유일한 전처리 핵심.** 파일 I/O 도 pandas 도 없다.

    링크별 비동기 패킷을 공통 시간축으로 재표본화하고 유효성 마스크를 만든다.
    앞단(CSV 를 읽었는지, 시리얼 링버퍼에서 가져왔는지)만 다르고 여기부터는 같다.
    학습과 배포의 전처리가 어긋나면 재현 불가능한 버그가 되므로 반드시 이 함수를 쓴다.

    순서가 중요하다: **guard band 제거 -> 표현 변환 -> 보간**.
    변환을 보간보다 먼저 해야 위상 랜덤성이 진폭을 갉아먹지 않는다(iq_to_features 참조).

    Args:
        per_link_times: 링크 순서(C.LINKS)대로 [N_l] 오름차순 시각(초)
        per_link_iq:    링크 순서대로 [N_l, 256] 생 I/Q
        grid_times:     [T] 목표 시각(초)
        trim_guard:     True 면 guard band 14개를 빼고 S=114 로 반환
        representation: "amp_phase"(기본) 또는 "iq"(구버전 재현용)
    Returns:
        feat [T, 3, S, 2] float32   ch0/ch1 의미는 representation 에 따름
        mask [T, 3] bool
    """
    t_dst = np.asarray(grid_times, dtype=np.float64)
    T = len(t_dst)
    sc_idx = _LIVE_IDX if trim_guard else np.arange(C.N_SUBCARRIERS)
    n_sc = len(sc_idx)
    feat = np.zeros((T, C.N_LINKS, n_sc, 2), dtype=np.float32)
    mask = np.zeros((T, C.N_LINKS), dtype=bool)

    for li in range(C.N_LINKS):
        t_src = np.asarray(per_link_times[li], dtype=np.float64)
        raw = np.asarray(per_link_iq[li], dtype=np.float32)
        if len(t_src) == 0:
            continue
        iq = raw.reshape(len(raw), C.N_SUBCARRIERS, 2)[:, sc_idx]     # [N, S, 2]
        x_src = iq_to_features(iq, sc_idx, representation)            # [N, S, 2]
        out, m = _resample_link(t_src, x_src.reshape(len(iq), -1), t_dst, max_gap)
        feat[:, li] = out.reshape(T, n_sc, 2)
        mask[:, li] = m
    return feat, mask


def load_csi_trial(
    csi_path: Path,
    grid_times: np.ndarray | None = None,
    max_gap: float = C.MAX_GAP_S,
    trim_guard: bool = True,
) -> CsiTrial:
    """한 trial 의 CSI 를 [T, 3, S, 2] + link_mask[T, 3] 으로 만든다. (학습 경로)

    CSV 를 읽어 링크별 패킷 배열을 만든 뒤 packets_to_grid() 로 넘긴다.
    배포 경로는 CSV 대신 시리얼 링버퍼에서 같은 배열을 만들어 같은 함수를 부른다.

    Args:
        csi_path: trial 의 csi.csv
        grid_times: 리샘플 목표 시각(초). 보통 align.frame_times() 로 만든 GT 프레임
            시각을 넘긴다. None 이면 0 부터 30fps 균일 격자를 쓴다.
        max_gap: 이 간격 안에 실측 패킷이 없으면 link_mask=False
        trim_guard: guard band 14개 제외 여부 (기본 제외 → S=114)
    """
    df, meta = read_csi_packets(csi_path)

    if grid_times is None:
        grid_times = default_grid(float(df["pc_elapsed_s"].max()))
    t_dst = np.asarray(grid_times, dtype=np.float64)
    T = len(t_dst)

    per_t: list[np.ndarray] = []
    per_x: list[np.ndarray] = []
    link_stats = {}
    duration = float(t_dst[-1] - t_dst[0]) if T > 1 else 1.0
    for tx in C.LINKS:
        sub = df[df["sender_id"] == tx]
        link_stats[tx] = {
            "n_packets": int(len(sub)),
            "hz": float(len(sub) / duration) if duration > 0 else 0.0,
        }
        if len(sub) == 0:
            per_t.append(np.zeros(0))
            per_x.append(np.zeros((0, C.CSI_RAW_LEN), dtype=np.float32))
            continue
        sub = sub.sort_values("pc_elapsed_s", kind="stable")
        per_t.append(sub["pc_elapsed_s"].to_numpy(dtype=np.float64))
        per_x.append(_parse_payload_block(sub["csi_data"]))

    iq, link_mask = packets_to_grid(per_t, per_x, t_dst, max_gap, trim_guard)

    meta.update({
        "preproc_version": C.PREPROC_VERSION,
        "links": link_stats,
        "link_coverage": {
            tx: float(link_mask[:, li].mean()) for tx, li in C.LINK_INDEX.items()
        },
        "any_link_coverage": float(link_mask.any(axis=1).mean()),
        "all_link_coverage": float(link_mask.all(axis=1).mean()),
        "max_gap_s": float(max_gap),
    })

    return CsiTrial(
        times=t_dst.astype(np.float32),
        iq=iq,
        link_mask=link_mask,
        meta=meta,
    )
