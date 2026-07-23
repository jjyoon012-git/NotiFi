from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


TRIAL_DURATION_S = 10.0
CUE_CYCLE_S = (2.5, 3.0, 3.5)
SUBJECTS = ("ajh", "lmh", "mhw", "yja")
ENVIRONMENTS = ("E01", "E02", "E03")


@dataclass(frozen=True)
class LabelSpec:
    id: str
    risk: str
    label: str
    korean: str
    context: str
    repeats_per_environment: int
    action_type: str
    start_pose: str
    timeline: tuple[tuple[str, str], ...]
    end_pose: str
    prohibited: str
    recollect: str
    variants: tuple[str, ...] = ()
    valid_for_reconstruction: bool = True

    @property
    def has_action_cue(self) -> bool:
        return self.action_type != "static"


LABEL_SPECS: tuple[LabelSpec, ...] = (
    LabelSpec(
        "S01",
        "SAFE",
        "walking",
        "직선 걷기",
        "floor",
        24,
        "transition",
        "1.5 m 보행 경로의 시작선 A에서 종료선 B를 바라보고 선다. 양팔은 몸통 옆에 자연스럽게 둔다.",
        (
            ("0.0-c", "시작선에서 발을 움직이지 않고 정지한다."),
            ("c-c+2.4", "표시선을 따라 A와 B 사이를 한 번 걷는다. 개인 기준속도 2.4초/1.5 m, 허용오차 ±0.3초를 지킨다."),
            ("c+2.4-10.0", "종료 표시 안에서 바로 서서 정지한다."),
        ),
        "종료선에서 정지 서기",
        "뛰기, 방향 전환, 멈칫하기, 발 끌기, 물건 들기, 경로 밖 이동",
        "이동시간이 2.1초 미만 또는 2.7초 초과, 경로 이탈 20 cm 초과, 비틀거림 또는 외부 접촉",
        ("A_to_B", "B_to_A"),
    ),
    LabelSpec(
        "S02",
        "SAFE",
        "standing_still",
        "정지 서기",
        "floor",
        12,
        "static",
        "활동 중심 C에서 카메라를 정면으로 본다. 발뒤꿈치 간격 20 cm, 팔은 몸통 옆, 손바닥은 허벅지 쪽이다.",
        (("0.0-10.0", "발, 팔, 고개, 몸통 위치를 바꾸지 않고 정지한다."),),
        "시작 자세와 동일",
        "체중 이동, 팔짱, 주머니에 손 넣기, 고개 회전, 말하기",
        "발이 표시에서 벗어나거나 팔 또는 몸통의 큰 움직임이 0.5초 이상 발생",
    ),
    LabelSpec(
        "S03",
        "SAFE",
        "sitting_still",
        "의자 정지 앉기",
        "chair",
        12,
        "static",
        "고정 의자 좌판 중앙에 앉는다. 양발은 바닥 발 표시 위, 무릎 90도, 손바닥은 허벅지 위, 등은 등받이에 가볍게 둔다.",
        (("0.0-10.0", "손, 발, 몸통을 움직이지 않고 시작 자세를 유지한다."),),
        "시작 자세와 동일",
        "다리 꼬기, 발 들기, 상체 숙이기, 몸 흔들기",
        "엉덩이 위치 이동, 한 발이 들림, 의자가 움직임, 상체가 10도 이상 기울어짐",
    ),
    LabelSpec(
        "S04",
        "SAFE",
        "lying_still",
        "침대 정지 눕기",
        "bed",
        18,
        "static",
        "침대 중앙에 바로 눕는다. 머리는 카메라에서 먼 쪽, 발은 카메라 쪽이다. 팔은 몸통에서 10 cm, 다리는 곧게 편다.",
        (("0.0-10.0", "팔, 다리, 머리, 몸통을 움직이지 않고 바로 누운 자세를 유지한다."),),
        "시작 자세와 동일",
        "뒤척임, 무릎 굽힘, 팔 위치 변경, 고개 돌리기",
        "관절 자세가 0.5초 이상 변하거나 침대가 움직임",
    ),
    LabelSpec(
        "S05",
        "SAFE",
        "lie_to_stand",
        "침대에서 정상적으로 일어나기",
        "bed",
        18,
        "transition",
        "S04와 동일한 바로 누운 자세. 지정된 한쪽 침대 이탈 방향을 모든 trial에서 유지한다.",
        (
            ("0.0-c", "바로 누운 자세를 유지한다."),
            ("c-c+0.8", "지정 측으로 옆으로 돌아눕는다."),
            ("c+0.8-c+1.8", "지정 팔로 침대를 밀며 상체를 세우고 두 다리를 바닥으로 내린다."),
            ("c+1.8-c+3.0", "양발로 바닥을 밀어 한 번에 완전히 일어선다."),
            ("c+3.0-10.0", "침대 옆 발 표시 위에서 정지 서기를 유지한다."),
        ),
        "침대 옆 정지 서기",
        "두 번 이상 일어서기 시도, 침대 또는 벽을 오래 잡기, 다시 앉기",
        "완전 기립까지 3.5초 초과, 중간 재시도, 손 또는 무릎이 바닥 매트에 닿음",
    ),
    LabelSpec(
        "S06",
        "SAFE",
        "stand_to_lie_normal",
        "침대에 정상적으로 눕기",
        "bed",
        30,
        "transition",
        "침대 긴 변 중앙 앞 발 표시에서 침대를 향해 선다. 허벅지와 침대 가장자리 간격은 10 cm다.",
        (
            ("0.0-c", "정지 서기를 유지한다."),
            ("c-c+0.8", "침대 가장자리 중앙에 앉는다."),
            ("c+0.8-c+1.8", "지정 전완으로 침대를 지지하며 어깨와 몸통을 천천히 내린다."),
            ("c+1.8-c+3.0", "두 다리를 침대 위로 올리고 바로 누운 자세로 정렬한다."),
            ("c+3.0-10.0", "침대 중앙에서 바로 누운 자세를 유지한다."),
        ),
        "침대 중앙 바로 눕기",
        "몸을 침대에 던지기, 침대 밖 매트 접촉, 머리 또는 어깨 충격",
        "엉덩이가 가장자리에 걸림, 동작 순서 위반, 2초 이내 급격한 눕기, 전신이 영상에서 잘림",
    ),
    LabelSpec(
        "S07",
        "SAFE",
        "absence",
        "부재",
        "empty_room",
        12,
        "static",
        "피험자, 운영자, 안전요원 모두 카메라와 활동 영역 밖으로 완전히 이동한다. 문은 세션 기준 상태로 고정한다.",
        (("0.0-10.0", "사람, 반려동물, 문, 커튼, 선풍기, 가구, 케이블이 움직이지 않는 빈 공간을 유지한다."),),
        "빈 공간 유지",
        "문 개폐, 사람 그림자, 반려동물, 선풍기 회전, TV 또는 음악",
        "어떤 동적 객체라도 프레임 또는 활동 영역에 들어옴",
        valid_for_reconstruction=False,
    ),
    LabelSpec(
        "S08",
        "SAFE",
        "sit_to_stand",
        "의자에서 정상적으로 일어서기",
        "chair",
        12,
        "transition",
        "S03과 동일한 착석 자세. 양발은 무릎 바로 아래 발 표시 위에 둔다.",
        (
            ("0.0-c", "정지 앉기를 유지한다."),
            ("c-c+0.6", "몸통을 약 15도 앞으로 기울인다."),
            ("c+0.6-c+1.8", "양발로 바닥을 밀어 한 번에 완전히 일어선다."),
            ("c+1.8-10.0", "의자 앞 표시에서 정지 서기를 유지한다."),
        ),
        "의자 앞 정지 서기",
        "팔걸이, 무릎 또는 의자 짚기, 재시도, 다시 앉기",
        "완전 기립까지 2.3초 초과, 손 지지, 두 번 이상 시도, 의자 이동",
    ),
    LabelSpec(
        "S09",
        "SAFE",
        "stand_to_sit",
        "의자에 정상적으로 앉기",
        "chair",
        12,
        "transition",
        "의자 앞 40 cm 발 표시에서 등을 의자에 향하고 선다.",
        (
            ("0.0-c", "정지 서기를 유지한다."),
            ("c-c+0.6", "한 걸음 뒤로 이동해 종아리가 의자 앞면에 가볍게 닿게 한다."),
            ("c+0.6-c+1.8", "무릎과 고관절을 함께 굽혀 좌판 중앙에 앉는다."),
            ("c+1.8-10.0", "양발은 바닥, 손은 허벅지 위에 둔 채 정지한다."),
        ),
        "의자 좌판 중앙 정지 앉기",
        "좌판 가장자리 걸침, 털썩 앉기, 의자 잡기, 옆으로 치우치기",
        "엉덩이 중심이 좌판 중심에서 10 cm 이상 벗어남, 의자 이동, 바닥 접촉",
    ),
    LabelSpec(
        "W01",
        "WARNING",
        "unstable_walking",
        "지속적 불안정 보행",
        "floor",
        20,
        "warning",
        "보행 경로 시작선 A에서 종료선 B를 바라보고 선다.",
        (
            ("0.0-c", "시작선에서 정지한다."),
            ("c-c+2.8", "A와 B 사이를 걷되 경로 1/3에서 몸통을 왼쪽 10 cm, 2/3에서 오른쪽 10 cm 이동한다. 발은 경로 안에 두고 멈추지 않는다."),
            ("c+2.8-10.0", "종료선에서 균형을 회복하고 정지한다."),
        ),
        "종료선에서 정지 서기",
        "벽 또는 가구 접촉, 발 교차, 실제 넘어짐, 두 번 이상 멈춤",
        "손, 무릎 또는 엉덩이가 지지물이나 바닥에 닿음, 경로 이탈 20 cm 초과",
        ("A_to_B", "B_to_A"),
    ),
    LabelSpec(
        "W02",
        "WARNING",
        "stumble_recover",
        "발을 헛디딘 뒤 회복",
        "floor",
        30,
        "warning",
        "보행 경로 시작선 A에서 종료선 B를 바라보고 선다. 실제 장애물은 두지 않고 C 지점에 평면 표식만 둔다.",
        (
            ("0.0-c", "시작선에서 정지한다."),
            ("c-c+1.4", "C 방향으로 정상 보행한다."),
            ("c+1.4-c+2.1", "C에서 앞발을 갑자기 멈추고 몸통을 앞으로 기울인다. 반대발로 40-60 cm 회복 스텝 한 번을 내딛고 양팔을 옆으로 벌린다."),
            ("c+2.1-c+3.4", "몸통을 다시 세우고 종료선까지 걷는다."),
            ("c+3.4-10.0", "종료선에서 정지한다."),
        ),
        "종료선에서 정지 서기",
        "실제 장애물, 손 또는 무릎 바닥 접촉, 두 번 이상 비틀거림, 낙상",
        "바닥 또는 가구 접촉, 회복 실패, 회복 스텝이 30 cm 미만 또는 70 cm 초과",
        ("recover_left", "recover_right"),
    ),
    LabelSpec(
        "W03",
        "WARNING",
        "bed_exit_failed",
        "침대에서 일어서려다 실패 후 다시 앉기",
        "bed",
        25,
        "warning",
        "S04와 동일한 바로 누운 자세. 이탈 측은 S05 및 D04와 동일하다.",
        (
            ("0.0-c", "바로 누운 자세를 유지한다."),
            ("c-c+0.8", "지정 측으로 돌아눕는다."),
            ("c+0.8-c+1.8", "상체를 세우고 두 다리를 내려 침대 가장자리에 앉는다."),
            ("c+1.8-c+2.8", "양발로 바닥을 밀어 엉덩이를 5-10 cm 들어 올리지만 완전히 서지 않는다."),
            ("c+2.8-c+3.6", "같은 좌석 위치로 다시 앉는다."),
            ("c+3.6-10.0", "침대 가장자리에서 정지 앉기를 유지한다."),
        ),
        "침대 가장자리 정지 앉기",
        "완전 기립, 침대 밖으로 미끄러짐, 바닥에 손 또는 무릎 접촉, 다시 눕기",
        "엉덩이가 15 cm 이상 상승, 완전히 서기, 매트 접촉, 두 번 이상 시도",
    ),
    LabelSpec(
        "D01",
        "DANGER",
        "fall_from_standing",
        "서 있는 상태에서 측면 낙상",
        "floor_mat",
        10,
        "danger",
        "이중 매트 중앙 C에서 선다. 지정 낙상 측에 안전요원이 대기하고 머리 쪽 매트는 이중으로 겹친다.",
        (
            ("0.0-c", "정지 서기를 유지한다."),
            ("c-c+0.6", "양 무릎을 굽히고 지정 측으로 체중을 이동한다."),
            ("c+0.6-c+1.8", "지정 측 엉덩이와 전완이 순서대로 매트에 닿도록 통제된 측면 하강을 수행한다."),
            ("c+1.8-10.0", "옆으로 누운 최종 자세를 유지한다."),
        ),
        "지정 측 측와위",
        "후방 자유낙하, 점프, 손목, 머리 또는 목 첫 접촉, 매트 밖 착지",
        "머리 또는 손목 충격, 발을 여러 번 옮김, actual onset 오차 0.5초 초과, 안전요원 접촉",
        ("fall_left", "fall_right"),
    ),
    LabelSpec(
        "D02",
        "DANGER",
        "fall_while_walking",
        "걷다가 발을 헛디뎌 낙상",
        "floor_path_mat",
        10,
        "danger",
        "보행 경로 시작선 A에서 종료선 B를 바라보고 선다. C는 매트 중앙이며 실제 장애물은 두지 않는다.",
        (
            ("0.0-c", "시작선에서 정지한다."),
            ("c-c+1.4", "A에서 C 방향으로 정상 보행한다."),
            ("c+1.4-c+2.0", "C에서 앞발을 갑자기 멈추고 몸통을 전방 대각으로 이동시킨다."),
            ("c+2.0-c+3.0", "양 무릎과 지정 측 전완을 순서대로 매트에 대며 통제된 낙상을 수행한다."),
            ("c+3.0-10.0", "매트 위 비직립 자세를 유지한다."),
        ),
        "무릎, 전완, 몸통이 매트에 닿은 비직립 자세",
        "손바닥 또는 손목 충격, 머리 접촉, 낙상 후 다시 일어나기, 매트 밖 접촉",
        "회복하여 선 경우, 무릎 또는 전완 순서 위반, 안전요원 접촉, 매트 밖 착지",
        ("fall_diagonal_left", "fall_diagonal_right"),
    ),
    LabelSpec(
        "D03",
        "DANGER",
        "fall_collapse",
        "다리에 힘이 풀리는 주저앉기형 낙상",
        "floor_mat",
        10,
        "danger",
        "이중 매트 중앙 C에서 양발을 어깨너비로 두고 선다.",
        (
            ("0.0-c", "정지 서기를 유지한다."),
            ("c-c+0.7", "양 무릎을 동시에 빠르게 굽혀 골반을 낮춘다."),
            ("c+0.7-c+1.8", "지정 측 무릎, 엉덩이, 전완 순으로 매트에 닿으며 옆으로 내려간다."),
            ("c+1.8-10.0", "옆으로 누운 최종 자세를 유지한다."),
        ),
        "지정 측 측와위",
        "단순 쪼그려 앉기, 다시 일어서기, 뒤로 젖히기, 점프",
        "엉덩이가 바닥에 닿지 않음, 무릎 꿇기로 끝남, 머리 또는 손목 충격, 동작 순서 위반",
        ("collapse_left", "collapse_right"),
    ),
    LabelSpec(
        "D04",
        "DANGER",
        "bed_fall",
        "침대에서 일어난 직후 옆으로 낙상",
        "bed_mat",
        10,
        "danger",
        "S04와 같은 바로 누운 자세. 침대 이탈 측 바닥 전체를 이중 매트로 덮고 안전요원이 지정 측에 선다.",
        (
            ("0.0-c", "바로 누운 자세를 유지한다."),
            ("c-c+0.8", "지정 측으로 돌아눕는다."),
            ("c+0.8-c+1.8", "상체를 세우고 두 다리를 내려 침대 가장자리에 앉는다."),
            ("c+1.8-c+2.8", "양발로 바닥을 밀어 완전히 일어선다."),
            ("c+2.8-c+4.0", "한쪽 무릎이 풀리는 동작으로 침대 옆 지정 측 엉덩이와 전완을 매트에 대며 내려간다."),
            ("c+4.0-10.0", "침대 옆 매트에서 옆으로 누운 자세를 유지한다."),
        ),
        "침대 옆 매트 측와위",
        "침대에서 굴러 떨어지기, 머리부터 떨어지기, 침대 또는 벽을 잡기, 완전 자유낙하",
        "완전 기립 전에 바로 바닥으로 내려감, 매트 밖 접촉, 머리 또는 손목 충격, 안전요원 접촉",
        ("bed_fall_left", "bed_fall_right"),
    ),
    LabelSpec(
        "D05",
        "DANGER",
        "chair_fall",
        "의자에 앉으려다 좌판을 놓쳐 낙상",
        "chair_mat",
        10,
        "danger",
        "S09와 같은 위치에서 등을 의자에 향하고 선다. 지정 낙상 측 의자 옆을 이중 매트로 덮고 의자를 고정한다.",
        (
            ("0.0-c", "정지 서기를 유지한다."),
            ("c-c+0.6", "한 걸음 뒤로 이동한다."),
            ("c+0.6-c+1.4", "엉덩이를 지정 측으로 15 cm 이상 치우치며 앉기 동작을 시작한다."),
            ("c+1.4-c+2.6", "좌판을 완전히 놓치고 지정 측 엉덩이와 전완 순서로 의자 옆 매트에 통제 착지한다."),
            ("c+2.6-10.0", "옆으로 누운 최종 자세를 유지한다."),
        ),
        "의자 옆 매트 측와위",
        "의자 전도, 좌판에 걸쳐 멈춤, 머리 또는 손목 첫 접촉, 매트 밖 착지",
        "엉덩이가 좌판에 닿음, 의자가 2 cm 이상 움직임, 안전요원 접촉, 매트 밖 착지",
        ("chair_fall_left", "chair_fall_right"),
    ),
)

LABELS = {spec.label: spec for spec in LABEL_SPECS}


def get_label(label: str) -> LabelSpec:
    try:
        return LABELS[label]
    except KeyError as exc:
        valid = ", ".join(LABELS)
        raise ValueError(f"Unknown label {label!r}. Valid labels: {valid}") from exc


def cue_for_trial(spec: LabelSpec, trial_number: int) -> float | None:
    if not spec.has_action_cue:
        return None
    return CUE_CYCLE_S[(trial_number - 1) % len(CUE_CYCLE_S)]


def variant_for_trial(spec: LabelSpec, trial_number: int) -> str:
    if not spec.variants:
        return "fixed"
    return spec.variants[(trial_number - 1) % len(spec.variants)]


def total_repeats(specs: Iterable[LabelSpec] = LABEL_SPECS) -> int:
    return sum(spec.repeats_per_environment for spec in specs)


def validate_specs() -> None:
    if len(LABEL_SPECS) != 17:
        raise AssertionError("NotiFi v2.0 must contain exactly 17 scenario labels")
    if total_repeats() != 275:
        raise AssertionError("NotiFi v2.0 must contain 275 trials per subject/environment")
    ids = [spec.id for spec in LABEL_SPECS]
    labels = [spec.label for spec in LABEL_SPECS]
    if len(ids) != len(set(ids)) or len(labels) != len(set(labels)):
        raise AssertionError("Label IDs and names must be unique")
    for spec in LABEL_SPECS:
        if spec.risk not in {"SAFE", "WARNING", "DANGER"}:
            raise AssertionError(f"Invalid risk for {spec.label}: {spec.risk}")


validate_specs()
