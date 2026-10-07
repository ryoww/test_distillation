"""スケジューリング系の参照解の形を厳密に検証するチェッカーが、1 箇所壊した解を弾くことを確認する。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.datagen import TEMPLATES
from src.utils.feasibility import check_feasibility_detailed

PROBLEM_DIR = Path(__file__).resolve().parents[1] / "data" / "problems"


def _load(problem_id: int) -> tuple[str, dict, dict]:
    record = json.loads((PROBLEM_DIR / f"prob_{problem_id:03d}.json").read_text(encoding="utf-8"))
    core_type = f"{record['domain']}_{record['math_type']}"
    return core_type, record["instance"], record["reference_solution"]


def _solved(problem_id: int) -> tuple[str, dict, dict]:
    """同梱参照解が壊れている問題は、雛形ソルバーの厳密解を「正しい解」として使う。"""
    core_type, instance, _ = _load(problem_id)
    return core_type, instance, TEMPLATES[problem_id].solve(instance)


def _check(core_type: str, instance: dict, solution: dict) -> dict:
    return check_feasibility_detailed(core_type, instance, solution)


def _assert_violation(result: dict, needle: str) -> None:
    assert result["verified"] is True
    assert result["feasible"] is False
    assert any(needle in v for v in result["violations"]), result["violations"]


def test_instance_without_jobs_and_unknown_shape_is_unverified():
    """What: jobs のない instance で読める形がなければ、満点ではなく未検証として返る。"""
    core_type, instance, reference = _load(9)
    result = _check(core_type, instance, reference)
    assert result["verified"] is False
    assert result["total_constraints"] == 0


def test_job_shop_reference_passes_and_missing_operation_is_reported():
    """What: prob_004 の参照解は通り、工程を 1 つ落とすと弾かれる。"""
    core_type, instance, reference = _load(4)
    assert _check(core_type, instance, reference)["feasible"] is True
    broken = copy.deepcopy(reference)
    broken["schedule"]["4"].pop()
    _assert_violation(_check(core_type, instance, broken), "job 4 does not schedule all")


def test_job_shop_machine_overlap_and_makespan_drift_are_reported():
    """What: 同じ機械の工程を重ねる、makespan を 1 ずらす、のどちらも違反になる。"""
    core_type, instance, reference = _load(4)
    overlapped = copy.deepcopy(reference)
    overlapped["schedule"]["3"][0].update(start_time=1, end_time=5)
    _assert_violation(_check(core_type, instance, overlapped), "machine 3")
    drifted = copy.deepcopy(reference)
    drifted["makespan"] += 1
    _assert_violation(_check(core_type, instance, drifted), "makespan 17 != last completion 16")


def test_nurse_roster_three_consecutive_nights_and_match_count_are_reported():
    """What: prob_005 で連続夜勤 3 日と希望一致数の誤申告が違反になる。"""
    core_type, instance, reference = _load(5)
    assert _check(core_type, instance, reference)["feasible"] is True
    nights = copy.deepcopy(reference)
    nights["roster"]["nurse_2"][2] = "night"
    _assert_violation(_check(core_type, instance, nights), "nurse 2 works 5 consecutive nights")
    miscounted = copy.deepcopy(reference)
    miscounted["preferred_shift_matches"] += 1
    _assert_violation(_check(core_type, instance, miscounted), "preferred_shift_matches 37")


def test_nurse_roster_short_staffed_day_is_reported():
    """What: ある日の夜勤人数が必要数を下回ると違反になる。"""
    core_type, instance, reference = _load(5)
    short = copy.deepcopy(reference)
    short["roster"]["nurse_6"][0] = "morning"
    short["preferred_shift_matches"] += 1
    short["match_rate"] = 37 / 42
    _assert_violation(_check(core_type, instance, short), "day 1 night: 1 nurses < required 2")


def test_meeting_rooms_capacity_and_unassigned_count_are_checked():
    """What: prob_006 で定員超過の部屋割りと unassigned_count の誤申告が違反になる。"""
    core_type, instance, reference = _load(6)
    assert _check(core_type, instance, reference)["feasible"] is True
    crowded = copy.deepcopy(reference)
    crowded["room_assignment"]["3"] = 3
    crowded["unassigned_count"] = 1
    _assert_violation(_check(core_type, instance, crowded), "meeting 3 has more attendees")
    lying = copy.deepcopy(reference)
    lying["unassigned_count"] = 1
    _assert_violation(_check(core_type, instance, lying), "unassigned_count 1 != 2")


def test_meeting_rooms_time_overlap_in_one_room_is_reported():
    """What: 同じ会議室で時間の重なる 2 会議は違反になる。"""
    core_type, instance, reference = _load(6)
    overlapped = copy.deepcopy(reference)
    overlapped["room_assignment"]["3"] = 1
    overlapped["unassigned_count"] = 1
    _assert_violation(_check(core_type, instance, overlapped), "room 1: meeting 3 and meeting 1")


def test_gate_assignment_size_and_overlap_are_checked():
    """What: prob_007 で小さいゲートへの大型機と同じゲートの時間重複が違反になる。"""
    core_type, instance, reference = _load(7)
    assert _check(core_type, instance, reference)["feasible"] is True
    too_small = copy.deepcopy(reference)
    too_small["gate_assignment"]["1"] = 2
    _assert_violation(_check(core_type, instance, too_small), "flight 1 is too large for gate 2")
    overlapped = copy.deepcopy(reference)
    overlapped["gate_assignment"]["6"] = 1
    _assert_violation(_check(core_type, instance, overlapped), "gate 1: flight 6 and flight 8")


def test_operating_rooms_surgeon_clash_and_start_sum_are_checked():
    """What: prob_008 で同じ医師の同時執刀と total_start_time の誤申告が違反になる。"""
    core_type, instance, reference = _load(8)
    assert _check(core_type, instance, reference)["feasible"] is True
    clash = copy.deepcopy(reference)
    clash["schedule"]["5"].update(start_time=8, end_time=8 + 130 / 60)
    _assert_violation(_check(core_type, instance, clash), "surgeon 1: surgery 2 and surgery 5")
    drifted = copy.deepcopy(reference)
    drifted["total_start_time"] += 1
    _assert_violation(_check(core_type, instance, drifted), "total_start_time 65")


def test_operating_rooms_wrong_room_type_and_late_finish_are_checked():
    """What: 手術類型と違う部屋、18:00 を超える終了は違反になる。"""
    core_type, instance, reference = _load(8)
    wrong_room = copy.deepcopy(reference)
    wrong_room["schedule"]["2"]["room"] = 3
    _assert_violation(_check(core_type, instance, wrong_room), "surgery 2 is in a cardiac room")
    late = copy.deepcopy(reference)
    late["schedule"]["1"].update(start_time=16, end_time=16 + 157 / 60)
    late["total_start_time"] += 2
    _assert_violation(_check(core_type, instance, late), "surgery 1 falls outside 8:00-18:00")


def test_rcpsp_exact_solution_passes_and_capacity_breach_is_reported():
    """What: prob_011 の厳密解は通り、作業員の合計が 6 を超える前倒しは違反になる。"""
    core_type, instance, solved = _solved(11)
    assert _check(core_type, instance, solved)["feasible"] is True
    crowded = copy.deepcopy(solved)
    crowded["schedule"]["5"] = {"start_time": 11, "end_time": 14}
    crowded["makespan"] = 14
    _assert_violation(_check(core_type, instance, crowded), "workers usage peaks at 10 > capacity")


def test_rcpsp_precedence_and_makespan_are_checked():
    """What: 先行活動の完了前に始める活動、makespan の 1 ずれは違反になる。"""
    core_type, instance, solved = _solved(11)
    early = copy.deepcopy(solved)
    early["schedule"]["2"] = {"start_time": 1, "end_time": 3}
    _assert_violation(_check(core_type, instance, early), "activity 2 starts before predecessor 1")
    drifted = copy.deepcopy(solved)
    drifted["makespan"] += 1
    _assert_violation(_check(core_type, instance, drifted), "makespan 18 != last completion 17")


def test_cluster_assignment_sums_requirements_per_node():
    """What: prob_012 の厳密解は通り、同じノードに載せたジョブの合計がメモリ容量を超えると違反になる。"""
    core_type, instance, solved = _solved(12)
    assert _check(core_type, instance, solved)["feasible"] is True
    packed = copy.deepcopy(solved)
    packed["node_assignment"]["6"] = 3
    packed["total_priority"] += 1
    packed["assigned_count"] += 1
    _assert_violation(_check(core_type, instance, packed), "node 3 memory_gb=128 < total")


def test_broadcast_lineup_slot_reuse_genre_minimum_and_total_are_checked():
    """What: prob_013 でスロットの二重使用、ジャンル最低数割れ、視聴率和の 1 ずれが違反になる。"""
    core_type, instance, reference = _load(13)
    assert _check(core_type, instance, reference)["feasible"] is True
    reused = copy.deepcopy(reference)
    reused["lineup"]["2"] = "8:00-9:00"
    _assert_violation(_check(core_type, instance, reused), "slot 8:00-9:00 holds programs 2 and 4")
    thin = copy.deepcopy(reference)
    del thin["lineup"]["8"]
    thin["total_expected_rating"] = round(reference["total_expected_rating"] - 7.857431808841254, 2)
    _assert_violation(_check(core_type, instance, thin), "genre documentary: 1 programs < minimum")
    drifted = copy.deepcopy(reference)
    drifted["total_expected_rating"] += 1
    _assert_violation(_check(core_type, instance, drifted), "total_expected_rating 148.55")


def _complete_timetable() -> dict:
    """同梱参照解に不足していた週 3 時限の科目の残り時限を、衝突しないように足した完全な授業表。"""
    _, _, reference = _load(15)
    timetable = copy.deepcopy(reference["timetable"])
    timetable["3"] = [timetable["3"], {"day": "tue", "period": 2, "room": 2},
                      {"day": "wed", "period": 2, "room": 2}]
    timetable["4"] = [timetable["4"], {"day": "wed", "period": 1, "room": 1},
                      {"day": "thu", "period": 1, "room": 1}]
    timetable["7"] = [timetable["7"], {"day": "tue", "period": 3, "room": 1},
                      {"day": "wed", "period": 3, "room": 1}]
    return {"timetable": timetable, "total_period_sum": 0, "all_assigned": True}


def test_class_timetable_requires_every_weekly_period():
    """What: prob_015 は全時限を置いた授業表なら通り、同梱参照解のように 1 時限だけでは弾かれる。"""
    core_type, instance, reference = _load(15)
    assert _check(core_type, instance, _complete_timetable())["feasible"] is True
    _assert_violation(_check(core_type, instance, reference), "class 3 has 1 periods, needs 3")


def test_class_timetable_teacher_clash_and_small_room_are_reported():
    """What: 同じ教員の同時限重複と容量不足の教室は違反になる。"""
    core_type, instance, _ = _load(15)
    clash = _complete_timetable()
    clash["timetable"]["6"] = {"day": "tue", "period": 1, "room": 2}
    _assert_violation(_check(core_type, instance, clash), "teacher 1 is double booked on tue 1")
    small = _complete_timetable()
    small["timetable"]["8"]["room"] = 1
    _assert_violation(_check(core_type, instance, small), "class 8 needs a larger room than 1")


def test_event_staff_roster_missing_placement_and_availability_are_checked():
    """What: prob_019 で最低人数割れ、利用不可時間帯への配置、総配置数の誤申告が違反になる。"""
    core_type, instance, reference = _load(19)
    assert _check(core_type, instance, reference)["feasible"] is True
    short = copy.deepcopy(reference)
    del short["roster"]["9"]["11:00-12:00"]
    short["total_assignments"] -= 1
    _assert_violation(_check(core_type, instance, short), "role 2 slot 4: 0 staff < required 1")
    unavailable = copy.deepcopy(reference)
    unavailable["roster"]["4"]["12:00-13:00"] = "受付"
    unavailable["total_assignments"] += 1
    _assert_violation(_check(core_type, instance, unavailable), "staff 4 is unavailable at 12:00")
    miscounted = copy.deepcopy(reference)
    miscounted["total_assignments"] += 1
    _assert_violation(_check(core_type, instance, miscounted), "total_assignments 14 != 13")


def test_event_staff_roster_skill_mismatch_is_reported():
    """What: 役割の必要スキルを持たないスタッフの配置は違反になる。"""
    core_type, instance, reference = _load(19)
    unskilled = copy.deepcopy(reference)
    unskilled["roster"]["8"]["8:00-9:00"] = "音響"
    _assert_violation(_check(core_type, instance, unskilled), "staff 8 lacks the skill for 音響")


def test_multi_project_exact_solution_passes_and_dropped_activity_is_reported():
    """What: prob_020 の厳密解は通り、活動を 1 つ落とすと弾かれる。"""
    core_type, instance, solved = _solved(20)
    assert _check(core_type, instance, solved)["feasible"] is True
    dropped = copy.deepcopy(solved)
    del dropped["schedule"]["14"]
    _assert_violation(_check(core_type, instance, dropped), "activity 14 is not scheduled")


def test_multi_project_peak_and_capacity_are_checked():
    """What: ピーク値の 1 ずれと、容量 8 を超える重ね合わせは違反になる。"""
    core_type, instance, solved = _solved(20)
    drifted = copy.deepcopy(solved)
    drifted["peak_resource_usage"] += 1
    _assert_violation(_check(core_type, instance, drifted), "peak_resource_usage 7 != recomputed 6")
    crowded = copy.deepcopy(solved)
    crowded["schedule"]["2"] = {"start": 3, "end": 8, "project": 1}
    crowded["schedule"]["3"] = {"start": 3, "end": 8, "project": 1}
    crowded["makespan"] = 14
    _assert_violation(_check(core_type, instance, crowded), "> capacity 8")


@pytest.mark.parametrize("problem_id", [4, 5, 6, 7, 8, 11, 12, 13, 19, 20])
def test_template_solver_output_is_verified_feasible(problem_id):
    """What: 雛形ソルバーの出力は、対応する厳密チェッカーで verified かつ feasible になる。"""
    core_type, instance, solved = _solved(problem_id)
    result = _check(core_type, instance, solved)
    assert result["verified"] is True
    assert result["feasible"] is True, result["violations"]
