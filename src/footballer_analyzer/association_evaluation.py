"""固定分母评估关联边、融合组与第一推荐；未知身份一律不计为正确。"""

from itertools import combinations

from .field_review import merge_groups
from .reassociation import association_view


def evaluate(snapshot, truth, review, run_dir):
    """使用完整片段清单与独立身份真值计算验收指标和逐边错误。"""
    segments = [dict(item) for item in snapshot["segments"]]
    ids = {item["record_id"] for item in segments}
    labels = truth.get("records", {})
    if truth.get("input_signature") != snapshot["input_signature"] or set(labels) != ids:
        raise ValueError("真值摘要或片段清单不匹配；不得改变验收分母")
    if not ids:
        raise ValueError("验收输入不能为空")
    for label in labels.values():
        if not isinstance(label, dict) or type(label.get("internally_consistent")) not in {bool, type(None)}:
            raise ValueError("每条真值必须记录内部串号检查结果")
        if label.get("player_id") is not None and (not isinstance(label["player_id"], str) or not label["player_id"].strip()):
            raise ValueError("球员身份必须是非空字符串或未知值 null")

    def identity(record_id):
        """内部串号或未检查的片段不作为确定身份。"""
        label = labels.get(record_id, {})
        return label.get("player_id") if label.get("internally_consistent") is True else None

    def correct(source, target):
        """无法确认的边按未正确处理。"""
        return bool(identity(source) and identity(source) == identity(target))

    # 硬指标只使用自动边，避免人工操作帮助达到减半要求。
    operations = [item for item in review["operations"] if item.get("execution", "manual") == "auto"]
    history = merge_groups(segments, operations, snapshot["input_signature"])
    active = [item for item in history if item["status"] == "active"]
    edges = [{"operation_id": item["operation_id"], "source": item["source_record_id"], "target": item["target_record_id"],
              "score": item.get("score"), "correct": correct(item["source_record_id"], item["target_record_id"])} for item in active]
    edge_precision = sum(item["correct"] for item in edges) / len(edges) if edges else None
    groups = {}
    for item in segments:
        groups.setdefault(item["merge_group_id"], []).append(item["record_id"])
    pairs = [(source, target) for members in groups.values() for source, target in combinations(members, 2)]
    pair_precision = sum(correct(*pair) for pair in pairs) / len(pairs) if pairs else None
    contaminated = [{"group_id": key, "records": members} for key, members in groups.items()
                    if len(members) > 1 and any(not correct(*pair) for pair in combinations(members, 2))]
    view = association_view(run_dir, segments, {**review, "operations": operations}, snapshot["input_signature"])
    recommendations = list({(item["source_record_id"], item["target_record_id"]): item for item in view["recommendations"].values()}.values())
    recommendation_rows = [{"source": item["source_record_id"], "target": item["target_record_id"], "score": item["score"],
                            "correct": correct(item["source_record_id"], item["target_record_id"])} for item in recommendations]
    # 覆盖率按尚未连接的已确认同身份组对计算；未知不伪造可关联分母。
    remaining = set()
    for left, right in combinations(segments, 2):
        if left["merge_group_id"] != right["merge_group_id"] and correct(left["record_id"], right["record_id"]):
            remaining.add(tuple(sorted([left["merge_group_id"], right["merge_group_id"]])))
    by_id = {item["record_id"]: item for item in segments}
    covered = {tuple(sorted([by_id[item["source_record_id"]]["merge_group_id"], by_id[item["target_record_id"]]["merge_group_id"]]))
               for item in recommendations if correct(item["source_record_id"], item["target_record_id"])}
    annotated = sum(identity(key) is not None for key in ids)
    recommendation_precision = sum(item["correct"] for item in recommendation_rows) / len(recommendation_rows) if recommendation_rows else None
    independent = truth.get("independent") is True and isinstance(truth.get("annotator"), str) and bool(truth["annotator"].strip())
    return {"original_records": len(ids), "groups_after_auto": len(groups), "automatic_reduction": len(ids) - len(groups),
            "automatic_edges": len(edges), "correct_edges": sum(item["correct"] for item in edges), "edge_precision": edge_precision,
            "group_pair_count": len(pairs), "group_pair_precision": pair_precision, "contaminated_or_unconfirmed_groups": contaminated,
            "recommendation_count": len(recommendations), "recommendation_record_count": len(view["recommendations"]),
            "recommendation_precision": recommendation_precision, "remaining_known_group_pairs": len(remaining),
            "recommendation_coverage": len(covered & remaining) / len(remaining) if remaining else None,
            "annotated_records": annotated, "truth_complete": annotated == len(ids),
            "independent_truth": independent,
            "hard_metrics_passed": independent and annotated == len(ids) and edge_precision is not None and edge_precision >= .99
                and pair_precision is not None and pair_precision >= .99 and len(groups) <= len(ids) // 2,
            "recommendations_passed": independent and annotated == len(ids) and recommendation_precision is not None and recommendation_precision >= .90,
            "edges": edges, "recommendations": recommendation_rows}
