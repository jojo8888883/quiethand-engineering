"""CPU-only synthetic interface example; no perception or video evaluation."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quiethand import EvidenceRecord, PatchConstraint, Provenance, compile_exact_outer_certificate
from quiethand.object_identity import parse_visual_binding, resolve_role_entity


def build_demo():
    event = {
        "event_id": "synthetic:identity-example",
        "object_state": {"entities": [
            {"entity_id": "plate", "mesh_path": "meshes/plate.obj"},
            {"entity_id": "bowl", "mesh_path": "meshes/bowl.obj"},
        ]},
    }
    # This is supplied fixture data, not a new answer obtained from a model.
    raw = json.dumps({
        "region_0": {"entity_id": "bowl", "reason": "人工构造示例：深碗"},
        "region_1": {"entity_id": None, "reason": "人工构造示例：不可确定"},
    }, ensure_ascii=False)
    binding = parse_visual_binding(raw, event)
    observed = EvidenceRecord(
        key="object_identity", status="derived", value=binding["role_to_entity"]["tool"],
        unit="entity_id", frame_id="camera", timestamp_s=1.0,
        provenance=Provenance("derived", "synthetic-interface-demo", "1", "local-demo"),
    )
    missing = EvidenceRecord(
        key="target_identity", status="ambiguous", value=None,
        unit="entity_id", frame_id="camera", timestamp_s=1.0,
        provenance=Provenance("derived", "synthetic-interface-demo", "1", "local-demo"),
        failure_reason="synthetic fixture deliberately leaves the second region unresolved",
    )
    patch = PatchConstraint(
        patch_id="synthetic-contact", hand_link="left_palm", surface_component="surface-0",
        point_m=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0),
    )
    certificate = compile_exact_outer_certificate([patch], ell_m=0.1)
    return {
        "example_kind": "synthetic_interface_only",
        "model_called": False,
        "说明": "演示接口复用，不是视频预测、真实力估计或新实验结果。",
        "role_to_entity": binding["role_to_entity"],
        "selected_mesh": resolve_role_entity(event, binding, "tool"),
        "unresolved_target": resolve_role_entity(event, binding, "target"),
        "evidence": [observed.to_dict(), missing.to_dict()],
        "synthetic_constraint": certificate.to_dict(),
    }


if __name__ == "__main__":
    print(json.dumps(build_demo(), ensure_ascii=False, indent=2, allow_nan=False))
