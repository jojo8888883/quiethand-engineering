"""Bind semantic regions to physical CAD entities before estimating poses.

Semantic role names never select a mesh. A role's pose remains in the coordinate
frame of its explicitly named CAD entity, which must travel with the pose.
"""

import json


ROLES = ("tool", "target")


class ObjectIdentityError(ValueError):
    pass


def entity_catalog(object_state):
    entities = object_state["entities"]
    catalog = {item["entity_id"]: item for item in entities}
    if not catalog or len(catalog) != len(entities):
        raise ObjectIdentityError("CAD identities must be nonempty and unique")
    for entity_id, item in catalog.items():
        if not isinstance(entity_id, str) or not entity_id or not item.get("mesh_path"):
            raise ObjectIdentityError("CAD entity needs identity and metric mesh path")
    return catalog


def parse_visual_binding(raw, event):
    """The VLM sees region numbers, never native tool/target or action labels."""
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"region_0", "region_1"}:
        raise ObjectIdentityError("expected region_0 and region_1")
    catalog = entity_catalog(event["object_state"])
    assignments = {}
    reasons = {}
    for index, role in enumerate(ROLES):
        region = value[f"region_{index}"]
        if not isinstance(region, dict) or set(region) != {"entity_id", "reason"}:
            raise ObjectIdentityError("region needs entity_id and reason")
        identity = region["entity_id"]
        if identity is not None and (not isinstance(identity, str) or identity not in catalog):
            raise ObjectIdentityError("model named an unknown CAD entity")
        if not isinstance(region["reason"], str) or not region["reason"].strip():
            raise ObjectIdentityError("model must explain its visual match or uncertainty")
        assignments[role] = identity
        reasons[role] = region["reason"]
    bound = [v for v in assignments.values() if v is not None]
    if len(bound) != len(set(bound)):
        raise ObjectIdentityError("two regions cannot bind the same physical entity")
    return {"event_id": event["event_id"], "role_to_entity": assignments,
            "reasons": reasons, "source": "qwen_rgb_cad_visual_match",
            "native_pose_used": False, "human_corrections_used": False}


def resolve_role_entity(event, binding, role):
    if binding["event_id"] != event["event_id"]:
        raise ObjectIdentityError("binding belongs to another event")
    catalog = entity_catalog(event["object_state"])
    assignments = binding["role_to_entity"]
    if set(assignments) != set(ROLES):
        raise ObjectIdentityError("binding must explicitly include both semantic regions")
    bound = [v for v in assignments.values() if v is not None]
    if any(not isinstance(v, str) or v not in catalog for v in bound) or len(bound) != len(set(bound)):
        raise ObjectIdentityError("invalid or duplicate entity assignment")
    identity = assignments[role]
    return None if identity is None else catalog[identity]
