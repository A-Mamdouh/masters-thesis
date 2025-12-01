from __future__ import annotations

from typing import Dict, List, Tuple



SEM_TYPES: Dict[str, str] = {
    "human": "sType_human",
    "location": "sType_location",
    "occasion": "sType_occasion",
    "object": "sType_object",
    "unknown": "sType_unknown",
}

INDIVIDUAL_TYPES: Dict[str, str] = {
    "alice": SEM_TYPES["human"],
    "bob": SEM_TYPES["human"],
    "charlie": SEM_TYPES["human"],
    "david": SEM_TYPES["human"],
    "john": SEM_TYPES["human"],
    "unknown_person": SEM_TYPES["human"],
    "cake": SEM_TYPES["object"],
    "unknown_object": SEM_TYPES["object"],
    "home": SEM_TYPES["location"],
    "park": SEM_TYPES["location"],
    "unknown_location": SEM_TYPES["location"],
    "birthday": SEM_TYPES["occasion"],
    "newyearseve": SEM_TYPES["occasion"],
    "unknown_occasion": SEM_TYPES["occasion"],
}

SITUATION_BY_VERB: Dict[str, Tuple[str, ...]] = {
    "celebrate": ("situation_celebration",),
    "bring": ("situation_celebration",),
    "attend": ("situation_celebration",),
    "be_at": ("situation_celebration", "situation_murder"),
    "murder": ("situation_murder",),
    "investigate": ("situation_murder",),
}

FRAME_BY_VERB: Dict[str, str] = {
    "celebrate": "frame_celebrate",
    "bring": "frame_bring",
    "attend": "frame_attend",
    "be_at": "frame_be_at",
    "murder": "frame_murder",
    "investigate": "frame_investigate",
}

ROLE_SPECS: Dict[str, List[Tuple[str, str]]] = {
    "celebrate": [
        ("role_occasion", "occasion"),
        ("role_location", "location"),
    ],
    "attend": [
        ("role_attendee", "human"),
        ("role_occasion", "occasion"),
    ],
    "be_at": [
        ("role_attendee", "human"),
        ("role_location", "location"),
    ],
    "bring": [
        ("role_bringer", "human"),
        ("role_brought", "object"),
        ("role_location", "location"),
    ],
    "murder": [
        ("role_murderer", "human"),
        ("role_victim", "human"),
        ("role_location", "location"),
    ],
    "investigate": [
        ("role_investigator", "human"),
        ("role_suspect", "human"),
        ("role_victim", "human"),
        ("role_location", "location"),
    ],
}


def get_sem_type(name: str, expected: str | None = None) -> str:
    normalized = name.lower()
    if normalized in INDIVIDUAL_TYPES:
        return INDIVIDUAL_TYPES[normalized]
    if expected and expected in SEM_TYPES:
        return SEM_TYPES[expected]
    return SEM_TYPES["unknown"]


__all__ = [
    "SEM_TYPES",
    "INDIVIDUAL_TYPES",
    "SITUATION_BY_VERB",
    "FRAME_BY_VERB",
    "ROLE_SPECS",
    "get_sem_type",
]
