"""Framework-free helpers shared between the legacy and modern UIs."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from queue import Queue


_DIGITS_RE = re.compile(r"^[0-9]+$")


def is_a_number(string: str) -> bool:
    return bool(_DIGITS_RE.match(string))


def empty_queue(q: Queue) -> None:
    while not q.empty():
        item = q.get()
        logging.debug(f"Emptying queue: Got {item[0]}")


def sort_nested_json(data):
    if isinstance(data, dict):
        return {k: sort_nested_json(data[k]) for k in sorted(data)}
    if isinstance(data, list):
        return [sort_nested_json(item) for item in data]
    return data


def generate_dict_hash(dictionary: dict) -> str:
    serialized = json.dumps(dictionary, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def find_closest(sorted_list, target):
    if not sorted_list:
        return None
    left, right = 0, len(sorted_list) - 1
    if target <= sorted_list[left]:
        return sorted_list[left]
    if target >= sorted_list[right]:
        return sorted_list[right]
    while left <= right:
        mid = (left + right) // 2
        if sorted_list[mid] == target:
            return sorted_list[mid]
        if sorted_list[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    if abs(sorted_list[left] - target) < abs(sorted_list[right] - target):
        return sorted_list[left]
    return sorted_list[right]
