"""Deterministic per-student parameter salting for CS482 Milestone 0."""
import hashlib
import re


def get_student_params(student_id: str) -> dict:
    """Return the exact parameters specified in the CS482 practicum handout."""
    student_id = str(student_id).strip()
    if not re.fullmatch(r"\d{8,}", student_id):
        raise ValueError("student_id must contain at least 8 digits")

    h = int(hashlib.sha256(student_id.encode("utf-8")).hexdigest()[:8], 16)
    n = 1024 + ((h % 9) * 64)
    buffer_elements = 16_777_216
    seed = h % 100_007

    return {
        "student_id": student_id,
        "N": n,
        "buffer_elements": buffer_elements,
        "seed": seed,
    }
