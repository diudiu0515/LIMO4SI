"""Deterministic multi-human identity components."""

from .global_assignment import AssignmentPolicy, AssignmentResult, assign_global_identities
from .quality import validate_identity_result

__all__ = ["AssignmentPolicy", "AssignmentResult", "assign_global_identities", "validate_identity_result"]
