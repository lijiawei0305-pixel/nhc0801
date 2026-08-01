"""Dataset path conventions, development split, and weighted NPZ readers."""

from nhc_deprot.data.development_split import (
    DevelopmentSplit,
    SealedFinalTestCommitment,
    load_development_split,
    load_packaged_v004_day1_split,
)
from nhc_deprot.data.errors import DatasetError
from nhc_deprot.data.paths import (
    PARENT_PROTOCOL_SHA256,
    SEALED_FINAL_TEST_COMMITMENT_SHA256,
    TRAIN_ROOTS,
    VALIDATION_ROOTS,
    autofill_run_dir,
    frame_path,
)
from nhc_deprot.data.teacher_frames import (
    TeacherFrameRef,
    inventory_candidates,
    list_candidate_frame_refs,
)
from nhc_deprot.data.weight_policy import (
    assign_candidate_endpoint_weights,
    audit_split_weight_sums,
)
from nhc_deprot.data.weighted_dataset import (
    REQUIRED_ARRAYS,
    WeightedDatasetAudit,
    audit_public_weighted_result,
    audit_weighted_dataset,
    default_v004_weighted_dataset_root,
    load_split_sample_weights,
)

__all__ = [
    "DatasetError",
    "DevelopmentSplit",
    "PARENT_PROTOCOL_SHA256",
    "REQUIRED_ARRAYS",
    "SEALED_FINAL_TEST_COMMITMENT_SHA256",
    "SealedFinalTestCommitment",
    "TRAIN_ROOTS",
    "TeacherFrameRef",
    "VALIDATION_ROOTS",
    "WeightedDatasetAudit",
    "assign_candidate_endpoint_weights",
    "audit_public_weighted_result",
    "audit_split_weight_sums",
    "audit_weighted_dataset",
    "autofill_run_dir",
    "default_v004_weighted_dataset_root",
    "frame_path",
    "inventory_candidates",
    "list_candidate_frame_refs",
    "load_development_split",
    "load_packaged_v004_day1_split",
    "load_split_sample_weights",
]
