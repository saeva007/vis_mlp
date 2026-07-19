# Low-visibility data quality audit

This is a read-only audit. It does not rewrite raw NetCDF files or any
``X/y/dynamic/visibility`` arrays.

## PM policy

The authoritative policies are:

- ``pmst_canonical_units_v2_20260630``
- ``pm_explicit_legacy_scale_then_train_median_qc_v2_20260701``

The audit imports ``ifs_baseline/pmst_overlap_common.py`` and refuses to run if
those versions differ. ``canonicalize_pm_concentration()`` distinguishes raw
CAMS ``kg m-3``, physical ``ug m-3``, and historical ``kg m-3 * 1e12`` stored
values, then audits the common ``0..10000 ug m-3`` range. Historical arrays are
reported in both stored and canonical scales; they are never silently treated
as physical ``ug m-3`` values.

## Submit the complete audit

```bash
cd /public/home/putianshu/vis_mlp/train
mkdir -p logs /public/home/putianshu/vis_mlp/data_audits

RUN_TAG=lowvis_qc_$(date +%Y%m%d_%H%M%S)
AUDIT_DIR=/public/home/putianshu/vis_mlp/data_audits/${RUN_TAG}

sbatch --export=ALL,LOWVIS_AUDIT_OUT_DIR=${AUDIT_DIR} \
  sub_audit_lowvis_data_quality.slurm
```

Missing optional dataset directories are warnings. To require every default
directory and return a non-zero status when the report contains an error:

```bash
sbatch --export=ALL,LOWVIS_AUDIT_OUT_DIR=${AUDIT_DIR},LOWVIS_AUDIT_REQUIRE_ALL=1,LOWVIS_AUDIT_STRICT=1 \
  sub_audit_lowvis_data_quality.slurm
```

For a short trajectory raw/stored smoke comparison before the full scan:

```bash
sbatch --export=ALL,LOWVIS_AUDIT_OUT_DIR=${AUDIT_DIR}_smoke,LOWVIS_AUDIT_MAX_TRAJECTORY_RAW_ROWS=20000 \
  sub_audit_lowvis_data_quality.slurm
```

The most important files are:

- ``audit_summary.json`` and ``issues.csv``
- ``visibility_quality.csv``
- ``dynamic_feature_quality.csv`` and ``pm_quality.csv``
- ``trajectory_coverage_by_lead.csv``
- ``trajectory_raw_consistency.csv`` and mismatch examples
- ``raw_visibility_by_hour.csv`` and worst timestamps
- ``split_overlap.csv`` and ``metadata_quality.csv``

## Package the reports

The archive contains reports and provenance only, not the large source arrays.

```bash
cd /public/home/putianshu/vis_mlp/data_audits
tar -czf ${RUN_TAG}.tar.gz ${RUN_TAG}
sha256sum ${RUN_TAG}.tar.gz > ${RUN_TAG}.tar.gz.sha256
ls -lh ${RUN_TAG}.tar.gz ${RUN_TAG}.tar.gz.sha256
```

If the login shell no longer has ``RUN_TAG``, replace it with the exact audit
directory basename printed as ``[audit] output=...`` in the Slurm log.
