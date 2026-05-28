# Quarkdown white-paper draft bundle

This bundle contains a working paper draft authored in Quarkdown source format.

## Main file

- `main.qd` - paged Quarkdown manuscript source.

## Local data and figures

- `data/verified_initial_results.csv`
- `data/context_scaling.csv`
- `data/batch_scaling.csv`
- `data/cuda_oxide_inspection_gate.csv`
- `source_reports/cuda_oxide_inspection_gate_result.txt`
- `assets/context_scaling.svg` and `.png`
- `assets/batch_scaling.svg` and `.png`

## Compile with Quarkdown

Quarkdown's official repository documents the `paged` document type and the following CLI form for compilation:

```bash
quarkdown c main.qd --pdf
```

A normal HTML build can be generated with:

```bash
quarkdown c main.qd
```

From the bundle root:

```bash
cd local_llm_framework_whitepaper_quarkdown
quarkdown c main.qd --pdf
```

## Evidence note

The initial Phase 3 and verification values are based on supplied local reports. The AWQ-Marlin context and batch tables are based on the author's subsequent local result summary. The updated draft also incorporates the supplied cuda-oxide inspection-gate result, which reports that the proposed Ampere AWQ quantized-linear experiment was halted before implementation because the inspected cuda-oxide v0.1.0 capability surface did not expose an Ampere tensor-core abstraction required for a credible Marlin competitor. The raw follow-up benchmark artifacts, inspected repository revisions and source-location evidence should be included in a publication repository before external submission.

## Rendering status in this delivery

The Quarkdown `.qd` source and figure/data bundle are complete. A Quarkdown-generated PDF is not included in this delivery because the Quarkdown executable distribution could not be installed in the available execution container. Compile the source with Quarkdown v2.1.2 or a pinned later version in the publication repository.


## Revision included in this bundle

This revision adds a dedicated inspection-gate section, updates the abstract and conclusion, records the `ABANDON_PROPOSED_KERNEL_TARGET` decision, and distinguishes a reported toolchain capability blocker from an unmeasured performance prediction. No cuda-oxide AWQ kernel result is claimed because no such kernel was implemented or benchmarked.
