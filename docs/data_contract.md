# Data contract

This is the implemented boundary between release-specific products and the HBI engine. Raw LVK/PESummary ingestion is intentionally outside `gwpop-search`.

## Core rule

The HBI layer never guesses a PE prior, injection draw density, coordinate Jacobian, or selection normalization from a release name. Adapters provide explicit denominator densities and an explicit density basis.

## CoordinateBasis

`CoordinateBasis` identifies the measure a density is expressed against. Its stable identity is derived from:

- name;
- ordered independent coordinates;
- frame convention;
- spin parameterization;
- density-measure description;
- contract version.

PE and selection products must have exactly matching basis identities before inference.

Derived/advisory sample columns may exist without being independent density coordinates.

## PosteriorCatalog

Logical contents:

```
event_names:            [n_events]
offsets:                [n_events + 1]
samples/<coordinate>:   [n_total_samples]
log_ref_density:        [n_total_samples]
availability:           [n_events, n_stored_fields]
basis:                  CoordinateBasis
metadata:               mapping
```

The event-i slice is `offsets[i]:offsets[i+1]`.

`log_ref_density` is the complete PE denominator density supplied by the adapter in the declared basis. It must be finite on every retained sample. Zero-density samples are not repaired with floors.

The container supports ragged event sample counts and stores the union of sample fields, with per-event availability checks.

## SelectionCatalog

Logical contents:

```
samples/<coordinate>:   [n_selected]
log_draw_density:       [n_selected]
campaign_id:            [n_selected]
campaigns:              typed per-campaign metadata
basis:                  CoordinateBasis
mode:                   raw_draw | estimator_ready
estimator_semantics:    required for estimator_ready
```

Two modes are deliberately distinct.

### raw_draw

The denominator is the actual draw density for each campaign. Every campaign must carry `n_draw`; Phase 2 will construct the Monte Carlo selection estimator explicitly from the campaign metadata.

### estimator_ready

The adapter-provided denominator already encodes a complete estimator convention. The HBI engine must use those semantics directly and must not automatically apply an additional `1/n_draw`, observing-time factor, or campaign-mixture factor.

The mode is a typed enum so these two cases cannot be silently mixed.

## Reference-density support

Any denominator used for a retained nonzero-weight sample must be finite and strictly positive. Generic HBI code never introduces numerical density floors.

An external schema that intentionally encodes zero-importance rows must get a dedicated adapter treatment; that convention is not inferred from large or small numbers.

## Internal round-trip format

Phase 1 defines simple internal HDF5 formats:

- `gwpop-search-pe-1.0`
- `gwpop-search-selection-1.0`

These are implementation/test formats, not frozen public release manifests. Production provenance remains deferred.

## gwcat v2 adapter

Reference inspected read-only:

```
repository: ignaciomagana/gwcat
branch: master
commit: 8f9e2f12b499a6b2bf16ed938f66d020b12c44c2
```

The adapter consumes exported `gwcat-pe-2.0/2.1` and `gwcat-selection-2.0/2.1` products. It does not ingest raw PE files.

Supported Phase-1 spaces:

- `chieff`
- `chieff_chip`
- `component`

The current gwcat registry defines the fitted core as:

```
m1det, q, dL, ra, dec
```

plus the selected independent spin coordinates. Therefore the canonical gwcat adapter basis includes sky; source-frame masses, redshift, `m2det`, and derived spin quantities are advisory columns unless they belong to the chosen independent spin basis.

For PE:

```
log_ref_density = log(p_pe)
```

with no reconstruction of the prior.

For selection:

```
log_draw_density = log(pdraw)
mode = estimator_ready
```

The exported gwcat `pdraw` already contains its documented campaign-mixture/exposure convention. The adapter preserves it exactly. A Phase-1 parity test verifies that `sum(p_pop / pdraw)` is unchanged by adaptation and specifically verifies that the adapter does not divide by `ndraw` again.

The component space may expose `chi_eff` as a useful derived sample column, but `chi_eff` is not included in the component density basis.

### Audited canonicalization

Reviewed gwcat-v2 exports are converted to the internal HDF5 pair with an
explicit spin-basis requirement:

```bash
gwpop-search canonicalize-gwcat-v2 \
  --pe-export /path/to/gwcat_pe.h5 \
  --selection-export /path/to/gwcat_selection.h5 \
  --spin-basis chieff \
  --output-dir /frozen/canonical
```

The output directory contains:

```text
pe.h5
selection.h5
canonicalization_report.json
```

The report hashes both source exports and both canonical outputs, records the
basis identity, event ordering, PE/selection counts, adapter metadata,
selection mode/semantics, and the denominator contract. Canonicalization is
idempotent only for the exact same source hashes and explicit spin basis.
Partial or conflicting pre-existing outputs fail instead of being overwritten.

No PE prior or selection density is reconstructed during this step:
`p_pe` and estimator-ready `pdraw` remain authoritative.

## Pair validation

`validate_pair(pe, selection, required_coordinates=...)` checks:

1. exact basis compatibility;
2. required PE coordinates are present and available for every event;
3. required selection coordinates are present.

Any mismatch fails before likelihood evaluation.

## Deferred provenance

Production dataset manifests, release URLs, checksums, final GWTC-5 event cuts, waveform policies, and injection campaign selections remain intentionally deferred until the HBI engine is validated.
