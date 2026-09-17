# Data contract

This document defines the intended internal data boundary. Raw LVK/PESummary ingestion is out of scope for the HBI core.

The design is informed by `gwcat`, but `gwpop-search` must not depend on private implementation details of that repository.

## Principles

- PE and selection products are separate typed objects.
- Both declare a coordinate basis.
- Both carry an explicit reference density in that basis.
- Required coordinates fail loudly when absent.
- Ragged PE sample counts are first-class.
- Multiple selection campaigns remain identifiable.
- Reference-density factors are applied exactly once.
- Provenance fields can grow later without changing the mathematical interface.

## PosteriorCatalog

Logical contents:

```
event_names:            [n_events]
offsets:                [n_events + 1]
samples/<coordinate>:   [n_total_samples]
log_ref_density:        [n_total_samples]
availability:           [n_events, n_coordinates]
metadata:               event/sample-set metadata
basis:                  CoordinateBasis
```

The event i slice is

```
offsets[i] : offsets[i + 1]
```

`log_ref_density` means the complete density that must be divided out for population reweighting **in the declared exported basis**. An adapter may compute it from a gwcat `p_pe` field via `log(p_pe)`; HBI code does not reconstruct the PE prior from release assumptions.

The initial internal API should expose:

```python
catalog.n_events
catalog.event_names
catalog.sample_count(i)
catalog.get_event(i, fields)
catalog.require(fields)
catalog.basis
```

## SelectionCatalog

Logical contents per detected/usable injection row:

```
samples/<coordinate>:   [n_selected]
log_draw_density:       [n_selected]
campaign_id:            [n_selected] or campaign slices
campaign metadata:
    n_draw
    observing_time
    detection rule metadata
    any estimator-normalization metadata required by the adapter
basis:                  CoordinateBasis
```

The exact canonical estimator interface will be fixed in Phase 1 using a parity test against the chosen gwcat export convention.

The HBI layer should receive one of two explicit representations, never infer which one it has:

1. **raw-draw representation:** true draw density + `n_draw` / exposure per campaign, from which the HBI engine builds the Monte Carlo selection estimator;
2. **estimator-ready representation:** an adapter-provided denominator/normalization whose semantics are complete and versioned (e.g. a validated gwcat-style `pdraw` product).

Mixing these representations is an error.

## CoordinateBasis

A basis identifier must include enough information to reject incompatible PE/selection pairs before inference.

Initial fields:

```
name
coordinates
frame conventions
spin parameterization
density measure/version
```

Examples might distinguish

```
source_m1_q_z_chieff
detector_m1_q_dL_chieff
source_m1_q_z_chieff_chip
component_spin_source
```

Do not rely on a free-form string alone; use a typed structure plus a stable serialized identity.

## Required-field contract

Population models declare their required coordinates. The loader checks those requirements against every event and the selection product.

A missing required quantity is a hard error naming:

- the coordinate;
- the event/campaign;
- the requested model/basis.

Do not silently NaN-drop events or selection rows.

## Support contract

Reference densities in a denominator must be finite and strictly positive wherever a retained PE sample or nonzero-weight selection sample is used.

Out-of-support rows may be represented only when the adapter's schema explicitly assigns them zero importance weight and records that convention. Generic HBI code must not invent density floors.

## gwcat compatibility direction

The read-only `gwcat` inspection showed several conventions worth preserving:

- concatenated PE columns + offsets;
- union parameter schema + availability mask;
- strict export requirements;
- first-class `SelectionSet` / `CombinedSelectionSet`;
- explicit spin-basis/reference-density semantics;
- paired PE/selection validation.

Phase 1 should write a gwcat adapter against a selected public/stable export API. It should not duplicate gwcat's raw release ingestion.

For a gwcat PE product, the adapter should treat exported `p_pe` as the reference density supplied by the data product.

For a gwcat selection product, the adapter should treat exported `pdraw` according to the exact documented schema/version and prove estimator parity with a direct calculation fixture before production use.

## Deferred provenance

Production manifests, release URLs, file checksums, event cuts, waveform policies, and injection campaign IDs are deliberately deferred until the interface is tested. When added, they belong in dataset manifests and metadata, not in the core HBI functions.
