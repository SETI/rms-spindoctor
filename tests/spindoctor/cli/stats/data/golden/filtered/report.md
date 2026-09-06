# Navigation statistics report

Filters: instrument = coiss, image number >= 1294561202, image number <= 1294563000

## Images selected

| instrument | images | first image | last image | first avail. date | last avail. date |
|---|---|---|---|---|---|
| coiss | 4 (100.0%) | N1294561202 | N1294563000 | 2005-05-22T02:12:16 | 2005-05-22T02:25:36 |

Total images: 4

## Files that yielded no record

A file named like a navigation document that no record could be read out of is counted here and
nowhere else: it records no instrument, no date and no image number, so this count covers the whole
of every selected root and none of the filters above narrows it. One kind of such file is counted
by a report over the documents and by no report from an index: one the storage could not deliver at
all. An ingest records no refusal for that, because a retrieval that failed once is worth trying
again rather than being remembered as a file that will not read.

Files that yielded no record: 0

## Success / failure

| status | coiss | total |
|---|---|---|
| success | 2 (50.0%) | 2 (50.0%) |
| error | 1 (25.0%) | 1 (25.0%) |
| failed | 1 (25.0%) | 1 (25.0%) |

![status](status_counts.png)

### Failure reasons

| status | reason | coiss | total |
|---|---|---|---|
| error | missing_spice_data | 1 (25.0%) | 1 (25.0%) |
| failed | all_features_gated | 1 (25.0%) | 1 (25.0%) |

Examples (up to 3 per reason and instrument):

- missing_spice_data / coiss: N1294563000
- all_features_gated / coiss: N1294562000

![failure reasons](failure_reasons.png)

## Failure taxonomy by image content

Failed images classified by what the feature inventory says was in
the scene.

| content | coiss | total |
|---|---|---|
| single-body | 1 (25.0%) | 1 (25.0%) |
| no-features | 1 (25.0%) | 1 (25.0%) |

| content | reason | coiss | total |
|---|---|---|---|
| single-body | all_features_gated | 1 (25.0%) | 1 (25.0%) |
| no-features | missing_spice_data | 1 (25.0%) | 1 (25.0%) |

Examples (up to 3 per content category and instrument):

- single-body / coiss: N1294562000
- no-features / coiss: N1294563000

### Per-body failure shares

How often each named body appears in failed versus successful
images; a body with a high failure share is a modeling problem.

| body | instrument | failed images | successful images | failure share |
|---|---|---|---|---|
| IAPETUS | coiss | 1 (25.0%) | 1 (25.0%) | 0.500 |

Examples (up to 3 per body and instrument):

- IAPETUS / coiss: N1294562000

## Technique usage

Images on which each technique ran.

| technique | coiss | total |
|---|---|---|
| BodyLimbNav | 1 (25.0%) | 1 (25.0%) |
| RingEdgeNav | 1 (25.0%) | 1 (25.0%) |
| StarFieldFromCatalogNav | 1 (25.0%) | 1 (25.0%) |

### Per-technique detail

| technique | instrument | images | non-spurious | mean confidence |
|---|---|---|---|---|
| BodyLimbNav | coiss | 1 (25.0%) | 1 (100.0%) | 0.840 |
| RingEdgeNav | coiss | 1 (25.0%) | 1 (100.0%) | 0.740 |
| StarFieldFromCatalogNav | coiss | 1 (25.0%) | 1 (100.0%) | 0.930 |

![technique usage](technique_usage.png)

## Model and source usage

Images in which each source appears.

| model | source | coiss | total |
|---|---|---|---|
| body:IAPETUS | IAPETUS | 2 (50.0%) | 2 (50.0%) |
| rings:SATURN | SATURN | 1 (25.0%) | 1 (25.0%) |
| stars | UCAC4 | 1 (25.0%) | 1 (25.0%) |

### Per-source feature counts

| model | source | instrument | features | gated |
|---|---|---|---|---|
| body:IAPETUS | IAPETUS | coiss | 4 | 2 |
| rings:SATURN | SATURN | coiss | 2 | 1 |
| stars | UCAC4 | coiss | 7 | 1 |

## Offset statistics (successful images)

Grouped by camera: pointing errors of different cameras are unrelated
and are never pooled.  Percentages are of the instrument total.

| instrument | camera | axis | images | mean | median | stdev | min | max |
|---|---|---|---|---|---|---|---|---|
| coiss | NAC | dV | 1 (25.0%) | 3.250 | 3.250 | 0.000 | 3.250 | 3.250 |
| coiss | NAC | dU | 1 (25.0%) | -1.500 | -1.500 | 0.000 | -1.500 | -1.500 |
| coiss | WAC | dV | 1 (25.0%) | 0.350 | 0.350 | 0.000 | 0.350 | 0.350 |
| coiss | WAC | dU | 1 (25.0%) | -0.120 | -0.120 | 0.000 | -0.120 | -0.120 |

![offsets coiss NAC](offsets_hist_coiss_NAC.png)

![offsets coiss WAC](offsets_hist_coiss_WAC.png)

### By instrument, camera, and image size

| instrument | camera | size (v x u) | images | dV mean | dV stdev | dV min | dV max | dU mean | dU stdev | dU min | dU max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| coiss | NAC | 1024x1024 | 1 (25.0%) | 3.250 | 0.000 | 3.250 | 3.250 | -1.500 | 0.000 | -1.500 | -1.500 |
| coiss | WAC | 512x512 | 1 (25.0%) | 0.350 | 0.000 | 0.350 | 0.350 | -0.120 | 0.000 | -0.120 | -0.120 |

## Suspect offsets (near the search limit)

Successful images whose fused offset reaches at least 0.90 of the per-axis maximum expected pointing offset (the configured extfov search margin) on either axis.  These offsets may be correlation artifacts pinned to the search boundary.

Suspect images: 0 (0.0%) of 2 screened.

| category | coiss | total |
|---|---|---|
| suspect | 0 (0.0%) | 0 (0.0%) |

## BOTSIM pair consistency (Cassini ISS)

BOTSIM observations shutter the NAC and WAC simultaneously (the image
names share one spacecraft-clock count).  One WAC pixel is ten NAC
pixels, so a consistent pair satisfies NAC offset ~= 10 x WAC offset
per axis.  Residuals below are NAC - 10 x WAC, in NAC pixels.

| metric | value |
|---|---|
| pairs identified | 1 |
| pairs with both navigated | 1 |
| median residual (px) | 0.391 |
| p95 residual (px) | 0.391 |

Worst 1 pair(s):

| clock | NAC image | WAC image | residual dV | residual dU | residual |
|---|---|---|---|---|---|
| 1294561202 | N1294561202 | W1294561202 | -0.250 | -0.300 | 0.391 |

## Cross-technique agreement

Euclidean distance between per-technique offsets on images where both
techniques produced non-spurious results.

| instrument | technique pair | images | median (px) | p95 (px) |
|---|---|---|---|---|
| coiss | BodyLimbNav vs StarFieldFromCatalogNav | 1 (25.0%) | 0.262 | 0.262 |

## Confidence calibration (agreement as accuracy proxy)

For each confidence tier: how well the techniques that fed the fused
offset agreed with one another.  Without ground truth, cross-technique
agreement is the standing production check that confidence tiers are
meaningful (the calibrated anchor is the simulated-scene campaign).

| tier | coiss | total |
|---|---|---|
| high | 1 (25.0%) | 1 (25.0%) |
| medium | 1 (25.0%) | 1 (25.0%) |
| low | 0 (0.0%) | 0 (0.0%) |
| failed | 1 (25.0%) | 1 (25.0%) |
| conflicted | 0 (0.0%) | 0 (0.0%) |

| tier | instrument | images | with >=2 techniques | median max-disagreement (px) | p95 (px) |
|---|---|---|---|---|---|
| high | coiss | 1 (25.0%) | 1 (100.0%) | 0.262 | 0.262 |
| medium | coiss | 1 (25.0%) | 0 (0.0%) | - | - |
| low | coiss | 0 (0.0%) | 0 (0.0%) | - | - |
| failed | coiss | 1 (25.0%) | 0 (0.0%) | - | - |
| conflicted | coiss | 0 (0.0%) | 0 (0.0%) | - | - |

![agreement by tier](agreement_by_tier.png)

## Run-time statistics

| instrument | images | total (s) | min (s) | max (s) | mean (s) | median (s) | stdev (s) |
|---|---|---|---|---|---|---|---|
| coiss | 4 (100.0%) | 34.750 | 1.500 | 12.500 | 8.688 | 10.375 | 5.194 |

![run time](runtime_hist.png)

Slowest 3 image(s):

| image | instrument | elapsed (s) |
|---|---|---|
| N1294561202 | coiss | 12.500 |
| W1294561202 | coiss | 12.500 |
| N1294562000 | coiss | 8.250 |

## Peak-memory statistics

| instrument | images | min (GB) | max (GB) | mean (GB) | median (GB) | stdev (GB) |
|---|---|---|---|---|---|---|
| coiss | 4 (100.0%) | 0.250 | 5.000 | 2.188 | 1.750 | 2.014 |

![peak memory](memory_hist.png)

Hungriest 3 image(s):

| image | instrument | peak (GB) |
|---|---|---|
| N1294561202 | coiss | 5.000 |
| W1294561202 | coiss | 2.000 |
| N1294562000 | coiss | 1.500 |

## CSV export

One row per image: images.csv (4 row(s)).

