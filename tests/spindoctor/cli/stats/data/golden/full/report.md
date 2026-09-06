# Navigation statistics report

Filters: none (every image read)

## Images selected

| instrument | images | first image | last image | first avail. date | last avail. date |
|---|---|---|---|---|---|
| coiss | 5 (62.5%) | N1294561202 | N1294564000 | 2005-05-22T02:12:16 | 2005-05-22T02:58:56 |
| sim | 1 (12.5%) | sim_scene_000042 | sim_scene_000042 | 2000-01-01T12:00:36 | 2000-01-01T12:00:36 |
| vgiss | 2 (25.0%) | C1385455 | C1385460 | 1979-02-01T14:39:10 | 1979-02-01T14:55:50 |

Total images: 8

## Files that yielded no record

A file named like a navigation document that no record could be read out of is counted here and
nowhere else: it records no instrument, no date and no image number, so this count covers the whole
of every selected root and none of the filters above narrows it. One kind of such file is counted
by a report over the documents and by no report from an index: one the storage could not deliver at
all. An ingest records no refusal for that, because a retrieval that failed once is worth trying
again rather than being remembered as a file that will not read.

Files that yielded no record: 0

## Success / failure

| status | coiss | sim | vgiss | total |
|---|---|---|---|---|
| success | 3 (60.0%) | 1 (100.0%) | 1 (50.0%) | 5 (62.5%) |
| error | 1 (20.0%) | 0 (0.0%) | 0 (0.0%) | 1 (12.5%) |
| failed | 1 (20.0%) | 0 (0.0%) | 1 (50.0%) | 2 (25.0%) |

![status](status_counts.png)

### Failure reasons

| status | reason | coiss | sim | vgiss | total |
|---|---|---|---|---|---|
| error | missing_spice_data | 1 (20.0%) | 0 (0.0%) | 0 (0.0%) | 1 (12.5%) |
| failed | all_features_gated | 1 (20.0%) | 0 (0.0%) | 0 (0.0%) | 1 (12.5%) |
| failed | no_features_extracted | 0 (0.0%) | 0 (0.0%) | 1 (50.0%) | 1 (12.5%) |

Examples (up to 5 per reason and instrument):

- missing_spice_data / coiss: N1294563000
- all_features_gated / coiss: N1294562000
- no_features_extracted / vgiss: C1385460

Full lists: filelists/failure_reason_missing_spice_data_coiss.txt, filelists/failure_reason_all_features_gated_coiss.txt, filelists/failure_reason_no_features_extracted_vgiss.txt

![failure reasons](failure_reasons.png)

## Failure taxonomy by image content

Failed images classified by what the feature inventory says was in
the scene.

| content | coiss | sim | vgiss | total |
|---|---|---|---|---|
| single-body | 1 (20.0%) | 0 (0.0%) | 0 (0.0%) | 1 (12.5%) |
| no-features | 1 (20.0%) | 0 (0.0%) | 1 (50.0%) | 2 (25.0%) |

| content | reason | coiss | sim | vgiss | total |
|---|---|---|---|---|---|
| single-body | all_features_gated | 1 (20.0%) | 0 (0.0%) | 0 (0.0%) | 1 (12.5%) |
| no-features | missing_spice_data | 1 (20.0%) | 0 (0.0%) | 0 (0.0%) | 1 (12.5%) |
| no-features | no_features_extracted | 0 (0.0%) | 0 (0.0%) | 1 (50.0%) | 1 (12.5%) |

Examples (up to 5 per content category and instrument):

- single-body / coiss: N1294562000
- no-features / coiss: N1294563000
- no-features / vgiss: C1385460

Full lists: filelists/failed_content_single-body_coiss.txt, filelists/failed_content_no-features_coiss.txt, filelists/failed_content_no-features_vgiss.txt

### Per-body failure shares

How often each named body appears in failed versus successful
images; a body with a high failure share is a modeling problem.

| body | instrument | failed images | successful images | failure share |
|---|---|---|---|---|
| IAPETUS | coiss | 1 (20.0%) | 2 (40.0%) | 0.333 |
| MIMAS | sim | 0 (0.0%) | 1 (100.0%) | 0.000 |

Examples (up to 5 per body and instrument):

- IAPETUS / coiss: N1294562000

Full lists: filelists/failed_body_IAPETUS_coiss.txt

## Technique usage

Images on which each technique ran.

| technique | coiss | sim | vgiss | total |
|---|---|---|---|---|
| BodyLimbNav | 1 (20.0%) | 1 (100.0%) | 0 (0.0%) | 2 (25.0%) |
| RingEdgeNav | 1 (20.0%) | 0 (0.0%) | 1 (50.0%) | 2 (25.0%) |
| BodyBlobNav | 1 (20.0%) | 0 (0.0%) | 0 (0.0%) | 1 (12.5%) |
| BodyDiscCorrelateNav | 1 (20.0%) | 0 (0.0%) | 0 (0.0%) | 1 (12.5%) |
| StarFieldFromCatalogNav | 1 (20.0%) | 0 (0.0%) | 0 (0.0%) | 1 (12.5%) |
| StarUniqueMatchNav | 1 (20.0%) | 0 (0.0%) | 0 (0.0%) | 1 (12.5%) |

### Per-technique detail

| technique | instrument | images | non-spurious | mean confidence |
|---|---|---|---|---|
| BodyLimbNav | coiss | 1 (20.0%) | 1 (100.0%) | 0.840 |
| BodyLimbNav | sim | 1 (100.0%) | 1 (100.0%) | 0.880 |
| RingEdgeNav | coiss | 1 (20.0%) | 1 (100.0%) | 0.740 |
| RingEdgeNav | vgiss | 1 (50.0%) | 1 (100.0%) | 0.660 |
| BodyBlobNav | coiss | 1 (20.0%) | 1 (100.0%) | 0.240 |
| BodyDiscCorrelateNav | coiss | 1 (20.0%) | 1 (100.0%) | 0.420 |
| StarFieldFromCatalogNav | coiss | 1 (20.0%) | 1 (100.0%) | 0.930 |
| StarUniqueMatchNav | coiss | 1 (20.0%) | 0 (0.0%) | 0.120 |

![technique usage](technique_usage.png)

## Model and source usage

Images in which each source appears.

| model | source | coiss | sim | vgiss | total |
|---|---|---|---|---|---|
| body:IAPETUS | IAPETUS | 3 (60.0%) | 0 (0.0%) | 0 (0.0%) | 3 (37.5%) |
| body:MIMAS | MIMAS | 0 (0.0%) | 1 (100.0%) | 0 (0.0%) | 1 (12.5%) |
| rings:SATURN | SATURN | 1 (20.0%) | 0 (0.0%) | 1 (50.0%) | 2 (25.0%) |
| stars | UCAC4 | 2 (40.0%) | 0 (0.0%) | 0 (0.0%) | 2 (25.0%) |

### Per-source feature counts

| model | source | instrument | features | gated |
|---|---|---|---|---|
| body:IAPETUS | IAPETUS | coiss | 7 | 2 |
| body:MIMAS | MIMAS | sim | 1 | 0 |
| rings:SATURN | SATURN | coiss | 2 | 1 |
| rings:SATURN | SATURN | vgiss | 2 | 1 |
| stars | UCAC4 | coiss | 14 | 2 |

## Offset statistics (successful images)

Grouped by camera: pointing errors of different cameras are unrelated
and are never pooled.  Percentages are of the instrument total.

| instrument | camera | axis | images | mean | median | stdev | min | max |
|---|---|---|---|---|---|---|---|---|
| coiss | NAC | dV | 2 (40.0%) | 31.375 | 31.375 | 39.775 | 3.250 | 59.500 |
| coiss | NAC | dU | 2 (40.0%) | -6.750 | -6.750 | 7.425 | -12.000 | -1.500 |
| coiss | WAC | dV | 1 (20.0%) | 0.350 | 0.350 | 0.000 | 0.350 | 0.350 |
| coiss | WAC | dU | 1 (20.0%) | -0.120 | -0.120 | 0.000 | -0.120 | -0.120 |
| sim | SIM | dV | 1 (100.0%) | 1.500 | 1.500 | 0.000 | 1.500 | 1.500 |
| sim | SIM | dU | 1 (100.0%) | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 |
| vgiss | NAC | dV | 1 (50.0%) | -2.750 | -2.750 | 0.000 | -2.750 | -2.750 |
| vgiss | NAC | dU | 1 (50.0%) | 4.500 | 4.500 | 0.000 | 4.500 | 4.500 |

![offsets coiss NAC](offsets_hist_coiss_NAC.png)

![offsets coiss WAC](offsets_hist_coiss_WAC.png)

![offsets sim SIM](offsets_hist_sim_SIM.png)

![offsets vgiss NAC](offsets_hist_vgiss_NAC.png)

### By instrument, camera, and image size

| instrument | camera | size (v x u) | images | dV mean | dV stdev | dV min | dV max | dU mean | dU stdev | dU min | dU max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| coiss | NAC | 1024x1024 | 2 (40.0%) | 31.375 | 39.775 | 3.250 | 59.500 | -6.750 | 7.425 | -12.000 | -1.500 |
| coiss | WAC | 512x512 | 1 (20.0%) | 0.350 | 0.000 | 0.350 | 0.350 | -0.120 | 0.000 | -0.120 | -0.120 |
| sim | SIM | 256x256 | 1 (100.0%) | 1.500 | 0.000 | 1.500 | 1.500 | 0.500 | 0.000 | 0.500 | 0.500 |
| vgiss | NAC | 800x800 | 1 (50.0%) | -2.750 | 0.000 | -2.750 | -2.750 | 4.500 | 0.000 | 4.500 | 4.500 |

## Suspect offsets (near the search limit)

Successful images whose fused offset reaches at least 0.90 of the per-axis maximum expected pointing offset (the configured extfov search margin) on either axis.  These offsets may be correlation artifacts pinned to the search boundary.

Suspect images: 1 (12.5%) of 4 screened.

| category | coiss | sim | vgiss | total |
|---|---|---|---|---|
| suspect | 1 (20.0%) | 0 (0.0%) | 0 (0.0%) | 1 (12.5%) |

| image | instrument | dV | dU | magnitude | limit (v, u) |
|---|---|---|---|---|---|
| N1294564000 | coiss | 59.500 | -12.000 | 60.698 | (50.0, 140.0) |

Examples (up to 5 per category and instrument):

- suspect / coiss: N1294564000

Full lists: filelists/suspect_offsets_suspect_coiss.txt

Search limit could not be resolved for some images:

- vgiss: 'voyager_iss' has no extfov_margin_vu entry for image size 800 (1 image(s))

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
| coiss | BodyBlobNav vs BodyDiscCorrelateNav | 1 (20.0%) | 26.504 | 26.504 |
| coiss | BodyLimbNav vs StarFieldFromCatalogNav | 1 (20.0%) | 0.262 | 0.262 |

## Confidence calibration (agreement as accuracy proxy)

For each confidence tier: how well the techniques that fed the fused
offset agreed with one another.  Without ground truth, cross-technique
agreement is the standing production check that confidence tiers are
meaningful (the calibrated anchor is the simulated-scene campaign).

| tier | coiss | sim | vgiss | total |
|---|---|---|---|---|
| high | 1 (20.0%) | 1 (100.0%) | 0 (0.0%) | 2 (25.0%) |
| medium | 1 (20.0%) | 0 (0.0%) | 1 (50.0%) | 2 (25.0%) |
| low | 1 (20.0%) | 0 (0.0%) | 0 (0.0%) | 1 (12.5%) |
| failed | 1 (20.0%) | 0 (0.0%) | 1 (50.0%) | 2 (25.0%) |
| conflicted | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |

| tier | instrument | images | with >=2 techniques | median max-disagreement (px) | p95 (px) |
|---|---|---|---|---|---|
| high | coiss | 1 (20.0%) | 1 (100.0%) | 0.262 | 0.262 |
| high | sim | 1 (100.0%) | 0 (0.0%) | - | - |
| high | vgiss | 0 (0.0%) | 0 (0.0%) | - | - |
| medium | coiss | 1 (20.0%) | 0 (0.0%) | - | - |
| medium | sim | 0 (0.0%) | 0 (0.0%) | - | - |
| medium | vgiss | 1 (50.0%) | 0 (0.0%) | - | - |
| low | coiss | 1 (20.0%) | 1 (100.0%) | 26.504 | 26.504 |
| low | sim | 0 (0.0%) | 0 (0.0%) | - | - |
| low | vgiss | 0 (0.0%) | 0 (0.0%) | - | - |
| failed | coiss | 1 (20.0%) | 0 (0.0%) | - | - |
| failed | sim | 0 (0.0%) | 0 (0.0%) | - | - |
| failed | vgiss | 1 (50.0%) | 0 (0.0%) | - | - |
| conflicted | coiss | 0 (0.0%) | 0 (0.0%) | - | - |
| conflicted | sim | 0 (0.0%) | 0 (0.0%) | - | - |
| conflicted | vgiss | 0 (0.0%) | 0 (0.0%) | - | - |

![agreement by tier](agreement_by_tier.png)

## Ensemble outlier exclusions

| excluded techniques | coiss | sim | vgiss | total |
|---|---|---|---|---|
| BodyBlobNav | 1 (20.0%) | 0 (0.0%) | 0 (0.0%) | 1 (12.5%) |

Examples (up to 5 per exclusion set and instrument):

- BodyBlobNav / coiss: N1294564000

Full lists: filelists/excluded_BodyBlobNav_coiss.txt

## Run-time statistics

| instrument | images | total (s) | min (s) | max (s) | mean (s) | median (s) | stdev (s) |
|---|---|---|---|---|---|---|---|
| coiss | 5 (100.0%) | 66.500 | 1.500 | 31.750 | 13.300 | 12.500 | 11.252 |
| sim | 1 (100.0%) | 12.500 | 12.500 | 12.500 | 12.500 | 12.500 | 0.000 |
| vgiss | 2 (100.0%) | 20.750 | 8.250 | 12.500 | 10.375 | 10.375 | 3.005 |
| (all) | 8 (100.0%) | 99.750 | 1.500 | 31.750 | 12.469 | 12.500 | 8.682 |

![run time](runtime_hist.png)

Slowest 5 image(s):

| image | instrument | elapsed (s) |
|---|---|---|
| N1294564000 | coiss | 31.750 |
| C1385455 | vgiss | 12.500 |
| N1294561202 | coiss | 12.500 |
| W1294561202 | coiss | 12.500 |
| sim_scene_000042 | sim | 12.500 |

## Peak-memory statistics

| instrument | images | min (GB) | max (GB) | mean (GB) | median (GB) | stdev (GB) |
|---|---|---|---|---|---|---|
| coiss | 5 (100.0%) | 0.250 | 8.000 | 3.350 | 2.000 | 3.130 |
| sim | 1 (100.0%) | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| vgiss | 2 (100.0%) | 2.500 | 3.000 | 2.750 | 2.750 | 0.354 |
| (all) | 8 (100.0%) | 0.250 | 8.000 | 2.906 | 2.250 | 2.507 |

![peak memory](memory_hist.png)

Hungriest 5 image(s):

| image | instrument | peak (GB) |
|---|---|---|
| N1294564000 | coiss | 8.000 |
| N1294561202 | coiss | 5.000 |
| C1385455 | vgiss | 3.000 |
| C1385460 | vgiss | 2.500 |
| W1294561202 | coiss | 2.000 |

## CSV export

One row per image: images.csv (8 row(s)).

