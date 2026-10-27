Simulator scene catalog
=======================

Each YAML here is a synthetic frame the navigator can be run against, laid out
as:

    sim_scenes/<scene_class>/<scene_name>.yaml

The directory is the registry: <scene_class> is the immediate parent directory
and must be one of the declared classes; <scene_name> must equal the filename
stem.  The schema is defined and validated by src/spindoctor/sim/scene.py; the
structural invariants are enforced by tests/integration/test_sim_scenes.py.

The fields are the flat runtime sim_params names the renderer consumes, so a
validated scene file is the sim_params dict directly (load_sim_scene returns it
unchanged; schema_version and scene_name are metadata the renderer ignores).

This mirrors the operator-curated image library (images/README.txt) but the
scenes are generated on demand by the simulator, so they augment -- never
replace -- the real-data calibration target.

Scene classes
-------------

  phase_sweep_regular_body    - an ellipsoid body at varying phase angle
  phase_sweep_irregular_body  - a mesh (irregular) body at varying phase angle
  noise_sweep                 - a fixed scene at varying read-noise level
  smear_sweep                 - varying star smear
  range_sweep                 - a body at varying apparent size / distance
  multi_body_geometry         - controlled multi-body arrangements
  algorithmic_invariants      - clean planted-offset scenes for unit recovery
  regression                  - pinned scenes reproducing fixed defects
  artifact_sweep              - planted-offset scenes under modest structured
                                telemetry loss (missing / truncated lines), in
                                uniform and adversarial placement variants; the
                                navigator must still recover the offset
  star_confounder             - one/two/three navigable stars in fields of
                                non-navigable confounders (the 1/2/3-star lock
                                regimes), plus a saturated-star and a double-star
                                scene; the navigator recovers the offset within a
                                tolerance that absorbs any centroid bias
  ring_system                 - optical-depth ring systems exercising the
                                perturbed-orbit and clutter machinery: a
                                satellite edge wave, an m = 2 forced ringlet,
                                seeded spokes, an embedded moonlet/propeller
                                with a non-navigable gap, and a planted
                                per-feature orbit error; each carries an
                                ``expected`` block pinned to its measured
                                outcome
  expected_fail               - scenes whose correct outcome is failure / low
                                confidence, never a confident wrong offset (every
                                star scattered off catalog, or a lone star drowned
                                in clutter); each carries an ``expected`` block
                                asserted by the sim expected-outcome machinery

Pixel convention
----------------

Every position that places something in a scene is a PIXEL CORNER: a star's
v / u, a body's center_v / center_u, and the ring system's geometry center_v /
center_u.  Integer N is the boundary between pixel N-1 and pixel N, so the
center of pixel N is N + 0.5 and the center of a size_v by size_u frame is
(size_v / 2, size_u / 2).  This is the oops uv convention, the one the FOV,
backplane, and star-record code underneath already use, so a position written
here is the same number those read.

Displacements carry no origin convention and need no such care: offset_v /
offset_u, move_v / move_u, catalog_error_v / catalog_error_u, and a
companion's sep_px are differences between two positions.

The planted offset_rotation_deg turns POSITIONS about ONE point in that same
convention: the frame's center (size_v / 2, size_u / 2).  Catalog star
positions, body centers, and the ring system's center all pivot there, so a
rolled scene plants a truth its three families of content agree on.  The two
orientations describing a whole projected pattern take the roll as well: a
body's rotation_z and the ring system's node_deg.

It does not turn the per-object quantities already stated in the detector
frame: move_v / move_u and a companion's angle_deg render at the angle the
scene wrote, whatever the roll, so stating them as they are to appear on the
detector is what a scene author does.  The sky_counts background field is not
rotated either -- its positions are drawn uniformly at random, so it carries no
orientation to turn.

The two centers inside the optics block -- distortion.center_v / center_u and
stray_light.center_v / center_u -- are pixel corners too.  What they name is
where a whole-frame field is centered rather than where an object sits, and the
same number written in either block names one point.  They do not turn with
offset_rotation_deg: the roll moves the scene across the detector, while these
name a place on the detector itself, so they render where the scene wrote them
whatever the roll.

Fields
------

  schema_version   (int, required)   must be 2
  scene_name       (str, required)   must equal the filename stem
  instrument       (str, required)   a sim instrument (coiss_nac, coiss_wac,
                                     coiss_calib_nac, coiss_calib_wac, gossi,
                                     nhlorri, vgiss) or 'generic'
  size_v, size_u   (int, required)   image height/width in pixels
  random_seed      (int, required)   scene seed (drives all sim randomness)
  exposure_sec     (float, opt)      exposure seconds (default 1.0)
  offset_v, offset_u (float, opt)    planted pointing offset (px) the navigator
                                     must recover (default 0.0)
  offset_rotation_deg (float, opt)   planted boresight roll (deg, default 0.0)
  midtime_utc      (str, optional)   ISO timestamp, informational
  closest_planet   (str, optional)   ring-model planet (default SATURN)
  time, ring_epoch (float, opt)      TDB seconds for ring calculations
  bodies           (list, optional)  per-body params (see below)
  ring_system      (mapping, opt)    the optical-depth ring system: a shared
                                     geometry block (center_v/u, opening_deg_obs,
                                     opening_deg_sun, node_deg), optional
                                     range_km / km_per_pixel (per-pixel depth
                                     ordering against bodies) and phase_deg,
                                     a features list, and truth-side azimuthal
                                     (modulation / spokes / shadow) and moonlets
                                     blocks.  Each feature carries kind (ringlet
                                     | gap | edge | ramp | wave), tau, the
                                     kind-specific shape keys (width_px, side,
                                     wavelength_px, damping_px), navigable (false
                                     by default: non-navigable features render as
                                     confounders the navigator never learns), a
                                     catalog orbit (a_px, ae_px, long_peri_deg,
                                     rate_peri_deg_per_day, m >= 2 modes,
                                     edge_wave), an idealized
                                     declared_orbit_sigma, and truth-side
                                     orbit_error / albedo / phase_g -- the
                                     planted ephemeris error displaces only the
                                     RENDERED feature; the navigator predicts
                                     from the catalog orbit.  See "Ring system
                                     keys" below for every key's unit
  stars            (list, optional)  explicit star dicts (name, v, u, vmag,
                                     navigable, catalog_error_v/u, companion,
                                     delta_mag, ...).  navigable=false renders the
                                     star as a confounder the navigator never
                                     learns (dropped from nav_params); the
                                     catalog_error, companion (sep_px/delta_mag/
                                     angle_deg), and delta_mag keys displace or
                                     rebrighten the RENDERED star off its catalog
                                     values -- image-side truth the navigator does
                                     not see
  star_catalog_scatter_px (float, opt) scene-level per-star position-scatter sigma
                                     (px): every rendered star is displaced by a
                                     seeded Gaussian draw of this sigma, added to
                                     any explicit per-star catalog_error
  sky_counts       (mapping, opt)    background-sky star field: a (intercept),
                                     b (slope) of log10 N(<m) = a + b*m per
                                     square degree, density_factor multiplier,
                                     diffuse_e_per_px flat floor
  expected         (mapping, opt)    the scene's expected navigation outcome
                                     (status success|failed|conflicted,
                                     confidence_tier or null, optional
                                     status_reason); read only by the sim
                                     expected-outcome test machinery -- fed to
                                     neither the renderer nor the navigator
  noise            (mapping, opt)    poisson, read_noise_dn, bias_dn,
                                     cosmic_ray_events_per_pixel_per_sec (expected
                                     events per pixel per second of exposure; CCD
                                     chain only, refused non-zero on the vidicon),
                                     missing_data_rate (raw-DN frames only, refused
                                     non-zero on a calibrated_if frame), bloom_length,
                                     signal_full_scale_frac, dark_current_e_per_sec,
                                     hot_pixel_fraction,
                                     hot_pixel_amplitude_e, hot_pixel_column_factor,
                                     banding_amplitude_e, banding_period_px,
                                     bias_pedestal_sigma_dn, bias_row_gradient_dn,
                                     bias_col_gradient_dn, vidicon {read_noise_line_dn,
                                     read_noise_pixel_dn, coherent_amplitude_dn,
                                     coherent_period_px}; unknown keys fail validation,
                                     and a key set to null counts as unset
  oversample       (int, opt)        radiance oversampling factor (default 4 when a PSF
                                     is active, else 1)
  optics           (mapping, opt)    whole-scene optical effects: psf (sigma_v, sigma_u,
                                     w, r0, n; or match_navigator, preserved as authored
                                     and resolved by the renderer), smear (list of
                                     dv_px/du_px/object_class), distortion (k1, k2,
                                     center_v, center_u, nonradial_rms_px), ghosts (list
                                     of dv_px/du_px/amplitude/defocus_sigma), stray_light
                                     (amplitude, direction_deg, center_v, center_u,
                                     model linear|radial)
  detector         (mapping, opt)    detector-chain override: gain_state (must be
                                     cataloged for the instrument), detector_model
                                     (ccd | vidicon), exposure_ref_sec, quantization
                                     (exact | 8bit | uneven_12bit | sqrt_lut); omitted
                                     or null keys track the instrument catalog
  artifacts        (mapping, opt)    instrument_defaults (bool): opt into the emulated
                                     camera's physical signal chain at catalog values
                                     (PSF, distortion residual, shot noise, dark / hot /
                                     bloom / banding / bias); artifact modes stay off.
                                     adversarial (bool): bias each mode's placement onto
                                     the navigation features.  Every other key is an
                                     artifact mode (missing_lines, hot_pixels, ...)
                                     whose map carries one activation key (see
                                     "Artifact activation keys" below) plus the mode's
                                     own shape parameters
  spk_error        (mapping, opt)    planted spacecraft-ephemeris parallax: dv_px, du_px,
                                     reference_range_km (needs range_km on each body
                                     and on the ring_system)
  instrument_config (mapping, opt)   per-instrument config overrides deep-merged over
                                     the named instrument's block (star_psf_sigma,
                                     data_units, noise.*, extfov_margin_vu, ...).  Omit
                                     to inherit everything; override individual keys to
                                     pin them to the scene; name 'generic' and override
                                     everything to fully self-specify so a later
                                     camera-config change cannot shift the scene.  (The
                                     top-level noise block still wins over
                                     instrument_config.noise for rendering.)
  fit_camera_rotation (bool, opt)    force whether navigation solves a camera roll

The instrument fixes the frame's units and its missing-data marker, in one of
three classes:

  coiss_nac, coiss_wac, gossi,      DN, marker 0; the bias pedestal keeps the dark
  nhlorri, generic                  sky above the marker
  coiss_calib_nac, coiss_calib_wac  I/F, marker NaN
  vgiss                             I/F, marker 0, as in the Voyager GEOMED products

An I/F frame spans roughly 0 to 1 (a vgiss frame tops out at 1.84, a
coiss_calib_nac frame near 2.0; under instrument_defaults read noise takes part
of the dark sky slightly below 0, about 5% of a vgiss disc scene's pixels and
more than a quarter of a coiss_calib_nac one's), so a display or threshold
scaled for DN shows it as black.  A calibrated frame's noise-free dark sky is
exactly 0 I/F.  On vgiss that equals the marker, so the navigator counts the sky
as missing data and refuses the frame (missing_data_dominant) when more than 30%
of its pixels are exactly 0.  Pin instrument_config.noise.marker_value to NaN to
navigate such a scene; it then no longer exercises the Voyager marker.  The scene
editor's status bar shows each render's units, range, and marker share, counted
on the frame the navigator reads (planted offset applied), and the navigator's
verdict when it is blank, overexposed, or mostly missing data.  Loading a vgiss
scene for navigation logs a warning on the image logger when the pixels the scene
renders at the marker put the frame over the limit.  A frame that the scene's
planted data loss alone puts over the limit doesn't get that warning.

Body params follow the renderer: shape_model (ellipsoid | polyhedral_mesh),
center_v, center_u, axis1, axis2, axis3, illumination_angle, phase_angle,
range_km, and -- for polyhedral_mesh -- mesh_lumpiness (the non-ellipsoidal
relief's RMS is mesh_lumpiness / 8 of the radius), mesh_seed (which shape),
mesh_n_lat / mesh_n_lon (tessellation, default 16 / 32; the catalog's mesh
scenes use 64 / 128 so no facets show), mesh_detail_octaves, and
pose_euler_deg.

A body may also carry an optional nav_override mapping.  The renderer ignores it
(it always draws the true geometry), but the navigator builds its predicted body
from the body params with nav_override overlaid -- the channel that makes the
navigation geometry diverge from the render geometry.  Use it to render an
irregular mesh yet predict its smooth (ellipsoidal) limit (mesh_lumpiness 0.0)
for a shape mismatch, or to predict the same body at a different pose_euler_deg
for a pose disagreement.  The override never changes the center, so the
predicted body stays at the unshifted position the planted offset is measured
from.

The planted offset_v/offset_u is applied as the rendered offset, so a navigator
predicting the unshifted geometry must recover it.

Artifact activation keys
------------------------

Each artifact mode in the artifacts block turns on through exactly one key, and
the key's name says what the number means:

  expected_events  (float >= 0)      count modes: the expected number of events
                                     per frame, drawn Poisson.  missing_lines,
                                     partial_lines, missing_blocks, line_garble,
                                     pixel_spikes, dead_pixels, dead_columns
                                     (each capped at the lines, blocks, columns,
                                     or pixels its events land on), and
                                     radiation_transients (uncapped; per frame
                                     at environment_factor 1 and
                                     readout_dwell_sec 0, and scaled by
                                     environment_factor * (exposure_sec +
                                     readout_dwell_sec) / exposure_sec)
  expected_events_per_sec (float >= 0)
                                     bright_dark_pairs only: the expected number
                                     of pairs per second of exposure, drawn
                                     Poisson and uncapped; the frame expects
                                     this times exposure_sec
  probability      (float in [0, 1]) every other mode except hot_pixels: the
                                     per-frame probability the mode fires at
                                     all (1.0 = always).  alternating_lines,
                                     edited_frame, truncated_frame, cutout_window,
                                     truth_window, embedded_header,
                                     compression_dct, reseau_scars,
                                     resample_texture, and the detector modes
                                     (banding_coherent, bias_structure, dark_ramp,
                                     bloom, frame_transfer_smear, serial_tail,
                                     beam_bend, residual_image, quantization_lut,
                                     quantization_ls8b, contouring_8bit,
                                     fixed_pattern)
  pixel_fraction   (float in [0, 1]) hot_pixels only: the fraction of detector
                                     pixels that are hot

dead_pixels and dead_columns also take count (int >= 0), a fixed number of dead
pixels or columns in place of the Poisson draw; a map may set count or
expected_events, never both (a key set to null counts as unset).  Every key
defaults to 0, and a mode whose key is 0 (or whose count is 0) renders as a
no-op.  For example:

    artifacts:
      missing_lines: {expected_events: 3.0}
      truncated_frame: {probability: 1.0, fraction: 0.25}
      dead_pixels: {count: 10}
      hot_pixels: {pixel_fraction: 0.002}

The key that used to cover all four meanings, incidence, is refused by the
validator, and again at render time, with a message naming the mode's
replacement and what that key holds on the mode.

Ring system keys
----------------

Every length and angle key in the ring_system block names its unit.  A length
ending in _px is in ring-plane pixels: a radial distance in the ring plane,
measured in detector pixels.  It projects to that many image pixels along the
line of nodes and is foreshortened by sin(B) across it, so on a face-on ring
(opening angle 90) it is a plain image distance.  An angle ends in _deg, except
the edge wave's damping_rad.  Every longitude is measured in the ring plane from
the ascending node.

  ring_system           phase_deg (the solar phase angle), range_km,
                        km_per_pixel
  geometry              center_v, center_u (pixel corners), opening_deg_obs,
                        opening_deg_sun, node_deg
  features[]            width_px, side, wavelength_px, damping_px (the shape
                        keys; see the table below), tau, albedo, phase_g
  features[].orbit      a_px (the semimajor axis, required), ae_px (the
                        eccentricity times a_px), long_peri_deg,
                        rate_peri_deg_per_day
  orbit.modes[]         m (an integer of at least 2: mode 1 is the base
                        ellipse), amp_px, long_peri_deg (the longitude of one
                        of the mode's m pericenters); the edge radius is
                        a_px - amp_px * cos(m * (lam - long_peri_deg))
  orbit.edge_wave       amp_px, wavelength_px (an arc length along the orbit),
                        damping_rad (at most 2), lam0_deg (the launch
                        longitude)
  orbit_error           delta_a_px, delta_ae_px, delta_long_peri_deg
  declared_orbit_sigma  sigma_a_px, sigma_ae_px, sigma_long_peri_deg
  azimuthal.modulation  amplitude, m (an integer of at least 1, default 1),
                        lam_peak_deg (the longitude of a brightness peak)
  azimuthal.shadow      start_deg, extent_deg, darkness
  azimuthal.spokes      count, r_inner_px, r_outer_px, contrast, width_deg
  moonlets[]            a_px, lam_deg, radius_px, amplitude, and a propeller
                        {length_deg, width_px, contrast}

A wave feature's damping_px and an edge wave's damping_rad are both the e-folding
length of a damped wave's envelope.  The wave feature's runs radially, in
ring-plane pixels outward of its launch radius.  The edge wave's runs in
longitude, in radians downstream of its launch longitude.

Each kind of feature takes its own radial shape keys.  Every kind requires kind,
tau, and an orbit with orbit.a_px.  The validator rejects a shape key on a kind
that does not take it, so a stray key cannot silently author a different
feature.

  key            ringlet   gap       edge      ramp      wave
  width_px       required  required  rejected  required  rejected
  side           rejected  rejected  optional  optional  rejected
  wavelength_px  rejected  rejected  rejected  rejected  required
  damping_px     rejected  rejected  rejected  rejected  required
  default side   n/a       n/a       in        out       n/a

An edge is a half-plane, not a line.  The features compose into one shared
optical-depth map, and an edge fills every radius on one side of its orbit:
side in (the default) carries its tau everywhere inward of a_px, and side out
everywhere outward.  Two edges therefore cannot bound an annulus.  An edge with
side out at 60 px and one with side in at 95 px overlap, and the would-be
annulus is only the band where tau doubles.  Author a band as a ringlet (a_px 60,
width_px 35), or carve it into a sheet with a gap.

The validator refuses each retired ring key with a message naming its
replacement: a, ae, long_peri, and rate_peri in an orbit; width, wavelength, and
damping on a feature; amp and peri in an m-mode; amp, wavelength, damp, and lam0
in an edge wave; phase_deg in the brightness modulation; r_inner and r_outer in
the spokes; and a moonlet's a.

Adding a scene
--------------

Drop a new <scene_name>.yaml under the appropriate class directory.  Run
``pytest tests/integration/test_sim_scenes.py`` to confirm it validates and
renders.
