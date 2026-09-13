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
centre of pixel N is N + 0.5 and the centre of a size_v by size_u frame is
(size_v / 2, size_u / 2).  This is the oops uv convention, the one the FOV,
backplane, and star-record code underneath already use, so a position written
here is the same number those read.

Displacements carry no datum and need no such care: offset_v / offset_u,
move_v / move_u, catalog_error_v / catalog_error_u, and a companion's sep_px
are differences between two positions.

The two centres inside the optics block -- distortion.center_v / center_u and
stray_light.center_v / center_u -- are pixel indices instead.  They name where
a whole-frame field is centred rather than where an object sits.

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
                                     kind-specific shape keys (width, side,
                                     wavelength, damping), navigable (false by
                                     default: non-navigable features render as
                                     confounders the navigator never learns), a
                                     catalog orbit (a, ae, long_peri, rate_peri,
                                     m >= 2 modes, edge_wave), an idealized
                                     declared_orbit_sigma, and truth-side
                                     orbit_error / albedo / phase_g -- the
                                     planted ephemeris error displaces only the
                                     RENDERED feature; the navigator predicts
                                     from the catalog orbit
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
                                     cosmic_ray_rate_per_sec, missing_data_rate,
                                     bloom_length, signal_full_scale_frac, pixel_area_cm2,
                                     dark_current_e_per_sec, hot_pixel_fraction,
                                     hot_pixel_amplitude_e, hot_pixel_column_factor,
                                     banding_amplitude_e, banding_period_px,
                                     bias_pedestal_sigma_dn, bias_row_gradient_dn,
                                     bias_col_gradient_dn, vidicon {read_noise_line_dn,
                                     read_noise_pixel_dn, coherent_amplitude_dn,
                                     coherent_period_px}; unknown keys fail validation
  oversample       (int, opt)        radiance oversampling factor (default 4 when a PSF
                                     is active, else 1)
  optics           (mapping, opt)    whole-scene optical effects: psf (sigma_v, sigma_u,
                                     w, r0, n; or match_navigator, preserved as authored
                                     and resolved by the renderer), smear (list of
                                     dv_px/du_px/object_class), distortion (k1, k2,
                                     center_v, center_u, nonradial_rms_px), ghosts (list
                                     of dv_px/du_px/amplitude/defocus_sigma), stray_light
                                     (amplitude, direction_deg, model linear|radial)
  detector         (mapping, opt)    detector-chain override: gain_state (must be
                                     catalogued for the instrument), detector_model
                                     (ccd | vidicon), exposure_ref_sec, quantization
                                     (exact | 8bit | uneven_12bit | sqrt_lut); omitted
                                     keys track the instrument catalog
  artifacts        (mapping, opt)    instrument_defaults (bool): opt into the emulated
                                     camera's physical signal chain at catalog values
                                     (PSF, distortion residual, shot noise, dark / hot /
                                     bloom / banding / bias); loss modes stay 0
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

Body params follow the renderer: shape_model (ellipsoid | polyhedral_mesh),
center_v, center_u, axis1, axis2, axis3, illumination_angle, phase_angle,
range_km, and -- for polyhedral_mesh -- mesh_lumpiness, mesh_seed,
pose_euler_deg.

A body may also carry an optional nav_override mapping.  The renderer ignores it
(it always draws the true geometry), but the navigator builds its predicted body
from the body params with nav_override overlaid -- the channel that makes the
navigation geometry diverge from the render geometry.  Use it to render an
irregular mesh yet predict its smooth (ellipsoidal) limit (mesh_lumpiness 0.0)
for a shape mismatch, or to predict the same body at a different pose_euler_deg
for a pose disagreement.  The override never changes the centre, so the
predicted body stays at the unshifted position the planted offset is measured
from.

The planted offset_v/offset_u is applied as the rendered offset, so a navigator
predicting the unshifted geometry must recover it.

Adding a scene
--------------

Drop a new <scene_name>.yaml under the appropriate class directory.  Run
``pytest tests/integration/test_sim_scenes.py`` to confirm it validates and
renders.
