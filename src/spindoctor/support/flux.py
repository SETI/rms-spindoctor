# import logging


# import scipy.constants as const
# import scipy.integrate as integrate

# import oops
# from psfmodel.gaussian import GaussianPSF


# import spindoctor.config
# from spindoctor.misc import simple_filter_name_f1f2
# import spindoctor.plot3d

# _LOGGING_NAME = 'cb.' + __name__


# ===============================================================================
#
# FILTER CONVOLUTIONS
#
# ===============================================================================

# def _interpolate_and_convolve_2(x1, y1, x2, y2):
#     """Convolve two tabulations and return the intersected interval."""
#     min_x = max(np.min(x1), np.min(x2))
#     max_x = min(np.max(x1), np.max(x2))
#     new_x = np.arange(min_x, max_x+0.1)

#     new_y1 = interp.interp1d(x1, y1)(new_x)
#     new_y2 = interp.interp1d(x2, y2)(new_x)

#     return new_x, new_y1*new_y2

# def _interpolate_and_convolve_3(x1, y1, x2, y2, x3, y3):
#     """Convolve three tabulations and return the intersected interval."""
#     min_x = max(np.ceil(np.min(x1)), np.ceil(np.min(x2)), np.ceil(np.min(x3)))
#     max_x = min(np.floor(np.max(x1)), np.floor(np.max(x2)), np.floor(np.max(x3)))
#     new_x = np.arange(min_x, max_x+0.1)

#     new_y1 = interp.interp1d(x1, y1)(new_x)
#     new_y2 = interp.interp1d(x2, y2)(new_x)
#     new_y3 = interp.interp1d(x3, y3)(new_x)

#     return new_x, new_y1*new_y2*new_y3


# ===============================================================================
#
# CISSCAL-Related Functions
#
# ===============================================================================

# The steps taken by CISSCAL for radiometric calibration
# (cassimg__radiomcalib.pro) are:
#
# Step 1: Conversion from 8 to 12 bits if 12-to-8 table was used
# Step 2: Correction for uneven bit weighting
# Step 3: Bias subtraction
# Step 4: Subtraction of 2-hz noise
# Step 5: Subtraction of appropriate dark frame
# Step 6: Correct for bright/dark differences in antiblooming mode
# Step 7: Correct for nonlinearity
# Step 8: Flat field correction
# Step 9: Convert to Flux
#   DNtoElectrons      ; multiply by gain factor
#   DivideByExpoT      ; divide by exp time, correcting for shutter offset
#   DivideByAreaPixel  ; divide by optics area and solid angle
#   DivideByEfficiency ; divide by T0*T1*T2*QE summed over passband
# Step 10: Absolute Correction Factors and Sensitivity vs. Time
# Step 11: Apply geometric correction if required
#
# From the CISSCAL User Guide March 20, 2009 section 5.9
# Converting from DN to Flux:
#   1) cassimg__dntoelectrons.pro
#          ELECTRONS = DN * GAIN / GAIN_RATIO[GAIN_MODE_ID]
#   2) cassimg__dividebyexpot.pro
#          We ignore shutter offset timing for our purposes
#          DATA = ELECTRONS / (ExpT/1000) [ExpT in ms]
#   3) cassimg__dividebyareapixel.pro
#          SUM_FACTOR = (SAMPLES/1024) * (LINES/1024)
#          DATA = DATA * SUM_FACTOR / (SOLID_ANGLE * OPTICS_AREA)
#   4) cassimg__dividebyefficiency.pro
#      This is Quantum Efficiency * Optics Transmission *
#             Filter 1 * Filter 2
#          DATA = DATA / INTEG(QE * TRANS)
#      In I/F mode:
#          FLUX = SOLAR_FLUX / (PI * DIST^2)
#          DATA = DATA / INTEG(QE * TRANS * FLUX)
# This yields a result in phot/cm^2/s/nm/ster
#
# Our job here is to take the output of this calibration pipeline and undo
# Step 9 to get back to raw Data Numbers (DN).
#
# calibrate_iof_image_as_dn does all of this work.


# _CISSCAL_DETECTOR_GAIN = {'NAC': 30.27, 'WAC': 27.68}
# _CISSCAL_DETECTOR_GAIN_RATIO = {'NAC': [0.135386, 0.309569, 1.0, 2.357285],
#                                 'WAC': [0.125446, 0.290637, 1.0, 2.360374]}
# _CISSCAL_DETECTOR_SOLID_ANGLE = {'NAC': 3.58885e-11, 'WAC': 3.56994e-9} # Steradians
# _CISSCAL_DETECTOR_OPTICS_AREA = {'NAC': 284.86, 'WAC': 29.43} # Aperture cm^2

# _IOF_DN_CONVERSION_FACTOR_CACHE = {}

# def calibrate_iof_image_as_dn(obs, data=None):
#     """Convert an image currently in I/F to post-LUT raw DN.

#     The input observation data is in I/F.
#     """
#     logger = logging.getLogger(_LOGGING_NAME+'.calibrate_iof_image_as_dn')

#     if data is None:
#         # Can be overridden if we want to calibrate some other data block
#         data = obs.data

#     key = (obs.clean_detector, obs.filter1, obs.filter2, obs.texp)
#     if key in _IOF_DN_CONVERSION_FACTOR_CACHE:
#         factor = _IOF_DN_CONVERSION_FACTOR_CACHE[key]
#         logger.debug('Calibration for %s %s %s %.2f cached; factor = %f',
#                      obs.clean_detector, obs.filter1, obs.filter2, obs.texp,
#                     factor)
#         return data * factor

#     logger.debug('Calibrating %s %s %s',
#                  obs.clean_detector, obs.filter1, obs.filter2)

#     # Initial image data is in I / F

#     # 4) cassimg__dividebyefficiency.pro
#     #        FLUX = SOLAR_FLUX / (PI * DIST^2)
#     #        DATA = DATA / INTEG(QE * TRANS * FLUX)
#     factor = _compute_cisscal_solar_flux_efficiency(obs)
#     # Image data now in photons / cm^2 / s / nm assuming no filters or QE
#     # correction

#     # 3) cassimg__dividebyareapixel.pro
#     #        SUM_FACTOR = (SAMPLES/1024) * (LINES/1024)
#     #        DATA = DATA * SUM_FACTOR / (SOLID_ANGLE * OPTICS_AREA)
#     # Use obs.data not data here because we want the real size of the original
#     # image.
#     sum_factor = obs.data_shape_xy[1] / 1024. * obs.data_shape_xy[0] / 1024.

#     area_factor = (sum_factor /
#                    (_CISSCAL_DETECTOR_SOLID_ANGLE[obs.clean_detector] *
#                     _CISSCAL_DETECTOR_OPTICS_AREA[obs.clean_detector]))
#     factor /= area_factor
#     # photons / s

#     # 2) cassimg__dividebyexpot.pro
#     #        We ignore shutter offset timing for our purposes
#     #        DATA = ELECTRONS / (ExpT/1000) [ExpT in ms]
#     factor *= obs.texp # texp is already in sec
#     # photons

#     # 1) cassimg__dntoelectrons.pro
#     #        ELECTRONS = DN * GAIN / GAIN_RATIO[GAIN_MODE_ID]
#     # IF self.Instrument EQ 'ISSNA' THEN BEGIN
#     #     Gain2 = 30.27
#     #     GainRatios = [0.135386, 0.309569, 1.0, 2.357285]
#     # TrueGain = Gain2/GainRatios
#     # *self.ImageP = *self.ImageP * TrueGain[GainState]

#     factor /= (_CISSCAL_DETECTOR_GAIN[obs.clean_detector] /
#                _CISSCAL_DETECTOR_GAIN_RATIO[obs.clean_detector][obs.gain_mode])
#     # photons

#     _IOF_DN_CONVERSION_FACTOR_CACHE[key] = factor

#     logger.debug('Final adjustment factor = %e', factor)

#     return data * factor

# def _read_cisscal_calib_file(filename):
#     """Read a CISSCAL calibration table."""
#     logger = logging.getLogger(_LOGGING_NAME+'.read_cisscal_calib_file')
#     logger.debug('Reading "%s"', filename)

#     with open(filename, 'r') as fp:
#         for line in fp:
#             if line.startswith('\\begindata'):
#                 break
#         else:
#             assert False
#         ret_list = []
#         for line in fp:
#             if line.startswith('\\enddata'):
#                 break
#             fields = line.strip('\r\n').split()
#             field_list = [float(x) for x in fields]
#             ret_list.append(field_list)

#     return ret_list


# _CISSCAL_FILTER_TRANSMISSION_CACHE = {}

# def _cisscal_filter_transmission(obs):
#     """Return the (wavelengths, transmission) for the joint filters."""
#     key = (obs.clean_detector, obs.filter1, obs.filter2)
#     if key not in _CISSCAL_FILTER_TRANSMISSION_CACHE:
#         filter_filename = ('iss' + obs.clean_detector.lower()[:2] +
#                            obs.filter1.lower())
#         filter_filename += obs.filter2.lower() + '_systrans.tab'
#         systrans_filename = os.path.join(spindoctor.config.CISSCAL_CALIB_ROOT,
#                                          'efficiency',
#                                          'systrans', filter_filename)
#         systrans_list = _read_cisscal_calib_file(systrans_filename)
#         systrans_wl = [x[0] for x in systrans_list] # Wavelength in nm
#         systrans_xmit = [x[1] for x in systrans_list]

#         _CISSCAL_FILTER_TRANSMISSION_CACHE[key] = (systrans_wl, systrans_xmit)

#     return _CISSCAL_FILTER_TRANSMISSION_CACHE[key]


# _CISSCAL_QE_CORRECTION_CACHE = {}

# def _cisscal_qe_correction(obs):
#     """Return the (wavelengths, vals) for QE correction."""
#     key = obs.clean_detector
#     if key not in _CISSCAL_QE_CORRECTION_CACHE:
#         qecorr_filename = os.path.join(
#                                spindoctor.config.CISSCAL_CALIB_ROOT, 'correction',
#                                obs.clean_detector.lower()+'_qe_correction.tab')
#         qecorr_list = _read_cisscal_calib_file(qecorr_filename)
#         qecorr_wl = [x[0] for x in qecorr_list] # Wavelength in nm
#         qecorr_val = [x[1] for x in qecorr_list]

#         _CISSCAL_QE_CORRECTION_CACHE[key] = (qecorr_wl, qecorr_val)

#     return _CISSCAL_QE_CORRECTION_CACHE[key]


# _CISSCAL_SOLAR_FLUX_CACHE = None

# def _cisscal_solar_flux():
#     """Return the (wavelengths, flux) for the solar flux in phot/cm^2/s/nm at
#     1 AU."""
#     global _CISSCAL_SOLAR_FLUX_CACHE

#     if _CISSCAL_SOLAR_FLUX_CACHE is None:
#         # Flux is in photons / cm^2 / s / angstrom at 1 AU
#         solarflux_filename = os.path.join(spindoctor.config.CISSCAL_CALIB_ROOT,
#                                           'efficiency',
#                                           'solarflux.tab')
#         solarflux_list = _read_cisscal_calib_file(solarflux_filename)
#         # lambda_f = temporary(lambda_f/ang_to_nm)
#         # flux = temporary(flux*ang_to_nm)
#         solarflux_wl = [x[0]/10. for x in solarflux_list] # Wavelength in nm
#         solarflux_flux = [x[1]*10. for x in solarflux_list]
#         # Flux is now in photons / cm^2 / s / nm at 1 AU

#         _CISSCAL_SOLAR_FLUX_CACHE = (solarflux_wl, solarflux_flux)

#     return _CISSCAL_SOLAR_FLUX_CACHE

# #===============================================================================

# def _compute_cisscal_efficiency(obs):
#     """Compute the integrated efficiency factor without solar flux."""
#     # From cassimg__dividebyefficiency.pro

#     logger = logging.getLogger(_LOGGING_NAME+'._compute_cisscal_efficiency')

#     # Read in filter transmission
#     systrans_wl, systrans_xmit = _cisscal_filter_transmission(obs)

#     # Read in QE correction
#     qecorr_wl, qecorr_val = _cisscal_qe_correction(obs)

#     min_wl = np.ceil(np.max([np.min(systrans_wl),
#                              np.min(qecorr_wl)]))
#     max_wl = np.floor(np.min([np.max(systrans_wl),
#                               np.max(qecorr_wl)]))

#     all_wl = systrans_wl + qecorr_wl
#     all_wl = list(set(all_wl)) # uniq
#     all_wl.sort()
#     all_wl = np.array(all_wl)
#     all_wl = all_wl[all_wl >= min_wl]
#     all_wl = all_wl[all_wl <= max_wl]

#     new_trans = interp.interp1d(systrans_wl, systrans_xmit)(all_wl)
#     new_qe = interp.interp1d(qecorr_wl, qecorr_val)(all_wl)

#     # Note the original IDL code uses 5-point Newton-Coates while
#     # we only use 3-point. This really shouldn't make any difference
#     # for our purposes.
#     eff_fact = integrate.simps(new_trans*new_qe, all_wl)

#     logger.debug('w/o solar flux wavelength range %f to %f, eff factor %f',
#                  min_wl, max_wl, eff_fact)

#     return eff_fact

# def _compute_cisscal_solar_flux_efficiency(obs):
#     """Compute the efficiency factor including solar flux."""
#     # From cassimg__dividebyefficiency.pro

#     logger = logging.getLogger(_LOGGING_NAME+
#                                '._compute_cisscal_solar_flux_efficiency')

#     # Read in filter transmission
#     systrans_wl, systrans_xmit = _cisscal_filter_transmission(obs)

#     # Read in QE correction
#     qecorr_wl, qecorr_val = _cisscal_qe_correction(obs)

#     # Read in solar flux
#     solarflux_wl, solarflux_flux = _cisscal_solar_flux()

#     # Compute distance Sun-Saturn in AU
#     solar_range = obs.sun_distance("SATURN")

#     logger.debug('Solar range = %f AU', solar_range)

#     # We do the convolutions in this particular manner because it's the
#     # way CISSCAL does it and we are trying to get as precise a result
#     # as we can while undoing CISSCAL's computations.

#     # minlam=ceil(max([min(lambda_t),min(lambda_f),min(lambda_q)]))
#     # maxlam=floor(min([max(lambda_t),max(lambda_f),max(lambda_q)]))
#     #
#     # lambda = [lambda_f,lambda_t,lambda_q]
#     # lambda = lambda[where((lambda ge minlam) and (lambda le maxlam))]
#     # lambda = lambda[uniq(lambda,sort(lambda))]

#     min_wl = np.ceil(max(np.min(systrans_wl),
#                          np.min(qecorr_wl),
#                          np.min(solarflux_wl)))
#     max_wl = np.floor(min(np.max(systrans_wl),
#                           np.max(qecorr_wl),
#                           np.max(solarflux_wl)))

#     all_wl = systrans_wl + qecorr_wl + solarflux_wl
#     all_wl = list(set(all_wl)) # uniq
#     all_wl.sort()
#     all_wl = np.array(all_wl)
#     all_wl = all_wl[all_wl >= min_wl]
#     all_wl = all_wl[all_wl <= max_wl]

#     # newtrans = interpol(trans,lambda_t,lambda)
#     # newqecorr = interpol(qecorr,lambda_q,lambda)
#     # newflux = interpol(flux,lambda_f,lambda)/(pifact * dfs^2)

#     new_trans = interp.interp1d(systrans_wl, systrans_xmit)(all_wl)
#     new_qe = interp.interp1d(qecorr_wl, qecorr_val)(all_wl)
#     new_flux = interp.interp1d(solarflux_wl, solarflux_flux)(all_wl)
#     new_flux /= (oops.PI * solar_range**2)
#     # Flux is now in photons / cm^2 / s / nm at Saturn's distance
#     # Dividing by pi is necessary because Solar Flux = pi F in I/F
#     # thus I/F = I/ [Solar flux / pi]

#     # Note the original IDL code uses 5-point Newton-Coates while
#     # we only use 3-point. This really shouldn't make any difference
#     # for our purposes.
#     eff_fact = integrate.simps(new_trans*new_qe*new_flux, all_wl)

#     logger.debug('w/solar flux wavelength range %f to %f, eff factor %f',
#                  min_wl, max_wl, eff_fact)

#     return eff_fact

# ===============================================================================

# _IOF_FLUX_CONVERSION_FACTOR_CACHE = {}

# def calibrate_iof_image_as_flux(obs):
#     """Convert an image currently in I/F to flux.

#     The input observation data is in I/F.
#     The output data is in phot/cm^2/s/nm/ster.
#     """
#     # We undo step 4 and then redo it with no stellar flux

#     logger = logging.getLogger(_LOGGING_NAME+'.calibrate_iof_image_as_flux')

#     key = (obs.clean_detector, obs.filter1, obs.filter2)
#     if key in _IOF_FLUX_CONVERSION_FACTOR_CACHE:
#         factor = _IOF_FLUX_CONVERSION_FACTOR_CACHE[key]
#         logger.debug('Calibration for %s %s %s cached; factor = %f',
#                      obs.clean_detector, obs.filter1, obs.filter2, factor)
#         return obs.data * factor

#     logger.debug('Calibrating %s %s %s',
#                  obs.clean_detector, obs.filter1, obs.filter2)

#     # Undo Step 4 by multiplying by system transmission
#     # efficiency including solar flux
#     factor = _compute_cisscal_solar_flux_efficiency(obs)

#     # Redo Step 4 by dividing by system transmission efficiency
#     # excluding solar flux
#     factor /= _compute_cisscal_efficiency(obs)

#     _IOF_FLUX_CONVERSION_FACTOR_CACHE[key] = factor

#     logger.debug('Final adjustment factor = %e', factor)

#     return obs.data * factor


# ===============================================================================
#
# CASSINI FILTER TRANSMISSION FUNCTIONS
#
# ===============================================================================

# _CASSINI_FILTER_TRANSMISSION = {}

# def _cassini_filter_transmission(detector, filter):
#     """Return the (wavelengths, transmission) for the given Cassini filter."""

#     logger = logging.getLogger(_LOGGING_NAME+
#                                '.cassini_filter_transmission')

#     if len(_CASSINI_FILTER_TRANSMISSION) == 0:
#         for iss_det in ('NAC', 'WAC'):
#             base_dirname = iss_det[0].lower() + '_c_trans_sum'
#             filename = os.path.join(spindoctor.config.CASSINI_CALIB_ROOT, base_dirname,
#                                     'all_filters.tab')
#             logger.debug('Reading "%s"', filename)
#             with open(filename, 'r') as filter_fp:
#                 header = filter_fp.readline().strip('\r\n')
#                 header_fields = header.split('\t')
#                 assert header_fields[0] == 'WAVELENGTH (nm)'
#                 filter_name_list = []
#                 for i in range(1, len(header_fields)):
#                     filter_name = header_fields[i]
#                     # For unknown reasons the headers of the NAC and WAC files are
#                     # formatted differently. They also have weird names for the
#                     # polarized filters.
#                     if filter_name[:7] == 'VIS POL': # NAC
#                         filter_name = 'P0'
#                     elif filter_name[:6] == 'IR POL': # NAC
#                         filter_name = 'IRP0'
#                     elif filter_name[:6] == 'IR_POL': # WAC
#                         filter_name = 'IRP0'
#                     elif filter_name[0].isdigit():
#                         filter_name = filter_name[-3:]
#                     else:
#                         filter_name = filter_name[:3]

#                     filter_name_list.append(filter_name)

#                 for i in range(len(filter_name_list)):
#                     key = (iss_det, filter_name_list[i])
#                     _CASSINI_FILTER_TRANSMISSION[key] = ([], []) # wl, xmission
#                 for filter_line in filter_fp:
#                     filter_line = filter_line.strip('\r\n')
#                     filter_fields = filter_line.split('\t')
#                     assert len(filter_fields) == len(filter_name_list)+1
#                     if len(filter_fields[0]) == 0:
#                         continue
#                     wl = float(filter_fields[0])
#                     for i in range(len(filter_name_list)):
#                         filter_field = filter_fields[i+1].strip('\r\n')
#                         if len(filter_field) > 0:
#                             key = (iss_det, filter_name_list[i])
#                             xmission = float(filter_field)
#                             _CASSINI_FILTER_TRANSMISSION[key][0].append(wl)
#                             _CASSINI_FILTER_TRANSMISSION[key][1].append(xmission)

#         pol = _CASSINI_FILTER_TRANSMISSION[('NAC', 'P0')]
#         _CASSINI_FILTER_TRANSMISSION[('NAC', 'P60')] = pol
#         _CASSINI_FILTER_TRANSMISSION[('NAC', 'P120')] = pol
#         pol = _CASSINI_FILTER_TRANSMISSION[('WAC', 'IRP0')]
#         _CASSINI_FILTER_TRANSMISSION[('NAC', 'IRP90')] = pol

#     return _CASSINI_FILTER_TRANSMISSION[(detector, filter)]


# def plot_cassini_filter_transmission():
#     """Plot the Cassini filter transmission functions."""

#     color_info = {
#         'CL1': ('#000000', '-'),
#         'CL2': ('#808080', '-'),
#         'BL1': ('#4040a0', '-'),
#         'BL2': ('#0000ff', '-'),
#         'UV1': ('#800080', '-'),
#         'UV2': ('#c000c0', '-'),
#         'UV3': ('#ff00ff', '-'),
#         'GRN': ('#00ff00', '-'),
#         'RED': ('#ff0000', '-'),
#         'VIO': ('#000040', '-'),
#         'IR1': ('#ff0000', '--'),
#         'IR2': ('#ff4040', '--'),
#         'IR3': ('#ff8080', '--'),
#         'IR4': ('#ffa0a0', '--'),
#         'IR5': ('#ffc0c0', '--'),

#         'MT1': ('#008080', ':'),
#         'MT2': ('#00c0c0', ':'),
#         'MT3': ('#00ffff', ':'),
#         'CB1': ('#400040', ':'),
#         'CB2': ('#408080', ':'),
#         'CB3': ('#4080ff', ':'),

#         'HAL': ('#ff8080', ':'),

#         'P0':  ('#404040', '--'),
#         'P60': ('#808080', '--'),
#         'P120':('#c0c0c0', '--'),
#         'IRP0':('#404080', '--'),
#         'IRP90':('#4040c0', '--')
#     }

#     _cassini_filter_transmission('NAC', 'CL1') # This reads all filters
#     for detector in ('NAC', 'WAC'):
#         fig = plt.figure()
#         plt.title(detector)
#         for key in sorted(_CASSINI_FILTER_TRANSMISSION.keys()):
#             filter_det, filter_name = key
#             if filter_det != detector:
#                 continue
#             if len(filter_name) == 1: # Bogus single-letter filters
#                 continue
#             wl_list, xmission_list = _CASSINI_FILTER_TRANSMISSION[key]
#             plt.plot(wl_list, xmission_list, color_info[filter_name][1],
#                      color=color_info[filter_name][0], label=filter_name)
#         plt.legend()
#     plt.show()


# ===============================================================================
#
# STANDARD PHOTOMETRIC FILTER TABLES
#
# ===============================================================================

# From Bessel 1990
# _JOHNSON_B_WL = np.arange(360.,561.,10)
# _JOHNSON_B = np.array([
#     0.000, 0.030, 0.134, 0.567, 0.920, 0.978, 1.000, 0.978, 0.935, 0.853, 0.740,
#     0.640, 0.536, 0.424, 0.325, 0.235, 0.150, 0.095, 0.043, 0.009, 0.000])

# _JOHNSON_V_WL = np.arange(470.,701.,10)
# _JOHNSON_V = np.array([
#     0.000, 0.030, 0.163, 0.458, 0.780, 0.967, 1.000, 0.973, 0.898, 0.792, 0.684,
#     0.574, 0.461, 0.359, 0.270, 0.197, 0.135, 0.081, 0.045, 0.025, 0.017, 0.013,
#     0.009, 0.000])

# def plot_johnson_filter_transmission():
#     """Plot the Johnson B and V filter transmission functions"""
#     fig = plt.figure()
#     plt.plot(_JOHNSON_B_WL, _JOHNSON_B, '-', color='blue', label='B')
#     plt.plot(_JOHNSON_V_WL, _JOHNSON_V, '-', color='green', label='V')
#     plt.legend()
#     plt.show()


# ===============================================================================
#
# OPERATIONS ON STELLAR SPECTRA
#
# ===============================================================================


def clean_sclass(sclass: str | None) -> str:
    """Return a clean stellar classification such as A0 or M8."""
    if sclass is None:
        sclass = 'XX'
    elif sclass[0] == 'g':
        sclass = sclass[1:]
    sclass = sclass[:2]
    return sclass


# _STELLAR_SPECTRUM_FILES = {
#     'O0': None,
#     'O1': None,
#     'O2': None,
#     'O3': None,
#     'O4': 'uko5v.dat',
#     'O5': 'uko5v.dat',
#     'O6': 'uko5v.dat',
#     'O7': None,
#     'O8': 'uko9v.dat',
#     'O9': 'uko9v.dat',
#     'B0': 'ukb0v.dat',
#     'B1': 'ukb1v.dat',
#     'B2': 'ukb1v.dat',
#     'B3': 'ukb3v.dat',
#     'B4': 'ukb3v.dat',
#     'B5': None,
#     'B6': None,
#     'B7': 'ukb8v.dat',
#     'B8': 'ukb8v.dat',
#     'B9': 'ukb9v.dat',
#     'A0': 'uka0v.dat',
#     'A1': 'uka0v.dat',
#     'A2': 'uka2v.dat',
#     'A3': 'uka3v.dat',
#     'A4': 'uka5v.dat',
#     'A5': 'uka5v.dat',
#     'A6': 'uka5v.dat',
#     'A7': 'uka7v.dat',
#     'A8': 'uka7v.dat',
#     'A9': None,
#     'F0': 'ukf0v.dat',
#     'F1': 'ukf0v.dat',
#     'F2': 'ukf2v.dat',
#     'F3': 'ukf2v.dat',
#     'F4': 'ukf5v.dat',
#     'F5': 'ukf5v.dat',
#     'F6': 'ukf6v.dat',
#     'F7': 'ukf6v.dat',
#     'F8': 'ukf8v.dat',
#     'F9': 'ukf8v.dat',
#     'G0': 'ukg0v.dat',
#     'G1': 'ukg0v.dat',
#     'G2': 'ukg2v.dat',
#     'G3': 'ukg2v.dat',
#     'G4': 'ukg5v.dat',
#     'G5': 'ukg5v.dat',
#     'G6': 'ukg5v.dat',
#     'G7': 'ukg8v.dat',
#     'G8': 'ukg8v.dat',
#     'G9': 'ukg8v.dat',
#     'K0': 'ukk0v.dat',
#     'K1': 'ukk0v.dat',
#     'K2': 'ukk2v.dat',
#     'K3': 'ukk3v.dat',
#     'K4': 'ukk4v_new.dat',
#     'K5': 'ukk5v.dat',
#     'K6': 'ukk5v.dat',
#     'K7': 'ukk7v.dat',
#     'K8': 'ukk7v.dat',
#     'K9': None,
#     'M0': 'ukm0v.dat',
#     'M1': 'ukm1v.dat',
#     'M2': 'ukm2v.dat',
#     'M3': 'ukm3v.dat',
#     'M4': 'ukm4v_new.dat',
#     'M5': 'ukm5v.dat',
#     'M6': 'ukm6v.dat',
#     'M7': 'ukm6v.dat',
#     'M8': None,
#     'M9': None
# }

# _STELLAR_SPECTRUM_CACHE = {}

# def _read_stellar_spectrum(wavelength, spectral_class):
#     """Read the stellar spectrum for a particular spectral class.

#     Wavelength is in nm.
#     Result is in photons / cm^2 / s / nm / steradian.
#     """
#     logger = logging.getLogger(_LOGGING_NAME+'._read_stellar_spectrum')

#     # Simulated spectra are from
#     # http://www.eso.org/sci/facilities/paranal/decommissioned/isaac/tools/lib.html
#     # (A.J. Pickles, PASP 110,863, 1998)
#     # http://www.eso.org/sci/facilities/paranal/decommissioned/isaac/tools/lib/hilib.pdf
#     # _new files are from  Ivanov et al. (2004, ApJS, 151, 387)
#     # http://www.eso.org/sci/facilities/paranal/decommissioned/isaac/tools/lib/Ivanov_etal_2004.pdf

#     spectral_class = clean_sclass(spectral_class)

#     if (spectral_class not in _STELLAR_SPECTRUM_FILES or
#         _STELLAR_SPECTRUM_FILES[spectral_class] is None):
#         return None

#     if spectral_class in _STELLAR_SPECTRUM_CACHE:
#         wl_arr, val_arr = _STELLAR_SPECTRUM_CACHE[spectral_class]
#     else:
#         filename = os.path.join(spindoctor.config.CB_SUPPORT_FILES_ROOT,
#                                 'stellar_spectra',
#                                 _STELLAR_SPECTRUM_FILES[spectral_class])

#         logger.debug('Reading stellar spectrum %s: "%s"', spectral_class,
#                      filename)

#         with open(filename, 'r') as fp:
#             for line in fp:
#                 if line.startswith('#'):
#                     continue
#                 wl_list = []
#                 val_list = []
#                 for line in fp:
#                     fields = line.strip('\r\n').split()
#                     field_list = [float(x) for x in fields]
#                     wl_list.append(field_list[0] / 10) # In nm
#                     val_list.append(field_list[1])
#                 if wl_list[0] > 100:
#                     wl_list.insert(0, 100.)
#                     val_list.insert(0, 0.)
#         wl_arr = np.array(wl_list)
#         val_arr = np.array(val_list)
#         _STELLAR_SPECTRUM_CACHE[spectral_class] = (wl_arr, val_arr)

#     new_val = interp.interp1d(wl_arr, val_arr)(wavelength)

#     return new_val

# def _compute_planck_curve(wavelength, T):
#     """Compute the Planck spectral radiance.

#     Wavelength is in nm. Temperature is in K.
#     Result is in photons / cm^2 / s / nm / steradian.
#     """
#     wavelength = np.asarray(wavelength) * 1e-9 # now in m

#     # const.c: m / s
#     # const.h: m^2 kg / s
#     # const.k: m^2 kg / s^2 / K
#     # Units:
#     #   [m/s] /
#     #     ([m^4] *
#     #       exp([m^2 kg / s] * [m / s] / ([m] * [m^2 kg / s^2 / K]))
#     # = [m/s] / ([m^4] * exp([K]))
#     # = 1 / [s m^3]
#     # We want [1 / s / cm^2 / nm], so multiply result by 1e-9 * 1e-4 = 1e-13
#     return (2*const.c/
#             (wavelength**4.*(np.exp(const.h*const.c/
#                                     (wavelength*const.k*T))-1.))) * 1e-13

# def plot_planck_vs_solar_flux():
#     """Plot a scale Planck curve vs. the solar flux."""
#     # phot/cm^2/s/nm at 1 AU
#     solarflux_wl, solarflux_flux = _cisscal_solar_flux()

#     planck_flux = _compute_planck_curve(solarflux_wl, 5778)
#     # Angular size of Sun at 1 AU
#     planck_flux *= oops.PI * (0.52/2 * oops.RPD) ** 2

#     g5v_flux = _read_stellar_spectrum(solarflux_wl, 'G2')
#     g5v_flux /= (const.h*const.c)/np.array(solarflux_wl)*1e9
#     # g5v_flux *= oops.PI * (0.52/2 * oops.RPD) ** 2
#     scale_factor2 = np.sum(solarflux_flux) / np.sum(g5v_flux)

#     print(oops.PI * (0.52/2 * oops.RPD) ** 2)
#     print(scale_factor2)
#     # scale_factor2 = oops.PI * (0.52/2 * oops.RPD) ** 2 * 1e9

#     fig = plt.figure()
#     plt.plot(solarflux_wl, solarflux_flux, '-', color='red', lw=2.5,
#              label='Sun')
#     plt.plot(solarflux_wl, planck_flux, '-', lw=2.5, color='blue',
#              label='Planck')
#     plt.plot(solarflux_wl, g5v_flux*scale_factor2, '-', lw=2.5, color='green',
#              label='Sim')
#     wl, solarflux_v = _interpolate_and_convolve_2(_JOHNSON_V_WL, _JOHNSON_V,
#                                                   solarflux_wl, solarflux_flux)
#     wl, planck_v = _interpolate_and_convolve_2(_JOHNSON_V_WL, _JOHNSON_V,
#                                                solarflux_wl, planck_flux)

#     scale_factor = np.mean(solarflux_v) / np.mean(planck_v)

#     plt.plot(wl, solarflux_v, '--', color='red', lw=1, label='Sun w/V')
#     plt.plot(wl, planck_v*scale_factor, '--', color='blue', lw=1,
#              label='Planck w/V')

#     wl, solarflux_b = _interpolate_and_convolve_2(_JOHNSON_B_WL, _JOHNSON_B,
#                                                   solarflux_wl, solarflux_flux)
#     wl, planck_b = _interpolate_and_convolve_2(_JOHNSON_B_WL, _JOHNSON_B,
#                                                solarflux_wl, planck_flux)

#     scale_factor = np.mean(solarflux_b) / np.mean(planck_b)

#     plt.plot(wl, solarflux_b, ':', color='red', lw=1, label='Sun w/B')
#     plt.plot(wl, planck_b*scale_factor, ':', color='blue', lw=1,
#              label='Planck w/B')

#     plt.legend()
#     plt.title('Planck vs. Solar Flux vs. Simulated')

#     plt.show()

# def _v_magnitude_to_photon_flux(v):
#     """Return the V-band photon flux for a star with the given Johnson V
#     magnitude.

#     Returned value is in photons / cm^2 / s
#     """
#     # http://www.astro.umd.edu/~ssm/ASTR620/mags.html#flux
#     # From Bessel, M. S. 1979, PASP, 91, 589
#     # V band flux at m = 0: 3.64e-23 W/m^2/Hz = 3640 Jansky
#     # V band dlambda/lambda = 0.16

#     # https://www.iota-es.de/photon_numbers.html
#     # V band = 0.88E6 photons / cm^2 / s
#     # Jansky = 1.51e3 photons / cm^2 / s / (dlambda/lambda)

#     # jy = 3640. * 10**(-0.4*v)
#     #
#     # # flux in photons / cm^2 / s
#     # flux = jy * 1.51e3 * 0.16
#     # The Jansky and photon formulas give essentially the same answer

#     flux = 0.88e6 * 10**(-0.4*v)

#     return flux

# def _b_magnitude_to_photon_flux(b):
#     """Return the V-band photon flux for a star with the given Johnson B
#     magnitude.

#     Returned value is in photons / cm^2 / s
#     """
#     # http://www.astro.umd.edu/~ssm/ASTR620/mags.html#flux
#     # From Bessel, M. S. 1979, PASP, 91, 589
#     # B band flux at m = 0: 4260
#     # B band dlambda/lambda = 0.22
#     # https://www.iota-es.de/photon_numbers.html
#     # B band = 1.41E6 photons / cm^2 / s

#     # # Jansky = 1.51e3 photons / cm^2 / s / (dlambda/lambda)
#     # jy = 4260. * 10**(-0.4*b)
#     #
#     # # flux in photons / cm^2 / s
#     # flux = jy * 1.51e3 * 0.22
#     # The Jansky and photon formulas give essentially the same answer

#     flux = 1.41e6 * 10**(-0.4*b)

#     return flux

# def _compute_stellar_spectrum(obs, star):
#     """Compute the stellar spectrum for a given star.

#     Returned value is in photons / cm^2 / s
#     """

#     logger = logging.getLogger(_LOGGING_NAME+'.compute_stellar_spectrum')

#     # Planck is in photons / cm^2 / s / nm / steradian
#     # However, it might as well be photons / cm^2 / s / nm because we're just
#     # going to scale it later
#     wl = np.arange(100., 1600.)
#     spectrum = _read_stellar_spectrum(wl, star.spectral_class)
#     if spectrum is None:
#         spectrum = _compute_planck_curve(wl, star.temperature)

#     wl_new, spec_v = _interpolate_and_convolve_2(_JOHNSON_V_WL, _JOHNSON_V,
#                                                  wl, spectrum)
#     # Total photons seen through V filter
#     spec_v_sum = np.sum(spec_v)
#     # Predicted photons seen through V - photons / cm^2 / s
#     predicted_v = _v_magnitude_to_photon_flux(star.johnson_mag_v)
# #    logger.debug('Star %9d Temp %9.2f Predicted V-band total flux %e',
# #                 star.unique_number, star.temperature, predicted_v)
#     scale_factor_v = predicted_v / spec_v_sum

#     wl_new, planck_b = _interpolate_and_convolve_2(_JOHNSON_B_WL, _JOHNSON_B,
#                                                    wl, spectrum)
#     # Total photons seen through B filter
#     spec_b_sum = np.sum(planck_b)
#     # Predicted photons seen through V - photons / cm^2 / s
#     predicted_b = _b_magnitude_to_photon_flux(star.johnson_mag_b)
# #    logger.debug('Star %9d Temp %9.2f Predicted V-band total flux %e',
# #                 star.unique_number, star.temperature, predicted_v)
#     scale_factor_b = predicted_b / spec_b_sum

#     ret_spectrum = spectrum*(scale_factor_b+scale_factor_v)/2
#     logger.debug('Star %9d MAG %6.3f BMAG %6.3f '+
#                  'VMAG %6.3f SCLASS %3s TEMP %6d Scale V %e Scale B %e '+
#                  'TOTPHOT %e',
#                  star.unique_number,
#                  0 if star.vmag is None else star.vmag,
#                  0 if star.johnson_mag_b is None else star.johnson_mag_b,
#                  0 if star.johnson_mag_v is None else star.johnson_mag_v,
#                  '' if star.spectral_class is None else star.spectral_class,
#                  0 if star.temperature is None else star.temperature,
#                  scale_factor_v, scale_factor_b, np.sum(ret_spectrum))

#     # Return is in photons / cm^2 / s
#     return wl, ret_spectrum

# def _compute_dn_from_spectrum(obs, spectrum_wl, spectrum):
#     """Compute the original DN expected from a given spectrum.

#     The spectrum is in photons / cm^2 / s / nm
#     """

#     logger = logging.getLogger(_LOGGING_NAME+'._compute_dn_from_spectrum')

#     # Read in filter transmission
#     systrans_wl, systrans_xmit = _cisscal_filter_transmission(obs)

#     # Read in QE correction
#     qecorr_wl, qecorr_val = _cisscal_qe_correction(obs)

#     conv_wl, conv_flux = _interpolate_and_convolve_3(
#                              systrans_wl, systrans_xmit, qecorr_wl, qecorr_val,
#                              spectrum_wl, spectrum)

#     conv_flux_sum = integrate.simps(conv_flux, conv_wl)
#     # photons / cm^2 / s

#     logger.debug('Total flux through %s+%s = %f photons/cm^2/s',
#                  obs.filter1, obs.filter2, conv_flux_sum)

#     if False: # Make True to compare flux with a CISSCAL-calibrated file
#         weights_wl, weights = _interpolate_and_convolve_2(systrans_wl,
#                                                           systrans_xmit,
#                                                           qecorr_wl,
#                                                           qecorr_val)
#         assert conv_wl[0] == weights_wl[0] and conv_wl[-1] == weights_wl[-1]

#         conv_flux_avg = conv_flux_sum / integrate.simps(weights, weights_wl)
#         conv_flux_avg /= CISSCAL_DETECTOR_SOLID_ANGLE[obs.clean_detector]
#         logger.debug('Total flux through %s+%s = %e /nm/sr',
#                      obs.filter1, obs.filter2, conv_flux_avg)

#     # 3) cassimg__dividebyareapixel.pro
#     # We want the total flux through the entire aperture, not per pixel
#     data = conv_flux_sum * _CISSCAL_DETECTOR_OPTICS_AREA[obs.clean_detector]
#     # photons / s for entire detector

#     # 2) cassimg__dividebyexpot.pro
#     #        We ignore shutter offset timing for our purposes
#     #        DATA = ELECTRONS / (ExpT/1000) [ExpT in ms]
#     electrons = data * obs.texp # texp is already in sec
#     # photons

#     # 1) cassimg__dntoelectrons.pro
#     #        ELECTRONS = DN * GAIN / GAIN_RATIO[GAIN_MODE_ID]

#     dn = (electrons /
#             (_CISSCAL_DETECTOR_GAIN[obs.clean_detector] /
#              _CISSCAL_DETECTOR_GAIN_RATIO[obs.clean_detector][obs.gain_mode]))
#     logger.debug('Returned Total DN = %f', dn)

#     return dn

# STAR_FINE_CALIBRATION = {
#     ('NAC','CLEAR','B'): 1.14,
#     ('NAC','UV3','B'): 0.47,
#     ('NAC','BL1','B'): 0.93,
#     ('NAC','GRN','B'): 1.23,
#     ('NAC','RED','B'): 1.34,
#     ('NAC','IR1','B'): 1.55,
#     ('NAC','IR3','B'): 1.95,
#     ('NAC','CLEAR','A'): 1.11,
#     ('NAC','UV3','A'): 0.65,
#     ('NAC','BL2','A'): 0.73,
#     ('NAC','BL1','A'): 0.88,
#     ('NAC','GRN','A'): 1.03,
#     ('NAC','CB1','A'): 1.06,
#     ('NAC','MT1','A'): 1.03,
#     ('NAC','RED','A'): 1.14,
#     ('NAC','HAL','A'): 1.01,
#     ('NAC','CB2','A'): 1.12,
#     ('NAC','MT2','A'): 1.17,
#     ('NAC','CB3','A'): 1.08,
#     ('NAC','MT3','A'): 1.22,
#     ('NAC','IR1','A'): 1.30,
#     ('NAC','IR2','A'): 1.27,
#     ('NAC','IR3','A'): 1.14,
#     ('NAC','IR4','A'): 1.08,
#     ('NAC','CLEAR','F'): 1.16,
#     ('NAC','UV3','F'): 0.77,
#     ('NAC','BL2','F'): 3.48,
#     ('NAC','BL1','F'): 0.87,
#     ('NAC','GRN','F'): 0.95,
#     ('NAC','CB1','F'): 1.04,
#     ('NAC','RED','F'): 1.10,
#     ('NAC','CB2','F'): 1.10,
#     ('NAC','MT2','F'): 1.48,
#     ('NAC','CB3','F'): 1.51,
#     ('NAC','MT3','F'): 1.50,
#     ('NAC','IR1','F'): 1.15,
#     ('NAC','IR2','F'): 3.06,
#     ('NAC','IR3','F'): 2.09,
#     ('NAC','CLEAR','G'): 1.07,
#     ('NAC','UV3','G'): 0.71,
#     ('NAC','BL2','G'): 0.72,
#     ('NAC','BL1','G'): 0.80,
#     ('NAC','GRN','G'): 0.91,
#     ('NAC','CB1','G'): 1.07,
#     ('NAC','MT1','G'): 1.06,
#     ('NAC','RED','G'): 1.08,
#     ('NAC','CB2','G'): 1.11,
#     ('NAC','MT2','G'): 1.16,
#     ('NAC','CB3','G'): 1.33,
#     ('NAC','MT3','G'): 1.26,
#     ('NAC','IR1','G'): 1.18,
#     ('NAC','IR2','G'): 1.22,
#     ('NAC','IR3','G'): 2.04,
#     ('NAC','IR4','G'): 1.32,
#     ('NAC','CLEAR','K'): 1.02,
#     ('NAC','UV3','K'): 0.37,
#     ('NAC','BL1','K'): 0.70,
#     ('NAC','GRN','K'): 0.86,
#     ('NAC','CB1','K'): 0.86,
#     ('NAC','MT1','K'): 0.92,
#     ('NAC','RED','K'): 0.93,
#     ('NAC','HAL','K'): 0.89,
#     ('NAC','CB2','K'): 0.94,
#     ('NAC','MT2','K'): 0.99,
#     ('NAC','CB3','K'): 1.21,
#     ('NAC','MT3','K'): 1.11,
#     ('NAC','IR1','K'): 0.92,
#     ('NAC','IR2','K'): 1.03,
#     ('NAC','IR3','K'): 1.05,
#     ('NAC','IR4','K'): 1.24,
#     ('NAC','CLEAR','M'): 0.97,
#     ('NAC','GRN','M'): 0.95,
#     ('NAC','RED','M'): 0.99,
#     ('NAC','IR1','M'): 0.39,
#     ('NAC','IR2','M'): 0.27,
#     ('NAC','IR3','M'): 0.73,
#     ('NAC','CLEAR','N'): 0.42,
#     ('NAC','BL1','N'): 0.73,
#     ('NAC','GRN','N'): 0.97,
#     ('NAC','RED','N'): 0.77,
#     ('NAC','IR1','N'): 0.40,
#     ('NAC','IR2','N'): 0.25,
#     ('NAC','CLEAR','P'): 0.34,
#     ('NAC','RED','P'): 0.76,
#     ('WAC','CLEAR','B'): 1.11,
#     ('WAC','VIO','B'): 0.88,
#     ('WAC','BL1','B'): 1.05,
#     ('WAC','GRN','B'): 1.20,
#     ('WAC','RED','B'): 1.38,
#     ('WAC','HAL','B'): 1.44,
#     ('WAC','CB2','B'): 1.51,
#     ('WAC','MT2','B'): 1.48,
#     ('WAC','IR2','B'): 1.53,
#     ('WAC','IR3','B'): 1.44,
#     ('WAC','CLEAR','A'): 1.28,
#     ('WAC','VIO','A'): 0.90,
#     ('WAC','BL1','A'): 0.87,
#     ('WAC','GRN','A'): 1.40,
#     ('WAC','RED','A'): 1.46,
#     ('WAC','HAL','A'): 1.40,
#     ('WAC','CB2','A'): 1.54,
#     ('WAC','MT2','A'): 1.52,
#     ('WAC','CB3','A'): 1.74,
#     ('WAC','MT3','A'): 1.71,
#     ('WAC','IR1','A'): 1.83,
#     ('WAC','IR2','A'): 1.69,
#     ('WAC','IR3','A'): 1.88,
#     ('WAC','IR4','A'): 1.86,
#     ('WAC','IR5','A'): 1.65,
#     ('WAC','CLEAR','F'): 1.27,
#     ('WAC','VIO','F'): 0.91,
#     ('WAC','BL1','F'): 0.86,
#     ('WAC','GRN','F'): 1.31,
#     ('WAC','RED','F'): 1.36,
#     ('WAC','HAL','F'): 1.26,
#     ('WAC','CB2','F'): 1.34,
#     ('WAC','MT2','F'): 1.35,
#     ('WAC','CB3','F'): 1.47,
#     ('WAC','MT3','F'): 1.46,
#     ('WAC','IR1','F'): 2.62,
#     ('WAC','IR2','F'): 1.46,
#     ('WAC','IR3','F'): 1.60,
#     ('WAC','IR4','F'): 1.53,
#     ('WAC','CLEAR','G'): 1.47,
#     ('WAC','VIO','G'): 1.04,
#     ('WAC','BL1','G'): 0.91,
#     ('WAC','GRN','G'): 1.38,
#     ('WAC','RED','G'): 1.59,
#     ('WAC','HAL','G'): 1.47,
#     ('WAC','CB2','G'): 1.72,
#     ('WAC','MT2','G'): 1.45,
#     ('WAC','MT3','G'): 1.49,
#     ('WAC','IR1','G'): 2.68,
#     ('WAC','IR2','G'): 1.69,
#     ('WAC','IR3','G'): 2.24,
#     ('WAC','CLEAR','K'): 1.22,
#     ('WAC','VIO','K'): 0.82,
#     ('WAC','BL1','K'): 0.81,
#     ('WAC','GRN','K'): 1.27,
#     ('WAC','RED','K'): 1.44,
#     ('WAC','HAL','K'): 2.11,
#     ('WAC','CB2','K'): 1.65,
#     ('WAC','MT2','K'): 1.58,
#     ('WAC','CB3','K'): 2.02,
#     ('WAC','MT3','K'): 2.00,
#     ('WAC','IR1','K'): 1.19,
#     ('WAC','IR2','K'): 1.69,
#     ('WAC','IR3','K'): 2.04,
#     ('WAC','IR4','K'): 2.34,
#     ('WAC','IR5','K'): 2.22,
#     ('WAC','CLEAR','M'): 1.01,
#     ('WAC','BL1','M'): 1.18,
#     ('WAC','GRN','M'): 1.24,
#     ('WAC','RED','M'): 1.21,
#     ('WAC','HAL','M'): 0.93,
#     ('WAC','CB2','M'): 1.04,
#     ('WAC','MT2','M'): 0.94,
#     ('WAC','CB3','M'): 2.31,
#     ('WAC','MT3','M'): 2.37,
#     ('WAC','IR1','M'): 1.10,
#     ('WAC','IR2','M'): 1.20,
#     ('WAC','IR3','M'): 1.05,
#     ('WAC','IR4','M'): 3.51,
#     ('WAC','IR5','M'): 3.23,
#     ('WAC','CLEAR','N'): 0.50,
#     ('WAC','GRN','N'): 1.30,
#     ('WAC','RED','N'): 0.96,
#     ('WAC','HAL','N'): 0.61,
#     ('WAC','CB2','N'): 0.27,
#     ('WAC','MT2','N'): 0.44,
#     ('WAC','CB3','N'): 1.07,
#     ('WAC','MT3','N'): 0.43,
#     ('WAC','IR1','N'): 0.40,
#     ('WAC','IR2','N'): 0.25,
#     ('WAC','IR3','N'): 0.27,
#     ('WAC','IR4','N'): 1.52,
#     ('WAC','IR5','N'): 1.16,
#     ('WAC','CLEAR','P'): 0.47,
#     ('WAC','GRN','P'): 1.42,
#     ('WAC','RED','P'): 0.97,
#     ('WAC','HAL','P'): 0.65,
#     ('WAC','CB2','P'): 0.35,
#     ('WAC','MT2','P'): 0.48,
#     ('WAC','CB3','P'): 0.19,
#     ('WAC','MT3','P'): 0.22,
#     ('WAC','IR1','P'): 0.35,
#     ('WAC','IR2','P'): 0.23,
#     ('WAC','IR3','P'): 0.21,
#     ('WAC','IR4','P'): 0.19,
#     ('WAC','IR5','P'): 0.30,
# }

# _CLIP_PSF_CACHE = {}

# def compute_dn_from_star(obs, star):
#     """Compute the theoretical integrated DN for a star."""
#     spectrum_wl, spectrum = _compute_stellar_spectrum(obs, star)
#     dn = _compute_dn_from_spectrum(obs, spectrum_wl, spectrum)

#     # Adjust predicted dn based on stellar class and filter using calibration
#     # data from various Cassini star fields.
#     correction = 1.0
#     f1 = obs.filter1 or ''
#     f2 = obs.filter2 or ''
#     filter_name = simple_filter_name_f1f2(f1, f2, True)
#     if filter_name.startswith('P+'):
#         filter_name = filter_name.split('+')[1]
#     if filter_name.find('+') != -1:
#         # Just take the first filter for lack of anything better to do
#         filter_name = filter_name.split('+')[0]

#     sclass = star.spectral_class
#     if sclass[0] != 'M':
#         sclass = sclass[0]
#     elif sclass >= 'M7':
#         sclass = 'P'
#     elif sclass >= 'M5':
#         sclass = 'N'
#     else:
#         sclass = 'M'

#     camera = obs.clean_detector

#     key = (camera, filter_name, sclass)
#     if key not in STAR_FINE_CALIBRATION:
#         correction = 1.
#     else:
#         correction = STAR_FINE_CALIBRATION[key]

#     new_dn = dn * correction

#     # Now adjust for clipping in the CCD - the maximum DN is 4095
#     # However, things start to spread out and act weird around 1024
#     if camera not in _CLIP_PSF_CACHE:
#         gausspsf = GaussianPSF(sigma=spindoctor.config.PSF_SIGMA[obs.clean_detector])
#         psf = gausspsf.eval_rect((15,15), offset=(0.5,0.5))

#         _CLIP_PSF_CACHE[camera] = psf

#     psf = _CLIP_PSF_CACHE[camera]
#     psf_dn = psf * new_dn
#     psf_dn[psf_dn > 4095] = 4095

#     return np.sum(psf_dn)
