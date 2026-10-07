"""
id01_h5.py
==========

Helper functions and classes to read beamline scan data out of ESRF/BLISS-style
HDF5 files (one group per scan, named "<scan_no>.1", with "measurement/",
"instrument/positioners/" etc. sub-groups).

WHAT THIS FILE IS FOR
----------------------
You give it a scan number and a filename, and it hands you back numpy arrays:
motor positions, detector images, counters (ROIs, timers, ...).

HOW TO DEBUG SOMETHING THAT ISN'T WORKING
------------------------------------------
1. Turn on `verbose=True` (default for most things) - it prints the scan
   command, the detector name it guessed, and the list of available
   channels/counters. That's almost always enough to see what's wrong.
2. If a channel/counter/motor "doesn't exist", the printed message tells you
   the exact HDF5 path it tried to read. Open the file yourself with:
       import h5py
       h5f = h5py.File(filename, "r")
       print(list(h5f["<scan_no>.1/measurement"].keys()))
   to compare against what the code expected.
3. Errors are no longer swallowed silently: if something fails, you get a
   printed message saying *what* failed and *why* (the original Python
   exception message), not just "problem!". Read that message first.

HOW TO ADD SUPPORT FOR A NEW DETECTOR
---------------------------------------
Add its HDF5 key (the name under "measurement/") to the KNOWN_DETECTORS list
just below these comments. That's the only change needed for
`Scan.get_detector_name()` to recognize it.

HOW TO ADD SUPPORT FOR A NEW SCAN TYPE
-----------------------------------------
1. Write a new class that inherits from `Scan` (copy `DmeshScan` or
   `StandardScan` as a starting point - they're the simplest).
2. In `__init__`, parse whatever the scan command string (`self.command`)
   gives you to figure out the motor names / scan shape.
3. Add a line for it in `open_scan()` at the bottom of this file, matching on
   a keyword that appears in the scan command.

A NOTE ABOUT `self.command`
-----------------------------
`self.command` is deliberately kept as the *raw* `str(bytes)` representation
coming straight out of h5py (e.g. `"b'lookupscan 10 [mm1,mm2]'"`, with the
`b'...'` wrapper included) rather than a cleanly decoded string. Some of the
motor-name parsing below (see `LookupScan`, `SXDM_Scan`) slices this string
assuming that wrapper is there. If you "clean up" this decoding, double check
those two classes still parse motor names correctly on a real file before
trusting them again.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import hdf5plugin  # noqa: F401  (import needed so h5py can decompress detector data - do not remove)
from datetime import datetime

### -----------------------------------------------------------------------
### Config you're likely to want to tweak
### -----------------------------------------------------------------------

# Detector names this file knows how to recognize, in the order they're
# checked. If your beamline gets a new detector, just add its HDF5 key here -
# nothing else needs to change.
KNOWN_DETECTORS = ["mpx1x4", "eiger2M", "mpxgaas", "andor_zyla", "citius"]

# Datetime format used for "start_time" / "end_time" fields in the hdf5 file.
HDF5_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f%z"


### -----------------------------------------------------------------------
### small internal helpers (not meant to be called directly by users)
### -----------------------------------------------------------------------


def _decode(value):
    """
    h5py sometimes hands back `bytes` for string datasets and sometimes a
    plain `str`, depending on how the file was written. This makes sure we
    always get a plain `str` back, regardless of which one it was.
    (Only used for things like dates/titles where we then parse the text -
    NOT used for `self.command`, see the module docstring above.)
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _parse_hdf5_datetime(raw_value):
    """Parse a start_time/end_time hdf5 field into a python datetime."""
    return datetime.strptime(_decode(raw_value), HDF5_DATETIME_FORMAT)


def _warn(context, message):
    """Consistent, greppable warning messages: '[Scan.get_energy] ...'"""
    print(f"[{context}] {message}")


### -----------------------------------------------------------------------
### hdf5 utility functions (module-level, no class needed)
### -----------------------------------------------------------------------


def get_scan_list(filename, verbose=True):
    """
    Return the sorted list of every scan number found in <filename>.
    If verbose, also prints each scan number alongside its title/command.
    """
    with h5.File(filename, "r") as h5f:
        scans = sorted(int(float(key)) for key in h5f.keys())

        if verbose:
            print("Available scans in hdf5 file:")
            for scan in scans:
                title = h5f["%i.1/title" % scan][()]
                print("%i ... %s" % (scan, _decode(title)))

    return scans


def get_command(scan_no, filename):
    """
    Return the scan command/title (as text) for scan <scan_no> in <filename>.
    """
    with h5.File(filename, "r") as h5f:
        return str(h5f["{}.1".format(scan_no)]["title"][()])


#################################################
#####           Master Scan class          ######
#################################################

# Scan class from Ewen Bellec


class Scan:
    def __init__(self, filename, scan_nb, verbose=True):
        """
        filename = path to hdf5 file (dataset level)
        scan_nb = scan number (accepts -n nomenclature, e.g. -1 = last scan)
        verbose = show some output while loading (recommended: keep True
                  when you're setting up a new analysis, it's the fastest
                  way to see what's actually in the file)
        """
        self.h5file = filename
        self.keys = self.sort_h5_keys()
        self.data_type = 'ID01'

        if scan_nb < 0:
            self.scan_string = self.keys[scan_nb]
        else:
            self.scan_string = "{}.1".format(scan_nb)

        self.verbose = verbose
        with h5.File(self.h5file, "r") as h5f:
            self.command = str(h5f[self.scan_string]["title"][()])

        if verbose:
            print(f"Scan no: {self.scan_string}")
            print(f"{self.command}")

        try:
            with h5.File(self.h5file, "r") as h5f:
                self.sample = h5f[self.scan_string]["sample/name"][()].decode('UTF-8')
        except KeyError:
            if self.verbose:
                _warn("Scan.__init__", "no 'sample/name' found for this scan (not necessarily a problem).")

        # Figure out which detector was used, then its image shape. If your
        # beamline has a detector not in KNOWN_DETECTORS (see top of file),
        # get_detector_name() will print the list of channels it *did* find so
        # you can add the right name to that list.
        try:
            self.get_detector_name()
        except KeyError as e:
            _warn("Scan.__init__", f"could not look for a detector (no 'measurement' group for "
                                     f"this scan?): {e}")
            self.detector = None

        if self.detector is not None:
            try:
                self.get_detector_shape()
            except KeyError as e:
                _warn("Scan.__init__", f"found detector '{self.detector}' but could not read its shape "
                                        f"(instrument/{self.detector}/dim_j or dim_i missing?): {e}")

    def show_scan_info(self):
        print(self.scan_string, self.command)

    def show_info(self):
        outs = ""
        for n, key in enumerate(self.keys):
            with h5.File(self.h5file, "r") as h5f:
                outs += key + " " + h5f[key + "/title"][()] + "\n"
        print(outs)
        return outs

    def sort_h5_keys(self):
        with h5.File(self.h5file, "r") as h5f:
            keys = list(h5f.keys())
        index_sort = np.argsort([int(key.split(".")[0]) for key in keys])
        keys_sort = [keys[index] for index in index_sort]
        return keys_sort

    def get_motor_position(self, motor_name):
        with h5.File(self.h5file, "r") as h5f:
            path = "instrument/positioners/{}".format(motor_name)
            try:
                motor_pos = h5f[self.scan_string][path][()]
            except KeyError:
                available = list(h5f[self.scan_string]["instrument/positioners/"].keys())
                raise KeyError(
                    f"motor '{motor_name}' not found in scan {self.scan_string} "
                    f"(path tried: {path}). Available motors: {available}"
                ) from None
        return motor_pos

    def get_all_motor_dictionary(self):
        with h5.File(self.h5file, "r") as h5f:
            motor_dict = {}
            for motor_name in h5f[self.scan_string]["instrument/positioners/"].keys():
                motor_dict[motor_name] = h5f[self.scan_string][
                    "instrument/positioners/{}".format(motor_name)
                ][()]
        setattr(self, "motor_dict", motor_dict)
        return motor_dict

    def get_detector_name(self):
        """
        Guess the detector name by checking which of KNOWN_DETECTORS is
        present under this scan's 'measurement/' group.
        Sets self.detector (None if nothing recognized was found).
        """
        with h5.File(self.h5file, "r") as h5f:
            measurement_keys = list(h5f[self.scan_string]["measurement/"].keys())

        detector = None
        for candidate in KNOWN_DETECTORS:
            if candidate in measurement_keys:
                detector = candidate
                break

        if detector is None:
            _warn(
                "Scan.get_detector_name",
                "none of the known detectors were recognized for scan "
                f"{self.scan_string}.\n"
                f"  known detectors : {KNOWN_DETECTORS}\n"
                f"  channels found  : {measurement_keys}\n"
                "  -> if one of the channels above IS your detector, add its "
                "name to KNOWN_DETECTORS at the top of id01_h5.py, or set "
                "`scan.detector = '<name>'` by hand.",
            )
        elif self.verbose:
            print("detector :", detector)

        self.detector = detector
        return detector

    def get_detector_shape(self):
        with h5.File(self.h5file, "r") as h5f:
            shape_0 = h5f[self.scan_string]["instrument/{}/dim_j".format(self.detector)][()]
            shape_1 = h5f[self.scan_string]["instrument/{}/dim_i".format(self.detector)][()]
        detector_shape = (shape_0, shape_1)

        if self.verbose:
            print("detector shape :", detector_shape)

        self.detector_shape = detector_shape
        return detector_shape

    def get_det_calib_info(self):
        det_calib = {}
        with h5.File(self.h5file, "r") as h5f:
            for key in ("distance", "beam_center_x", "beam_center_y", "x_pixel_size", "y_pixel_size"):
                det_calib[key] = h5f[self.scan_string][
                    "instrument/{}/{}".format(self.detector, key)
                ][()]
        self.det_calib = det_calib
        return det_calib

    def get_energy(self):
        """
        Energy in eV, from the monochromator wavelength when available,
        falling back to the 'mononrj' positioner (assumed to be in keV).
        """
        with h5.File(self.h5file, "r") as h5f:
            try:
                wavelength_m = h5f[
                    "{}/instrument/monochromator/WaveLength".format(self.scan_string)
                ][()]
                energy = (12.39842 / (wavelength_m * 1e10)) * 1e3  # energy in eV
            except KeyError:
                if self.verbose:
                    _warn("Scan.get_energy", "no monochromator/WaveLength found, "
                                              "falling back to 'mononrj' positioner.")
                energy = h5f[
                    "{}/instrument/positioners/mononrj".format(self.scan_string)
                ][()] * 1e3  # mononrj assumed in keV -> eV
        self.energy = energy
        return energy

    def get_all_counters_list(self, print_list=False):
        with h5.File(self.h5file, "r") as h5f:
            counter_list = list(h5f[self.scan_string]["measurement"].keys())
        if print_list:
            print(counter_list)
        return counter_list

    def get_counter(self, counter_name):
        with h5.File(self.h5file, "r") as h5f:
            try:
                return h5f[self.scan_string]["measurement/{}".format(counter_name)][()]
            except KeyError:
                available = list(h5f[self.scan_string]["measurement"].keys())
                raise KeyError(
                    f"counter '{counter_name}' not found in scan {self.scan_string}. "
                    f"Available counters: {available}"
                ) from None

    def print_counters_list(self, return_list=False):
        with h5.File(self.h5file, "r") as h5f:
            counter_list = list(h5f[self.scan_string]["measurement"].keys())

        print("available counters :")
        for counter in counter_list:
            print(counter, end="    ")
        print()
        if return_list:
            return counter_list

    def print_sum_counters_list(self, return_list=False):
        with h5.File(self.h5file, "r") as h5f:
            counter_list = list(h5f[self.scan_string]["measurement"].keys())

        counter_sum_list = []
        print("available sum ROIs :")
        for counter in counter_list:
            if (
                (self.detector) in counter
                and ("avg" not in counter)
                and ("max" not in counter)
                and ("min" not in counter)
                and ("std" not in counter)
                and (self.detector != counter)
            ):
                print(counter, end="    ")
                counter_sum_list.append(counter)
        print()
        if return_list:
            return counter_sum_list

    def get_detector_sum(self, plot=False):
        with h5.File(self.h5file, "r") as h5f:
            nb_img = h5f[self.scan_string][
                "measurement/{}".format(self.detector)
            ].__len__()
            for n in range(nb_img):
                if n == 0:
                    detector_sum = h5f[self.scan_string][
                        "measurement/{}".format(self.detector)
                    ].__getitem__(n)
                else:
                    detector_sum += h5f[self.scan_string][
                        "measurement/{}".format(self.detector)
                    ].__getitem__(n)

        if plot:
            plt.matshow(np.log(detector_sum))
            plt.title("log of sum of the detector images", fontsize=15)

        self.detector_sum = detector_sum
        return detector_sum

    def get_raw_data(self, roi=None):
        with h5.File(self.h5file, "r") as h5f:
            dataset = h5f[self.scan_string]["measurement/{}".format(self.detector)]
            if roi is None:
                data = dataset[()]
            elif len(roi) == 4:
                data = dataset[:, roi[0]:roi[1], roi[2]:roi[3]]
            elif len(roi) == 6:
                data = dataset[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
            else:
                raise ValueError(
                    f"get_raw_data: `roi` must have 4 elements ([y0,y1,x0,x1], for a "
                    f"3D stack) or 6 elements ([z0,z1,y0,y1,x0,x1]) - got {len(roi)}: {roi}"
                )
        return data

    def get_data(self, roi=None):

        data = self.get_raw_data(roi=roi)

        if self.verbose:
            print("data.shape", data.shape)

        #         self.data = data # might not be a good idea to save it in scan object if we return it as well
        return data

    def get_start_end_time(self, return_seconds=False):
        with h5.File(self.h5file, 'r') as h5f:
            start_time = _parse_hdf5_datetime(h5f[self.scan_string]['start_time'][()])
            end_time = _parse_hdf5_datetime(h5f[self.scan_string]['end_time'][()])

        if return_seconds:
            return start_time.timestamp(), end_time.timestamp()
        else:
            return start_time, end_time

    def get_point_times(self, return_seconds=False):
        if "SXDM_Scan" in str(type(self)):
            print(
                "Error : get_point_times doesn't work yet with sxdm scans. There might be a problem with the time counter. To be fixed!"
            )
            return

        start_time, end_time = self.get_start_end_time(return_seconds=False)

        elapsed_time = self.get_counter("elapsed_time")
        elapsed_time = elapsed_time + start_time.timestamp()
        elapsed_time = np.atleast_1d(elapsed_time)

        if return_seconds:
            return np.array(elapsed_time)
        else:
            time = [
                datetime.fromtimestamp(t).strftime("%d-%b-%Y (%H:%M:%S.%f)")
                for t in elapsed_time
            ]
            return np.array(time)

    def print_scan_motors(self, motor_list):
        print('scan no :', self.scan_string)
        for motor in motor_list:
            print(motor, round(np.nanmean(self.get_motor_position(motor)), 2))
        return


##################################################################################################################################
######################################           Dscan and Ascan          ########################################################
##################################################################################################################################


class StandardScan(Scan):
    def __init__(self, filename, scan_nb, verbose=False):
        super().__init__(filename, scan_nb, verbose=verbose)

        _ = self.get_dscan_motor_position()

    def get_dscan_motor_position(self):
        motor_name = self.command.split()[1]
        motor = self.get_motor_position(motor_name)

        if self.verbose:
            print("motor : {}".format(motor_name))

        self.motor = motor
        self.motor_name = motor_name

        return motor, motor_name

    def get_roi_data(self, roi_name, plot=False, fig_title=''):
        with h5.File(self.h5file, "r") as h5f:
            roidata = h5f[self.scan_string]["measurement/{}".format(roi_name)][()]
        if plot:
            plt.figure()
            plt.plot(self.motor, roidata, ".-")
            plt.xlabel(self.motor_name, fontsize=15)
            plt.ylabel(roi_name, fontsize=15)
            plt.title(fig_title, fontsize=15)
        return roidata


##################################################################################################################################
#########################################           LookupScan          ##########################################################
##################################################################################################################################

class LookupScan(Scan):
    def __init__(self, filename, scan_nb, verbose=False):

        super().__init__(filename, scan_nb, verbose=verbose)

        _ = self.get_lookupscan_motor_position()

    def get_lookupscan_motor_position(self):
        # NOTE: this parsing relies on self.command looking like
        # "b'lookupscan ... [mm1,mm2]'" (no spaces/quotes inside the
        # brackets) - see the module docstring's note about self.command.
        motor_keys = self.command.split()[-1][1:-2].split(",")
        motor_dict = {}
        for motor in motor_keys:
            motor_dict[motor] = self.get_motor_position(motor)

        if self.verbose:
            print("motors : ", motor_keys)

        self.motor_dict = motor_dict
        return motor_dict


##################################################################################################################################
#####################################           Dmesh scan          #######################################################
##################################################################################################################################


class DmeshScan(Scan):
    def __init__(self, filename, scan_nb, verbose=False):
        super().__init__(filename, scan_nb, verbose=verbose)

        _ = self.get_mesh_motor_position()

    def get_mesh_motor_position(self):
        motor1_name = self.command.split()[1]
        motor2_name = self.command.split()[5]

        motor1 = self.get_motor_position(motor1_name)
        motor2 = self.get_motor_position(motor2_name)

        shape = (int(self.command.split()[-2]) + 1, int(self.command.split()[4]) + 1)
        motor1 = np.reshape(motor1, shape)
        motor2 = np.reshape(motor2, shape)

        if self.verbose:
            print("motor1 : {}".format(motor1_name))
            print("motor2 : {}".format(motor2_name))

        self.motor1 = motor1
        self.motor2 = motor2
        self.motor1_name = motor1_name
        self.motor2_name = motor2_name

        return motor1, motor2, motor1_name, motor2_name

    def get_roi_data(self, roi_name, plot=False):
        with h5.File(self.h5file, "r") as h5f:
            roidata = h5f[self.scan_string]["measurement/{}".format(roi_name)][()]
        roidata = roidata.reshape(self.motor1.shape)

        if plot:
            plot_2d_map_sxdm_dmesh(self, roidata, roi_name)

        return roidata

    def get_data(self, roi=None):
        data = self.get_raw_data(roi=roi)
        data = np.reshape(data, (self.motor1.shape) + data.shape[-2:])

        if self.verbose:
            print("data.shape", data.shape)
            print(
                "shape dimensions : ( {}, {}, {}, {})".format(
                    self.motor2_name,
                    self.motor1_name,
                    "detector vertical axis",
                    "detector horizontal axis",
                )
            )
        #         self.data = data # might not be a good idea to save it in scan object if we return it as well
        return data


##################################################################################################################################
######################################             2D SXDM scan            #######################################################
##################################################################################################################################


class SXDM_Scan(Scan):
    def __init__(self, filename, scan_nb, verbose=False):
        super().__init__(filename, scan_nb, verbose=verbose)

        _ = self.get_sxdm_motors_position()

    def get_sxdm_motors_position(self):
        motor1_name = self.command.split()[1].replace(',', '')  # [:-1]
        motor2_name = self.command.split()[5].replace(',', '')  # [:-1]

        with h5.File(self.h5file, "r") as h5f:
            try:
                motor1 = h5f[self.scan_string][
                    "instrument/{}_position/value".format(motor1_name)
                ][()]
                motor2 = h5f[self.scan_string][
                    "instrument/{}_position/value".format(motor2_name)
                ][()]
            except KeyError:
                motor1 = h5f[self.scan_string][
                    "instrument/{}/value".format(motor1_name)
                ][()]
                motor2 = h5f[self.scan_string][
                    "instrument/{}/value".format(motor2_name)
                ][()]

        # Reshape the motor positions into 2D arrays
        dim1 = int(self.command.split()[8].replace(',', ''))  # [:-1])
        dim2 = int(self.command.split()[4].replace(',', ''))  # [:-1])
        motor1 = motor1.reshape(dim1, dim2)
        motor2 = motor2.reshape(dim1, dim2)

        if self.verbose:
            print("motor1 : {}".format(motor1_name))
            print("motor2 : {}".format(motor2_name))

        self.motor1 = motor1
        self.motor2 = motor2
        self.motor1_name = motor1_name
        self.motor2_name = motor2_name
        return motor1, motor2, motor1_name, motor2_name

    def get_roi_data(self, roi_name, plot=False):
        with h5.File(self.h5file, "r") as h5f:
            roidata = h5f[self.scan_string]["measurement/{}".format(roi_name)][()]
        roidata = roidata.reshape(self.motor1.shape)

        if plot:
            plot_2d_map_sxdm_dmesh(self, roidata, roi_name)

        return roidata

    def get_data(self, roi=None):
        data = self.get_raw_data(roi=roi)

        data = np.reshape(data, self.motor1.shape + data.shape[-2:])

        if self.verbose:
            print("data.shape", data.shape)
            print(
                "shape dimensions : ( {}, {}, {}, {})".format(
                    self.motor2_name,
                    self.motor1_name,
                    "detector vertical axis",
                    "detector horizontal axis",
                )
            )

        #         self.data = data # might not be a good idea to save it in scan object if we return it as well
        return data


##################################################################################################################################
######################################             3D SXDM scan            #######################################################
##################################################################################################################################


class SXDM_3D_Scan:
    def __init__(self, filename, motor3_name=None, verbose=True):
        self.h5file = filename
        self.scan1 = SXDM_Scan(filename, 1)
        self.verbose = verbose
        self.detector = self.scan1.detector

        self.nb_scan = len(get_scan_list(filename, verbose=False))

        if motor3_name is None:
            self.motor3_name = self.find_third_motor()
        else:
            self.motor3_name = motor3_name

        self.get_sxdm_3d_motors_position()

        if self.verbose:
            print("detector :", self.detector)
            print("motor3_name :", self.motor3_name)
            print(
                "(if motor3_name is wrong, please put the correct motor as a string in motor3_name argument)"
            )

    def find_third_motor(self):
        # I only check if the third motor is eta or phi. Hope that will be enough
        phi = np.zeros(self.nb_scan)
        eta = np.zeros(self.nb_scan)
        for n in range(self.nb_scan):
            scan_nb = n + 1
            sxdm_scan = SXDM_Scan(self.h5file, scan_nb, verbose=False)
            phi[n] += sxdm_scan.get_motor_position("phi")
            eta[n] += sxdm_scan.get_motor_position("eta")
        motor3_name = None
        if np.all(eta == eta[0]) and (not np.all(phi == phi[0])):
            motor3_name = "phi"
        if (not np.all(eta == eta[0])) and np.all(phi == phi[0]):
            motor3_name = "eta"
        if (not np.all(eta == eta[0])) and (not np.all(phi == phi[0])):
            print(
                "Error : couldn't find the third motor name."
                + "\nPlease add this motor name in the class argument motor3_name"
            )
        if motor3_name is None:
            raise ValueError(
                "SXDM_3D_Scan.find_third_motor: could not automatically determine the "
                "3rd (outer-loop) motor from 'eta'/'phi'. Pass it explicitly, e.g. "
                "SXDM_3D_Scan(filename, motor3_name='my_motor')."
            )
        return motor3_name

    def get_sxdm_3d_motors_position(self):
        for n in range(self.nb_scan):
            scan_nb = n + 1
            sxdm_scan = SXDM_Scan(self.h5file, scan_nb, verbose=False)
            sxdm_scan.get_sxdm_motors_position()

            if n == 0:
                motor1 = np.zeros((self.nb_scan,) + sxdm_scan.motor1.shape)
                motor2 = np.zeros((self.nb_scan,) + sxdm_scan.motor2.shape)
                motor3 = np.zeros(self.nb_scan)

            motor1[n] += sxdm_scan.motor1
            motor2[n] += sxdm_scan.motor2
            motor3[n] += sxdm_scan.get_motor_position(self.motor3_name)

        motor1_name = sxdm_scan.motor1_name
        motor2_name = sxdm_scan.motor2_name

        if self.verbose:
            print("motor1 : {},   shape : {}".format(motor1_name, motor1.shape))
            print("motor2 : {},   shape : {}".format(motor2_name, motor2.shape))
            print("motor3 : {},   shape : {}".format(self.motor3_name, motor3.shape))

        self.motor1 = motor1
        self.motor2 = motor2
        self.motor3 = motor3

        self.motor1_name = motor1_name
        self.motor2_name = motor2_name

        return motor1, motor2, motor3, motor1_name, motor2_name

    def print_counters_list(self):
        self.scan1.print_counters_list()

    def print_sum_counters_list(self):
        self.scan1.print_sum_counters_list()

    def get_roi_data(self, roi_name, plot=False, pcolormesh_plot=False):
        roidata = np.zeros(self.motor1.shape)
        for n in range(self.nb_scan):
            scan_nb = n + 1
            sxdm_scan = SXDM_Scan(self.h5file, scan_nb, verbose=False)
            roidata[n] += sxdm_scan.get_roi_data(roi_name)

        if plot:
            self.plot_roi_sxdm_3d(roidata, pcolormesh_plot=pcolormesh_plot)

        return roidata

    def plot_roi_sxdm_3d(self, roidata, pcolormesh_plot=False):
        nb_rows = int(np.ceil(self.nb_scan / 4))
        fig, ax = plt.subplots(nb_rows, 4, figsize=(14, nb_rows * 4))

        for n, axe in enumerate(ax.flatten()):
            if n < self.nb_scan:
                if pcolormesh_plot:
                    axe.pcolormesh(self.motor1[n], self.motor2[n], roidata[n])
                else:
                    axe.imshow(
                        roidata[n],
                        extent=[
                            np.min(self.motor1[n]),
                            np.max(self.motor1[n]),
                            np.min(self.motor2[n]),
                            np.max(self.motor2[n]),
                        ],
                        origin="lower",
                    )
                axe.set_xlabel(self.motor1_name, fontsize=15)
                axe.set_ylabel(self.motor2_name, fontsize=15)
                axe.set_title("{} : {}".format(self.motor3_name, self.motor3[n]))
            else:
                fig.delaxes(axe)
        fig.tight_layout()

    def get_detector_sum(self, plot=False):
        for n in range(self.nb_scan):
            print(self.nb_scan - n, end=" ")
            scan_nb = n + 1
            sxdm_scan = Scan(self.h5file, scan_nb, verbose=False)
            sxdm_scan.get_detector_sum()
            if n == 0:
                detector_sum = sxdm_scan.detector_sum
            else:
                detector_sum += sxdm_scan.detector_sum
        print("")

        if plot:
            plt.matshow(np.log(detector_sum))
            plt.title("log of sum of the detector images", fontsize=15)

        self.detector_sum = detector_sum
        return detector_sum

    def get_data(self, roi=None, plot=False):
        for n in range(self.nb_scan):
            print(self.nb_scan - n, end=" ")
            scan_nb = n + 1
            sxdm_scan = SXDM_Scan(self.h5file, scan_nb, verbose=False)
            data_one_scan = sxdm_scan.get_data(roi=roi)
            if n == 0:
                data = np.zeros((self.nb_scan,) + data_one_scan.shape)
            data[n] += data_one_scan
        print("")

        if self.verbose:
            print("data.shape", data.shape)
            print(
                "shape dimensions : ( {}, {}, {}, {}, {} )".format(
                    self.motor3_name,
                    self.motor2_name,
                    self.motor1_name,
                    "detector vertical axis",
                    "detector horizontal axis",
                )
            )

        #         self.data = data # I wouldn't do that if it makes a copy of data.
        return data


##################################################################################################################################
#######################################             ct scan            #########################################################
##################################################################################################################################


class Scan_ct(Scan):
    def __init__(self, filename, scan_nb, verbose=False):
        super().__init__(filename, scan_nb, verbose=verbose)

    def get_detector_sum(self, plot=False):

        with h5.File(self.h5file, "r") as h5f:
            detector_sum = h5f[self.scan_string][
                "measurement/{}".format(self.detector)
            ][()]

        self.detector_sum = detector_sum

        if plot:
            plt.matshow(np.log(detector_sum))
            plt.title(
                "log of sum of the detector images\nthis is actually just one image, not a sum",
                fontsize=15,
            )

        return detector_sum


##################################################################################################################################
#######################################             Utilities            #########################################################
##################################################################################################################################


def open_scan(filename, scan_nb, verbose=False):
    command = get_command(scan_nb, filename)

    if "scan" in command and "lookupscan" not in command and "loopscan" not in command:
        return StandardScan(filename, scan_nb, verbose=verbose)

    if "lookupscan" in command:
        return LookupScan(filename, scan_nb, verbose=verbose)

    if "mesh" in command:
        return DmeshScan(filename, scan_nb, verbose=verbose)

    if "sxdm" in command or "kmap" in command:
        return SXDM_Scan(filename, scan_nb, verbose=verbose)

    if "ct" in command:
        return Scan_ct(filename, scan_nb, verbose=verbose)

    if "loopscan" in command:
        return Scan(filename, scan_nb, verbose=verbose)

    raise ValueError(
        f"open_scan: could not recognize the scan type from command {command!r} "
        f"(scan {scan_nb} in {filename}). Add a matching rule to open_scan() in "
        "id01_h5.py if this is a new/legitimate scan type."
    )


def plot_2d_map_sxdm_dmesh(scan, roidata, roi_name):
    fig, ax = plt.subplots(1, 2, figsize=(12, 7))
    ax[0].pcolormesh(scan.motor1, scan.motor2, roidata)
    ax[0].set_title("pcolormesh", fontsize=20)
    ax[1].imshow(
        roidata,
        origin="lower",
        extent=[
            np.min(scan.motor1),
            np.max(scan.motor1),
            np.min(scan.motor2),
            np.max(scan.motor2),
        ],
        aspect="auto",
    )
    ax[1].set_title("imshow", fontsize=20)
    for axe in ax:
        axe.set_xlabel(scan.motor1_name, fontsize=15)
        axe.set_ylabel(scan.motor2_name, fontsize=15)

    fig.suptitle(roi_name, fontsize=20)
    fig.tight_layout()
