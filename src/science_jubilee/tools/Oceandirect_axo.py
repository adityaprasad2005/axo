import json
import logging
import os
import datetime as dt
import csv, pathlib
import re 

from typing import Dict

from typing import Tuple, Union
import sys
import pandas as pd
import numpy as np

sdk_path = r"C:\Program Files\Ocean Optics\OceanDirect SDK\Python"
sys.path.insert(0, sdk_path)
# this is the Ocean Optics SDK, which is (very unfortunately) not open-source
try:
    from oceandirect.OceanDirectAPI import OceanDirectAPI, OceanDirectError
except ImportError:
    raise ImportError(
        "The Ocean Optics SDK is not installed. Please install it from the Ocean Insight website."
    )

from science_jubilee.labware.Labware import Labware, Location, Well
from science_jubilee.tools.Tool import Tool, requires_active_tool

import numpy as np
import matplotlib.pyplot as plt


def _yaml_header(meta: Dict[str, object]) -> str:
    """Return a YAML‑style header string (every line starts with `# `) wrapped in
    `# ---` barriers so it’s visually distinct and YAML parsers can pick it
    up with a trivial pre‑process that strips the leading "# ".
    """
    lines = ["# ---"]
    for k, v in meta.items():
        lines.append(f"# {k}: {v}")
    lines.append("# ---")
    return "\n".join(lines) + "\n"

def _parse_header(path: pathlib.Path) -> Tuple[Dict[str, str], int]:
    """Return (header_dict, header_line_count)."""
    header: Dict[str, str] = {}
    line_count = 0
    with path.open() as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            line_count += 1
            if line.strip() in ("# ---", "#---"):
                continue
            if ":" in line:
                key, val = line[2:].split(":", 1)  # drop "# " prefix
                header[key.strip()] = val.strip()
    return header, line_count

class Spectrometer(Tool, OceanDirectAPI):
    
    DEFAULT_DIR: pathlib.Path = pathlib.Path("Spectra")
    
    def __init__(self, 
                 index, 
                 name, 
                 base_dir:  str | pathlib.Path, 
                 plate_id: str | None = None,
                 ref_dark: str = "dark.npy",
                 ref_white: str = "white.npy"):
        super().__init__(index, name)
        
        # ---------------------General Spectro Setup ------------------------#
        self.name = name
        self.index = index
        self.ocean = OceanDirectAPI()
        self.spectrometer, self.device_id = self.open_spectrometer()
        
        # ---------------------Storage Hierarchy --------------------#
        self.base_dir : pathlib.Path = (
            pathlib.Path(base_dir).expanduser().resolve()
            if base_dir is not None
            else self.DEFAULT_DIR.resolve()
        )
        
        self.base_dir.mkdir(parents = True, exist_ok= True)
        
        # Reference_Spectrum Folder
        self.ref_dir: pathlib.Path = self.base_dir / "refs"
        self.ref_dir.mkdir(exist_ok = True)
        
        # ----------------- plate / path handling ---------------------#
        self.plate_id : str = plate_id or dt.datetime.now().strftime("%Y%m%d")
        self.plate_dir : pathlib.Path = self.base_dir / self.plate_id          # Spectra/20250527
        self.plate_dir.mkdir(parents=True, exist_ok=True)

        # -----------------Reference Filenames (White&Dark-------------------------#
        self._dark_path = self.ref_dir / ref_dark
        self._white_path = self.ref_dir / ref_white
        
        # Cached Dark/White Spectra
        self.dark : np.ndarray | None = None
        self.white : np.ndarray | None = None
        self.dark_id: str | None = None
        self.white_id: str | None = None
        
        # Placeholder for last move
        self.current_well      = None
        self.current_location  = None
        
        # Try loading previous refs
        self._load_references()
        
        logging.info("Opened Spectrometer %s", self.device_id)
    
    # ------------------------- Device Management -------------------------------------------------------------------
    def find_spectrometers(self):
        """Probe and return list of device IDs."""

        count = self.ocean.find_devices()
        if count == 0:
            raise RuntimeError("No Ocean Insight Spectrometers Detected")
        
        return self.ocean.get_device_ids() 
    
    def open_spectrometer(self):
        device_ids = self.find_spectrometers()
        self.device_id = device_ids[0]
        self.spectrometer = self.ocean.open_device(self.device_id)
        
        print(f"Opened Spectrometer {self.device_id}")
        return self.spectrometer, self.device_id
        
    def close_spectrometer(self):
        self.ocean.close_device(self.device_id)
        print(f"Closed Spectrometer {self.device_id}")
        
    # ---------------------Spectrometer Configuration-------------------------------------------------------
    
    def configure_device(self, 
                       integration_time_us : int = 10000, 
                       scans_to_avg : int = 50,
                       boxcar_width : int = 50):
        """
        integration_time_us :
        scans_to_avg :
        boxcar_width : 
        """
        self.spectrometer.set_integration_time(integration_time_us)
        self.spectrometer.set_scans_to_average(scans_to_avg)
        self.spectrometer.set_boxcar_width(boxcar_width)
    
    
    def lamp_shutter(self, open: bool = False):
        """Open/close internal lamp shutter if the device supports it."""
        
        state = "Close" if open else "Open"
        try:
            self.spectrometer.Advanced.set_enable_lamp(open)
            if self.spectrometer.Advanced.get_enable_lamp() == open:
                print(f"Light shutter set to {state}")
                
        except AttributeError:
            print("This spectrometer has no controllable lamp shutter")
        
    
    # -------------------------- RAW ACQUISITION -----------------------#
    def measure_raw_spectrum(self):
        
        wl = np.array(self.spectrometer.get_wavelengths())
        vals = np.array(self.spectrometer.get_formatted_spectrum())
        
        # Always returns the same wavelength-axis UV-Vis spectrum
        return wl, vals
    
    @staticmethod
    def compute_absorbance(sample_vals: np.ndarray, 
                           dark_vals : np.ndarray, 
                           white_vals : np.ndarray,
                           eps : float = 1e-9,
                           clip : bool = True):
        """
        Compute absorbance spectrum from raw intensities:
        A(λ) = -log10( (I_sample - I_dark) / (I_white - I_dark) )
        eps is added to denominator to avoid divide-by-zero.
        """
        # dark-correct
        sample_dc = sample_vals - dark_vals
        white_dc  = white_vals  - dark_vals
        ratio     = sample_dc / (white_dc + eps)
        if clip:
            ratio = np.clip(ratio, 1e-9, 1.0)
         
        return -np.log10(ratio)
    
    
    #---------------------------------------- CSV Helper ----------------------------------------#
    
    def _csv_path(self, well_id : str | np.str_) -> pathlib.Path:
        """Spectra/<well_id>.csv"""
        return self.plate_dir / f"{str(well_id)}.csv" # spectra/…/A1.csv
    
    def _ensure_file(self, path: pathlib.Path, meta: Dict[str, object]):
        """Create file with header if it does not exist."""
        if path.exists():
            return
        # add plate_id to header
        meta = {"plate_id": self.plate_id, **meta}
        path.write_text(_yaml_header(meta) + "time_min,wavelength_nm,absorbance\n")
    
    @staticmethod
    def read_spectrum_csv(path: str | pathlib.Path) -> Tuple[Dict[str, str], pd.DataFrame]:
        """Return metadata dict *and* absorbance DataFrame."""
        p = pathlib.Path(path)
        meta, _ = _parse_header(p)
        df = pd.read_csv(p, comment="#", index_col="wavelength_nm")
        return meta, df
   
    
    
    # -------------------------CSV append ------------------------------------#
    
    def _append_spectrum_csv(self,
        well_id: str | np.str_,
        time_min: int,
        wl: np.ndarray,
        absorbance: np.ndarray,
        meta: Dict[str, object],
    ) -> None:
        
        path = self._csv_path(well_id)
        col_name = f"{time_min} min"

        # 1) read existing (header + DF)
        if path.exists():
            header_dict, header_lines = _parse_header(path)
            df = pd.read_csv(path, comment="#", index_col="wavelength_nm")
        else:
            header_dict = {"plate_id": self.plate_id, **meta}
            header_lines = 0
            df = pd.DataFrame(index=np.round(wl, 1))

        # 2) update column & header (header only on *first* creation)
        df[col_name] = absorbance

        if not path.exists():
            header_text = _yaml_header(header_dict)
            with path.open("w", newline="") as fh:
                fh.write(header_text)
                df.to_csv(fh, index_label="wavelength_nm")
        else:
            # overwrite file but keep original header block verbatim
            with path.open("r+") as fh:
                lines = fh.readlines()
            body = "".join(lines[header_lines:])  # skip old body
            with path.open("w", newline="") as fh:
                fh.writelines(lines[:header_lines])  # unchanged header
                df.to_csv(fh, index_label="wavelength_nm")
    # ------------------------- Move Spectrometer ---------------------------------- #
    # ------------------------- Data Collection ------------------------------------ #
    @requires_active_tool
    def position_probe(self,
                       location: Union[Well, Tuple, Location]) -> None:
        """
        Move the spectrometer probe above `location` and update `self.current_well`.

        Notes
        -----
        Safe-Z retract, XY move, then Z plunge (same pattern you used before).  
        Stores both `self.current_well` and `self.current_location` so that
        subsequent data-only calls can use that context.
        """
        x, y, z = Labware._getxyz(location)

        self._machine.safe_z_movement()
        self._machine.move_to(x=x, y=y, wait = True)
        self._machine.move_to(z=z, wait = True)
        # Replace 50 to well plate or reservoir height
        
        # ---------- robust well bookkeeping ----------
        if isinstance(location, Well):
            self.current_location = location
            self.current_well = location.name
        elif isinstance(location, Location):
            # Opentrons-style Location → Well
            self.current_location = location._labware
            self.current_well = self.current_location.name
        else:
            self.current_well = location                     # e.g. raw (x, y, z) tuple


    @requires_active_tool
    def wash_probe(self, wash_loc : Union[Well, Tuple, Location], n_cycles : int = 1):
        """
        Wash the probe with the supplied location.
        """
        for i in range(n_cycles):
            self.position_probe(wash_loc)


    @requires_active_tool
    def collect_spectrum(self,
                         location: Union[Well, Tuple, Location],
                         elapsed_min : int,
                         open : bool | None = None,
                         save: bool = False): 
        

        self.position_probe(location)
        
        if self.dark is None  or self.white is None:
            raise RuntimeError("Dark/white spectra not set")
        
        # Lamp Shutter control, if desired
        if open is not None: # only act when caller says True/False
            self.lamp_shutter(open = open) 
        
        # Acquire
        wl, vals = self.measure_raw_spectrum()
        
        absorbance = self.compute_absorbance(vals, self.dark, self.white)
        
        # --------save ------------
        if save:
            loc = self.current_location
            well_id = str(self.current_well) 
            meta = dict(
                well_id = well_id,
                slot= loc,
                pixels = len(wl),
                integration_time_us = self.spectrometer.get_integration_time(),
                dark_id = self.dark_id,
                white_id = self.white_id,
                wavelength_unit = "nm",
                absorbance_unit = "AU",
            )
            self._append_spectrum_csv(
                well_id, elapsed_min, wl, absorbance, meta
            )
        
    
        return wl, vals, absorbance

    
    #------------------ Reference Spectrum Setup with recall ----------------------#
    def _latest_ref(self, prefix: str) -> pathlib.Path | None:
        """Return Path to the newest '<prefix>_YYYYmmdd_HHMMSS.npy' file."""
        files = sorted(self.plate_dir.glob(f"{prefix}_*.npy"))
        return files[-1] if files else None # If at least one match was found, return the last element (newest). Otherwise return None

    def set_dark(self, n_avg: int = 5):
        """Capture dark spectrum; store with an ID timestamp."""
        # wl, vals = zip(*(self.measure_raw_spectrum() for _ in range(n_avg)))
        # # 5 time measures and take the means of the intensity axis
        # self.dark  = np.mean(vals, axis=0)
        wl, vals = self.measure_raw_spectrum()
        dark_data = np.column_stack((wl, vals))
        self.dark = vals
        self.dark_id  = f"dark_{dt.datetime.now():%Y%m%d_%H%M%S}"
        self._dark_path = self.ref_dir / f"{self.dark_id}.npy"
        # np.save(self._dark_path, self.dark)
        np.save(self._dark_path, dark_data)
        return self.dark_id

    def set_white(self, n_avg: int = 5):
        """Capture white (reference) spectrum; store with an ID timestamp."""
        # 5 time measures and take the means of the intensity axis
        # wl, vals = zip(*(self.measure_raw_spectrum() for _ in range(n_avg)))
        # self.white = np.mean(vals, axis=0)
        wl, vals = self.measure_raw_spectrum()
        white_data = np.column_stack((wl, vals))
        self.white = vals
        self.white_id = f"white_{dt.datetime.now():%Y%m%d_%H%M%S}"
        self._white_path = self.ref_dir / f"{self.white_id}.npy"
        # np.save(self._white_path, self.white)
        np.save(self._white_path, white_data)
        return self.white_id
    

    def _load_references(self):
        """
        Populate self.dark/white and their IDs from disk **if possible**.

        Priority:
        (i) newest   dark_<timestamp>.npy / white_<timestamp>.npy pair  
        (ii) legacy  dark.npy / white.npy       (keeps old projects usable)
        """
        dark_path  = self._latest_ref("dark")
        white_path = self._latest_ref("white")

        # --- newest scheme present ---
        if dark_path and white_path:
            self.dark      = np.load(dark_path)
            self.white     = np.load(white_path)
            self.dark_id   = dark_path.stem        # e.g. 'dark_20250528_154722'
            self.white_id  = white_path.stem
            self._dark_path, self._white_path = dark_path, white_path
            return 
        
        # --- fallback to legacy fixed names ---
        legacy_d = self._dark_path # points to e.g. dark.npy (or whatever ref_dark was)
        legacy_w = self._white_path # points to e.g. white.npy
        
        if legacy_d.exists() and legacy_w.exists():
            self.dark      = np.load(legacy_d)
            self.white     = np.load(legacy_w)
            self.dark_id   = legacy_d.stem
            self.white_id  = legacy_w.stem
            #self._dark_path, self._white_path = legacy_d, legacy_w
    
    
    # ---------------------------------------------------- Storage management -
    def set_storage_root(self, new_root: str | pathlib.Path) -> None:
        """Change where all subsequent spectra are written."""
        self.base_dir = pathlib.Path(new_root).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.ref_dir = self.base_dir / "ref"
        self.ref_dir.mkdir(exist_ok = True)
        
        self.plate_dir = self.base_dir / self.plate_id
        self.plate_dir.mkdir(parents=True, exist_ok=True)
        

    def plot_spectrum(self, 
                  location: Union[Well, Tuple, Location], 
                  elapsed_min: int = 15,
                  save_plot: bool = False,
                  show_plot: bool = True,
                  figsize: Tuple[int, int] = (10, 6)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Plot absorbance spectrum for a given location/well.
        """

        # Determine well_id from location
        if isinstance(location, Well):
            well_id = location.name
        elif isinstance(location, Location):
            well_id = location._labware.name
        else:
            well_id = str(location)

        # Check if CSV file exists
        csv_path = self._csv_path(well_id)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"No spectrum data found for {well_id}. "
                f"Expected file: {csv_path}. "
                f"Run collect_spectrum() first."
            )

        # Read the CSV data
        try:
            meta, df = self.read_spectrum_csv(csv_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read spectrum data from {csv_path}: {e}")

        # Check if the requested time point exists
        time_col = f"{elapsed_min} min"
        if time_col not in df.columns:
            available_times = [col for col in df.columns if col.endswith(' min')]
            raise ValueError(
                f"Time point '{time_col}' not found in data. "
                f"Available time points: {available_times}"
            )

        # Extract wavelengths and absorbance (already computed)
        wavelengths = df.index.values  # wavelength_nm is the index
        absorbance = df[time_col].values

        # Create the plot
        if show_plot or save_plot:
            plt.figure(figsize=figsize)
            plt.plot(wavelengths, absorbance, color='purple', linewidth=2, label='Absorbance')
            plt.xlabel("Wavelength (nm)", fontsize=12)
            plt.ylabel("Absorbance (AU)", fontsize=12)
            plt.title(f"Absorbance Spectrum - {well_id} ({time_col})", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

            # Save plot if requested
            if save_plot:
                plot_path = csv_path.parent / f"{well_id}_{elapsed_min}min_absorbance.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to: {plot_path}")

            if show_plot:
                plt.show()
            else:
                plt.close()

        return wavelengths, absorbance

    
