"""Extract spectral peaks from audio files."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks

from .convert import convert_pitch


@dataclass
class AudioPeaksDetail:
    """Intermediate data returned by :func:`audio_peaks`."""

    freq_spectrum: np.ndarray
    freq_axis: np.ndarray
    audio: np.ndarray
    fs: int
    smooth_spectrum: np.ndarray | None = None
    raw_spectrum: np.ndarray | None = None
    cents_axis: np.ndarray | None = None


def audio_peaks(
    audio_file: str | Path,
    *,
    sigma: float = 0.0,
    resolution: float = 1.0,
    ramp_duration: float = 0.0,
    f_min: float = 27.5,
    f_max: float = float("inf"),
    min_prominence: float = 0.01,
    noise_factor: float = 0.0,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, AudioPeaksDetail]:
    """Extract spectral peaks from an audio file.

    Parameters
    ----------
    audio_file : str or Path
        Path to audio file (any format supported by ``soundfile``).
    sigma : float
        Gaussian smoothing width in cents (0 = no smoothing).
    resolution : float
        Cents-grid spacing when ``sigma > 0``.
    ramp_duration : float
        Onset/offset ramp in seconds (Hann fade).
    f_min, f_max : float
        Frequency range in Hz.
    min_prominence : float
        Peak prominence threshold (fraction of max).
    noise_factor : float
        Noise-floor threshold (multiple of median).
    verbose : bool
        Print progress.

    Returns
    -------
    f : np.ndarray
        Peak frequencies in Hz.
    w : np.ndarray
        Normalised peak amplitudes in [0, 1].
    detail : AudioPeaksDetail
        Intermediate data.

    Examples
    --------
    Basic peak extraction (no smoothing)::

        f, w, detail = mpt.audio_peaks('audio/piano_C4.wav')

    With smoothing for vibrato-rich audio::

        f, w, detail = mpt.audio_peaks('audio/violin_A4.wav', sigma=12)

    Onset/offset ramp for an abruptly cut sample::

        f, w, detail = mpt.audio_peaks('audio/music_sample.wav',
                                        ramp_duration=0.02)

    Spectral pitch similarity of two audio files::

        fA, wA, _ = mpt.audio_peaks('audio/Piano_Emin.wav')
        fB, wB, _ = mpt.audio_peaks('audio/Piano_G7_3rd_inversion.wav')
        pA = mpt.convert_pitch(fA, 'hz', 'cents')
        pB = mpt.convert_pitch(fB, 'hz', 'cents')
        s = mpt.cos_sim_exp_tens_raw(pA, wA, pB, wB, 12, 2, True, True, 1200)

    Spectral entropy (no add_spectra needed)::

        f, w, _ = mpt.audio_peaks('audio/Piano_Cmin_open.wav')
        p = mpt.convert_pitch(f, 'hz', 'cents')
        H = mpt.spectral_entropy(p, w, 12)

    Roughness (Hz input — no conversion needed)::

        f, w, _ = mpt.audio_peaks('audio/Piano_Cmin_open.wav')
        r = mpt.roughness(f, w)
    """
    import soundfile as sf

    audio_file = Path(audio_file)
    if not audio_file.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    x, fs = sf.read(str(audio_file), dtype="float64")
    if x.ndim > 1:
        x = np.mean(x, axis=1)

    N = len(x)
    if verbose:
        print(
            f"audio_peaks: {audio_file.name}  "
            f"({N} samples, {N / fs:.1f} s, Fs = {fs} Hz)"
        )

    # Onset/offset ramp
    if ramp_duration > 0:
        ramp_samples = round(ramp_duration * fs)
        if 0 < ramp_samples <= N // 2:
            ramp = 0.5 * (1 - np.cos(np.pi * np.arange(ramp_samples) / ramp_samples))
            x[:ramp_samples] *= ramp
            x[-ramp_samples:] *= ramp[::-1]
        elif ramp_samples > N // 2:
            warnings.warn("Ramp duration exceeds half the signal length. Not applied.")

    # Single-sided magnitude spectrum
    X = np.fft.fft(x)
    n_fft = N // 2 + 1
    mag_spec = np.abs(X[:n_fft]) / N
    mag_spec[1:-1] *= 2  # account for negative frequencies

    f_axis = np.arange(n_fft) * (fs / N)

    # Clamp f_max to Nyquist
    nyquist = fs / 2
    if np.isinf(f_max) or f_max > nyquist:
        f_max = nyquist
    if f_min >= f_max:
        raise ValueError(f"f_min ({f_min}) must be less than f_max ({f_max}).")

    valid = (f_axis >= f_min) & (f_axis <= f_max)
    f_valid = f_axis[valid]
    mag_valid = mag_spec[valid]

    _empty = np.zeros(0)

    if len(f_valid) < 2:
        warnings.warn("Fewer than 2 FFT bins in the frequency range.")
        return _empty, _empty, AudioPeaksDetail(mag_spec, f_axis, x, fs)

    # Peak picking
    if sigma > 0:
        # Smoothed in log-frequency (cents) space
        c_valid = convert_pitch(f_valid, "hz", "cents")
        step = resolution
        c_min = np.ceil(c_valid.min() / step) * step
        c_max = np.floor(c_valid.max() / step) * step
        c_grid = np.arange(c_min, c_max + step / 2, step)

        if len(c_grid) < 2:
            warnings.warn("Cents range too narrow for requested resolution.")
            return _empty, _empty, AudioPeaksDetail(mag_spec, f_axis, x, fs)

        interp = PchipInterpolator(c_valid, mag_valid, extrapolate=False)
        raw_log_spec = np.nan_to_num(interp(c_grid), nan=0.0)
        raw_log_spec = np.maximum(raw_log_spec, 0)

        # Gaussian kernel
        half_width = int(np.ceil(4 * sigma / step))
        kx = np.arange(-half_width, half_width + 1) * step
        kernel = np.exp(-kx**2 / (2 * sigma**2))
        kernel /= kernel.sum()
        smooth_log_spec = np.convolve(raw_log_spec, kernel, mode="same")

        spec_for_peaks = smooth_log_spec

        if verbose:
            print(
                f"  Smoothed log-f spectrum: {c_min:.1f} to {c_max:.1f} "
                f"MIDI cents, {len(c_grid)} bins, sigma = {sigma} cents"
            )
    else:
        spec_for_peaks = mag_valid
        smooth_log_spec = None
        raw_log_spec = None
        c_grid = None

        if verbose:
            print(
                f"  Frequency range: {f_min:.1f} to {f_max:.1f} Hz, "
                f"{len(f_valid)} bins (no smoothing)"
            )

    if np.max(spec_for_peaks) == 0:
        if verbose:
            print("  No spectral energy in the specified range.")
        return (
            _empty,
            _empty,
            AudioPeaksDetail(mag_spec, f_axis, x, fs, smooth_log_spec, raw_log_spec, c_grid),
        )

    # Find peaks
    peak_prom = min_prominence * np.max(spec_for_peaks)
    pk_locs, pk_props = find_peaks(spec_for_peaks, prominence=peak_prom)
    pk_amps = spec_for_peaks[pk_locs]

    # Noise floor
    if noise_factor > 0:
        noise_floor = noise_factor * np.median(spec_for_peaks)
        keep = pk_amps > noise_floor
        pk_locs = pk_locs[keep]
        pk_amps = pk_amps[keep]

    if len(pk_amps) == 0:
        if verbose:
            print("  No peaks above threshold.")
        return (
            _empty,
            _empty,
            AudioPeaksDetail(mag_spec, f_axis, x, fs, smooth_log_spec, raw_log_spec, c_grid),
        )

    # Convert to Hz
    if sigma > 0:
        pk_cents = c_grid[pk_locs]
        f_out = convert_pitch(pk_cents, "cents", "hz")
    else:
        f_out = f_valid[pk_locs]

    w_out = pk_amps / np.max(pk_amps)

    if verbose:
        print(
            f"  Found {len(f_out)} peaks "
            f"(prominence threshold: {min_prominence * 100:.1f}% of max)."
        )

    detail = AudioPeaksDetail(
        freq_spectrum=mag_spec,
        freq_axis=f_axis,
        audio=x,
        fs=fs,
        smooth_spectrum=smooth_log_spec,
        raw_spectrum=raw_log_spec,
        cents_axis=c_grid,
    )

    return f_out, w_out, detail
