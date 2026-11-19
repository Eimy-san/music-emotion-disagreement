from __future__ import annotations

from pathlib import Path
from typing import Dict

import librosa
import numpy as np
import pandas as pd

from .config import CUE_CONFLICT_RULES


def extract_basic_features(audio_path: Path, sr: int = 22050) -> Dict[str, float]:
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    flux = np.mean(np.abs(np.diff(librosa.stft(y), axis=1))) if len(y) else np.nan
    dynamic_range = float(np.percentile(rms, 95) - np.percentile(rms, 5))

    return {
        "tempo": float(tempo),
        "onset_strength_mean": float(np.mean(onset_env)),
        "rms_mean": float(np.mean(rms)),
        "dynamic_range": dynamic_range,
        "spectral_centroid_mean": float(np.mean(spectral_centroid)),
        "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
        "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
        "spectral_flux_proxy": float(flux),
        "zero_crossing_rate_mean": float(np.mean(zcr)),
        "chroma_mean": float(np.mean(chroma)),
        "tonnetz_mean": float(np.mean(tonnetz)),
    }


def add_simple_feature_flags(df: pd.DataFrame) -> pd.DataFrame:
    flagged = df.copy()
    numeric_cols = [col for col in flagged.columns if pd.api.types.is_numeric_dtype(flagged[col])]
    for col in numeric_cols:
        median = flagged[col].median()
        flagged[f"high_{col}"] = flagged[col] > median
        flagged[f"low_{col}"] = flagged[col] <= median

    if "tempo" in flagged.columns:
        flagged["high_tempo"] = flagged["tempo"] > flagged["tempo"].median()
        flagged["low_tempo"] = flagged["tempo"] <= flagged["tempo"].median()
    if "spectral_centroid_mean" in flagged.columns:
        flagged["high_spectral_centroid"] = flagged["spectral_centroid_mean"] > flagged["spectral_centroid_mean"].median()
    if "spectral_flux_proxy" in flagged.columns:
        flagged["high_spectral_flux"] = flagged["spectral_flux_proxy"] > flagged["spectral_flux_proxy"].median()

    tonnetz_series = flagged["tonnetz_mean"] if "tonnetz_mean" in flagged.columns else pd.Series(0.0, index=flagged.index)
    dynamic_series = flagged["dynamic_range"] if "dynamic_range" in flagged.columns else pd.Series(0.0, index=flagged.index)
    flagged["minor_mode_proxy"] = tonnetz_series < tonnetz_series.median()
    flagged["stable_harmony_proxy"] = dynamic_series <= dynamic_series.median()
    return flagged


def add_cue_conflict_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = add_simple_feature_flags(df)
    for name, rule in CUE_CONFLICT_RULES.items():
        cols = rule["all_of"]
        missing = [col for col in cols if col not in out.columns]
        if missing:
            out[name] = False
            continue
        out[name] = out[cols].all(axis=1)
    return out

