from typing import Tuple
import re

# Mapeo de clave musical a notación Camelot (12B/12A system)
# (Escalas mayores = B, menores = A)
KEY_TO_CAMELOT = {
    "C": "8B",  "Cm": "5A",
    "C#": "3B", "C#m": "12A",
    "D": "10B", "Dm": "7A",
    "D#": "5B", "D#m": "2A",
    "E": "12B","Em": "9A",
    "F": "7B", "Fm": "4A",
    "F#": "2B","F#m": "11A",
    "G": "9B", "Gm": "6A",
    "G#": "4B","G#m": "1A",
    "A": "11B","Am": "8A",
    "A#": "6B","A#m": "3A",
    "B": "1B", "Bm": "10A"
}

def to_camelot(key_str: str) -> str:
    """Convierte una key tipo 'C_major' o 'A_minor' a Camelot (8B, 8A...)."""
    key_str = key_str.replace("_major", "").replace("_minor", "m")
    key_str = key_str.replace("_", "")
    return KEY_TO_CAMELOT.get(key_str, "Unknown")

def camelot_distance(key_a: str, key_b: str) -> float:
    """
    Devuelve una penalización basada en la rueda Camelot:
    0 → misma clave
    0.25 → adyacente (8A ↔ 9A o 7A)
    0.5 → opuesta (8A ↔ 8B)
    1.0 → lejana
    """
    if key_a == "Unknown" or key_b == "Unknown":
        return 0.5  # neutral

    if key_a == key_b:
        return 0.0

    num_a, mode_a = int(re.findall(r"\d+", key_a)[0]), key_a[-1]
    num_b, mode_b = int(re.findall(r"\d+", key_b)[0]), key_b[-1]

    # Diferencia circular
    diff = abs(num_a - num_b)
    if diff == 11:
        diff = 1  # rueda circular (12 ↔ 1)

    # Adyacentes en misma escala (A o B)
    if mode_a == mode_b and diff == 1:
        return 0.1

    # Misma posición pero distinto modo (8A ↔ 8B)
    if num_a == num_b and mode_a != mode_b:
        return 0.2

    return 1.0

def bpm_penalty(bpm_a: float, bpm_b: float) -> float:
    """Devuelve una penalización por diferencia de BPM."""
    diff = abs(bpm_a - bpm_b)
    if diff <= 2:
        return 0.0
    if diff <= 6:
        return 0.25
    if diff <= 10:
        return 0.5
    return 1.0

def compute_blending_score(distance: float, camelot_penalty: float, bpm_penalty_value: float) -> float:
    """
    Combina todas las penalizaciones en un solo score (1.0 = mezcla perfecta)
    Penalizaciones reducen el score en proporción.
    """
    base = 1 / (1 + distance)
    penalty = (camelot_penalty * 0.5 + bpm_penalty_value * 0.5)
    return max(0.0, base * (1 - penalty))