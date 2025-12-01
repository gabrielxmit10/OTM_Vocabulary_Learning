import os
import pickle
from pathlib import Path

LEMMA_DICT = {}
_LOADED = False

def load_morphobr(base_path):
    global _LOADED, LEMMA_DICT
    if _LOADED:
        return LEMMA_DICT

    print("Loading MorphoBr...")

    for root, dirs, files in os.walk(base_path):
        for filename in files:
            if not filename.endswith(".dict"):
                continue

            filepath = os.path.join(root, filename)
            
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    surface, analysis = line.split("\t")
                    parts = analysis.split("+")

                    lemma = parts[0]
                    pos = parts[1]

                    surface_lower = surface.lower()

                    if surface_lower not in LEMMA_DICT:
                        LEMMA_DICT[surface_lower] = {
                            "lemma": lemma,
                            "type": pos
                        }

    _LOADED = True
    print("MorphoBr loaded.")
    return LEMMA_DICT

def save_cache(cache_path):
    """Save LEMMA_DICT to pickle file"""
    with open(cache_path, "wb") as f:
        pickle.dump(LEMMA_DICT, f)
    print(f"Cache saved to {cache_path}")

def load_cache(cache_path):
    """Load LEMMA_DICT from pickle file"""
    global LEMMA_DICT, _LOADED
    with open(cache_path, "rb") as f:
        LEMMA_DICT = pickle.load(f)
    _LOADED = True
    print(f"Cache loaded from {cache_path}")

# Auto-load: try cache first, fallback to full load
_morphobr_path = Path(__file__).parent / "MorphoBr-master"
_cache_path = Path(__file__).parent / "morphobr_cache.pkl"

if _cache_path.exists():
    load_cache(_cache_path)
elif _morphobr_path.exists():
    load_morphobr(_morphobr_path)
    save_cache(_cache_path)  # Save for next time
else:
    print(f"Warning: MorphoBr not found at {_morphobr_path}")