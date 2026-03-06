#!/usr/bin/env python3
"""
cache_elevation_grids.py
────────────────────────
For each study tile, downloads Copernicus GLO-30 DEM elevation data via
vsicurl (Cloud-Optimised GeoTIFF on AWS S3) and saves a float32 numpy
elevation grid (_elev.npy) to the tile cache at the same pixel resolution
as the yew probability grid (~10 m/px).

Output: {TILE_CACHE}/{slug}_elev.npy  — shape (H, W), float32, metres ASL
"""

import sys, warnings, signal
from pathlib import Path
from math import cos, radians, ceil, floor

import numpy as np
from scipy.ndimage import zoom as nd_zoom
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.env import Env as RioEnv
from rasterio import transform as rtransform

warnings.filterwarnings("ignore")

TILE_TIMEOUT = 60   # seconds per tile before giving up

class TileTimeout(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TileTimeout("DEM fetch timed out")

ROOT       = Path(__file__).resolve().parents[2]
TILE_CACHE = ROOT / "results" / "analysis" / "cwh_spot_comparisons" / "tile_cache"

# ── Same study-area list as the other scripts ───────────────────────────────
STUDY_AREAS = [
    (48.440, -124.160, "Carmanah-Walbran"),
    (48.550, -124.420, "Port Renfrew"),
    (48.700, -123.700, "Sooke Hills"),
    (48.820, -124.050, "Cowichan Uplands"),
    (49.020, -124.200, "Nanaimo Lakes"),
    (49.620, -125.100, "Comox Uplands"),
    (49.700, -123.150, "Squamish Highlands"),
    (49.800, -125.400, "Strathcona Highlands"),
    (49.900, -126.000, "Gold River Forest"),
    (50.000, -125.250, "Kelsey Bay Forest"),
    (50.050, -125.700, "Muchalat Valley"),
    (50.150, -125.550, "Tahsis Narrows"),
    (50.200, -125.850, "Nootka Sound"),
    (50.200, -124.750, "Campbell River Uplands"),
    (50.250, -125.200, "Sayward Forest"),
    (50.350, -124.400, "Loughborough Inlet"),
    (50.400, -127.100, "Quatsino Sound"),
    (50.450, -127.700, "Holberg Inlet"),
    (50.500, -127.700, "Cape Scott Lowlands"),
    (50.550, -126.600, "Port Hardy Forest"),
    (50.600, -126.200, "Broughton Archipelago"),
    (50.700, -126.500, "Tribune Channel"),
    (50.700, -127.050, "Mahatta River"),
    (50.800, -126.300, "Kingcome Inlet"),
    (51.200, -127.050, "Smith Sound"),
    (49.380, -121.900, "Harrison Lowlands"),
    (49.300, -122.350, "Stave Lake"),
    (49.450, -122.800, "Garibaldi Foothills"),
    (49.400, -123.900, "Sechelt Peninsula"),
    (49.500, -124.500, "Powell River Forest"),
    (49.500, -126.000, "Barkley Sound Slopes"),
    (49.500, -124.900, "Ucluelet Peninsula"),
    (49.600, -123.700, "Howe Sound East"),
    (49.700, -123.550, "Sunshine Coast South"),
    (49.600, -124.200, "Theodosia Inlet"),
    (49.600, -122.400, "Coquitlam Watershed"),
    (49.130, -122.000, "Chilliwack Uplands"),
    (49.250, -121.600, "Hope Slopes"),
    (49.350, -121.200, "Lillooet R Corridor"),
    (50.100, -122.650, "Whistler Callaghan"),
    (49.700, -122.200, "Stave Lake East"),
    (50.450, -121.750, "Lillooet Lake Slopes"),
    (49.050, -122.750, "Langley Uplands"),
    (49.900, -124.100, "Jervis Inlet Slopes"),
    (50.650, -124.850, "Desolation Sound"),
    (50.950, -127.900, "Blunden Harbour"),
    (51.200, -128.650, "Clayoquot Sound"),
    (51.200, -127.350, "Rivers Inlet"),
    (51.350, -127.750, "Owikeno Lake"),
    (51.400, -127.000, "Burke Channel"),
    (51.550, -127.700, "Dean River Lower"),
    (51.550, -126.600, "Milbanke Sound"),
    (51.600, -126.100, "Ocean Falls"),
    (51.700, -128.500, "Calvert Island"),
    (51.750, -127.750, "Namu Lowlands"),
    (51.950, -128.250, "Bella Bella Forest"),
    (52.000, -127.500, "Laredo Sound East"),
    (52.050, -127.950, "Klemtu Forest"),
    (52.100, -128.350, "Princess Royal Island"),
    (52.150, -128.100, "Bella Bella Forest"),
    (52.150, -126.200, "Tweedsmuir South"),
    (52.350, -127.300, "Roscoe Inlet"),
    (52.500, -127.500, "Khutze Inlet"),
    (52.600, -128.500, "Mucha Inlet"),
    (52.700, -128.200, "Dean Channel"),
    (52.800, -128.150, "Gardner Canal Slopes"),
    (53.000, -128.400, "Kitimat Ranges"),
    (53.100, -129.500, "Porcher Island"),
    (53.200, -129.250, "Banks Island NE"),
    (53.350, -129.900, "Laredo Channel"),
    (53.500, -128.800, "Observatory Inlet"),
    (54.150, -129.700, "Prince Rupert Hills"),
    (54.350, -130.150, "Chatham Sound Slopes"),
    (54.000, -129.000, "Skeena Estuary"),
    (54.200, -130.000, "Work Channel"),
    (54.000, -132.200, "Haida Gwaii South"),
    (53.800, -132.000, "Skidegate Flats"),
    (53.900, -132.500, "Haida Gwaii Central"),
    (53.700, -132.200, "Haida Gwaii E Graham"),
    (53.500, -131.500, "Tow Hill Area"),
    (52.000, -128.750, "Seaforth Channel"),
    (52.350, -127.800, "Alberni Valley"),
    (51.450, -125.600, "Knight Inlet"),
    (51.600, -125.800, "Bute Inlet Slopes"),
    (50.900, -126.850, "Tsimpsean Peninsula"),
]


# Copernicus GLO-30 on AWS S3 (anonymous access via https)
COG_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"

def copernicus_tile_url(lat_floor, lon_floor):
    """Build URL for a 1°×1° Copernicus DEM COG tile."""
    ns = "N" if lat_floor >= 0 else "S"
    ew = "W" if lon_floor < 0 else "E"
    lat_abs = abs(lat_floor)
    lon_abs = abs(lon_floor)
    name = f"Copernicus_DSM_COG_10_{ns}{lat_abs:02d}_00_{ew}{lon_abs:03d}_00_DEM"
    url  = f"/vsicurl/{COG_BASE}/{name}/{name}.tif"
    return url


def centre_to_bbox(lat, lon, km=10):
    half_lat = (km * 1000 / 2) / 111320.0
    half_lon = (km * 1000 / 2) / (111320.0 * cos(radians(lat)))
    return (lat - half_lat, lat + half_lat,
            lon - half_lon, lon + half_lon)


def slugify(name):
    return name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def fetch_elevation(south, north, west, east, target_h, target_w):
    """
    Fetch Copernicus DEM elevation for bbox (degrees WGS84).
    Returns float32 array shape (target_h, target_w) in metres.
    Tiles that don't exist (ocean, no-data) are filled with 0.
    """
    lat_tiles = range(floor(south), ceil(north))
    lon_tiles = range(floor(west),  ceil(east))

    srcs = []
    opened = []
    with RioEnv(GDAL_HTTP_TIMEOUT="15", GDAL_HTTP_MAX_RETRY="2",
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE="NO"):
        for la in lat_tiles:
            for lo in lon_tiles:
                url = copernicus_tile_url(la, lo)
                try:
                    src = rasterio.open(url)
                    opened.append(src)
                    srcs.append(src)
                except Exception:
                    pass   # tile does not exist (ocean etc.) — ignore

        if not srcs:
            return np.zeros((target_h, target_w), dtype=np.float32)

        merged, merged_transform = rio_merge(srcs, nodata=-9999)
        elev = merged[0].astype(np.float32)
        elev[elev == -9999] = 0.0

        # Crop to the exact bbox
        rows, cols = rasterio.transform.rowcol(
            merged_transform,
            [west, east],
            [north, south],
        )
        row0, row1 = min(rows), max(rows)
        col0, col1 = min(cols), max(cols)
        row1 = min(row1, elev.shape[0])
        col1 = min(col1, elev.shape[1])
        elev = elev[row0:row1, col0:col1]

        for src in opened:
            src.close()

    if elev.size == 0:
        return np.zeros((target_h, target_w), dtype=np.float32)

    # Resample to target grid via zoom
    zoom_r = target_h / elev.shape[0]
    zoom_c = target_w / elev.shape[1]
    elev_resampled = nd_zoom(elev, (zoom_r, zoom_c), order=1).astype(np.float32)

    # Pad or trim to exact target shape
    out = np.zeros((target_h, target_w), dtype=np.float32)
    h = min(elev_resampled.shape[0], target_h)
    w = min(elev_resampled.shape[1], target_w)
    out[:h, :w] = elev_resampled[:h, :w]
    return out


def main():
    print("Cache Elevation Grids (Copernicus GLO-30 DEM)")
    print("=" * 55)

    done = skipped = failed = 0
    for entry in STUDY_AREAS:
        lat, lon, name = entry[:3]
        slug = slugify(name)
        out_path = TILE_CACHE / f"{slug}_elev.npy"

        # Check reference grid exists
        grid_path = TILE_CACHE / f"{slug}_grid.npy"
        if not grid_path.exists():
            print(f"  SKIP {name}: no grid cached")
            skipped += 1
            continue

        if out_path.exists():
            print(f"  already cached: {name}")
            done += 1
            continue

        ref = np.load(str(grid_path))
        H, W = ref.shape
        south, north, west, east = centre_to_bbox(lat, lon)

        print(f"  [{done+failed+1:2d}] {name:35s} ({H}×{W}) ... ", end="", flush=True)
        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(TILE_TIMEOUT)
            elev = fetch_elevation(south, north, west, east, H, W)
            signal.alarm(0)  # cancel alarm
            np.save(str(out_path), elev)
            print(f"ok  elev range {elev.min():.0f}–{elev.max():.0f} m")
            done += 1
        except (TileTimeout, Exception) as exc:
            signal.alarm(0)
            print(f"FAILED: {exc}")
            failed += 1

    print()
    print(f"Done: {done}  Skipped: {skipped}  Failed: {failed}")


if __name__ == "__main__":
    main()
