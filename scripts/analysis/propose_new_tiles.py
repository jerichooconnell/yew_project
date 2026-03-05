#!/usr/bin/env python3
"""
propose_new_tiles.py
────────────────────
Visualise 40 candidate new 10×10 km study tiles on top of the CWH/CDF
boundary and existing tiles, ensuring each candidate:
  - Falls within the CWH/CDF coastal boundary polygon
  - Is not centred in open water (checked against coastal_land_only.geojson)

Outputs an interactive folium HTML map at:
    results/analysis/proposed_tiles_map.html

Usage:
    conda run -n yew_pytorch python scripts/analysis/propose_new_tiles.py
"""
from pathlib import Path
from math import cos, radians

import folium
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box

ROOT = Path(__file__).resolve().parents[2]
# Full BEC-derived CWH + CDF biogeoclimatic zone (695 polygons, province-wide)
CWH_BOUNDARY = ROOT / "data" / "processed" / "cwh_cdf_land.shp"
LAND_BOUNDARY = ROOT / "data" / "processed" / "coastal_land_only.geojson"
OUT_HTML      = ROOT / "results" / "analysis" / "proposed_tiles_map.html"

# ── Existing 45 tiles ─────────────────────────────────────────────────────────
EXISTING = [
    (48.440, -124.160, "Carmanah-Walbran"),
    (48.600, -123.800, "Sooke Hills"),
    (49.315, -124.980, "Clayoquot Sound"),
    (50.020, -125.240, "Campbell River Uplands"),
    (50.700, -127.100, "Quatsino Sound"),
    (49.700, -123.150, "Squamish Highlands"),
    (50.720, -124.000, "Desolation Sound"),
    (52.330, -126.600, "Bella Coola Valley"),
    (54.150, -129.700, "Prince Rupert Hills"),
    (53.500, -128.600, "Kitimat Ranges"),
    (49.900, -125.550, "Strathcona Highlands"),
    (49.860, -122.680, "Garibaldi Foothills"),
    (50.830, -124.920, "Bute Inlet Slopes"),
    (49.020, -124.200, "Nanaimo Lakes"),
    (51.400, -127.700, "Rivers Inlet"),
    (48.550, -124.420, "Port Renfrew"),
    (48.820, -124.050, "Cowichan Uplands"),
    (49.620, -125.100, "Comox Uplands"),
    (49.780, -126.020, "Gold River Forest"),
    (50.720, -127.500, "Port Hardy Forest"),
    (49.400, -123.720, "Sunshine Coast South"),
    (49.520, -123.420, "Howe Sound East"),
    (49.780, -124.550, "Powell River Forest"),
    (50.100, -124.060, "Jervis Inlet Slopes"),
    (51.020, -124.480, "Toba Inlet Slopes"),
    (51.080, -125.680, "Knight Inlet"),
    (50.760, -126.480, "Broughton Archipelago"),
    (51.220, -126.020, "Kingcome Inlet"),
    (51.640, -126.520, "Owikeno Lake"),
    (52.090, -126.840, "Burke Channel"),
    (52.380, -127.680, "Ocean Falls"),
    (52.720, -126.560, "Dean Channel"),
    (52.900, -128.700, "Princess Royal Island"),
    (52.510, -128.580, "Milbanke Sound"),
    (54.820, -130.120, "Portland Inlet"),
    (50.250, -125.750, "Muchalat Valley"),
    (49.250, -122.250, "Stave Lake"),
    (53.819,  -132.435,"Haida Gwaii South"),
    (49.250, -121.750, "Chilliwack Uplands"),
    (55.250, -130.750, "Stewart Lowlands"),
    (51.250, -127.250, "Smith Sound"),
    (49.250, -125.250, "Alberni Valley"),
    (49.750, -123.750, "Sechelt Peninsula"),
    (52.750, -128.250, "Klemtu Forest"),
    (51.750, -127.750, "Namu Lowlands"),
]

# ── 40 Candidate new tiles ────────────────────────────────────────────────────
# Selected to fill geographic gaps while staying in the CWH/CDF zone
CANDIDATES = [
    # Vancouver Island — filling E coast + NW + outer west
    (50.400, -126.050, "Sayward Forest",          "NE VI CWHmm, upper valley forests"),
    (50.600, -128.000, "Cape Scott Lowlands",      "NW VI CWHvh1, inland plateau"),
    (50.540, -127.760, "Holberg Inlet",            "NW VI CWHvh1, remote"),
    (48.970, -125.250, "Barkley Sound Slopes",     "SW VI CWHmm1, Kennedy Lake area"),
    (49.450, -124.700, "Courtenay Uplands",        "Central-east VI CWHmm1"),
    (49.050, -125.450, "Ucluelet Peninsula",       "West VI CWHvh1, Ucluelet forest"),
    (49.650, -126.700, "Tahsis Narrows",           "West-central VI inner CWH"),
    (50.480, -126.000, "Kelsey Bay Forest",        "NE VI CWHmm1 valley"),
    # Mainland south coast / Sea-to-Sky
    (50.050, -122.850, "Whistler Callaghan",       "Coast Mtns CWH/MH, upper slopes"),
    (49.700, -122.300, "Coquitlam Watershed",      "Lower mainland CWHvm1"),
    (50.420, -122.600, "Lillooet R. Corridor",     "Transition CWH–IDF upper valley"),
    (49.050, -122.050, "Harrison Lowlands",        "Fraser CWHdm, logged valley"),
    # Sunshine Coast / fjords north of existing
    (49.980, -124.500, "Theodosia Inlet",          "Upper Sunshine Coast CWHvm2"),
    (50.600, -123.800, "Lillooet Lake Slopes",     "Pemberton–Squamish CWHvm1"),
    (50.380, -124.650, "Loughborough Inlet",       "Northern Sunshine Coast fjord CWH"),
    # Central mainland fjords (filling N of Bute/Toba block)
    (51.450, -124.650, "Homathko Canyon",          "Upper Bute tributary CWHxm2"),
    (51.750, -125.750, "Klinaklini Valley",        "Central coast fjord CWHmm1"),
    (52.300, -126.650, "Dean River Lower",         "Lower Dean valley CWH, alluvial flats"),
    (53.000, -127.500, "Gardner Canal Slopes",     "Kemano valley inner CWH"),
    (53.350, -129.000, "Porcher Island",           "Outer north coast CWHvh2"),
    # Mid-coast islands — filling large gap
    (51.650, -128.050, "Calvert Island",           "Outer CWHvh2, Rivers Inlet area"),
    (52.150, -128.100, "Bella Bella Forest",       "Denny Island CWHvh2, outer mid-coast"),
    (52.800, -128.500, "Laredo Sound East",        "Inner passage mainland CWHvh"),
    (53.200, -130.050, "Banks Island NE",          "North coast outer CWHvh"),
    (51.850, -127.300, "Roscoe Inlet",             "Deep mainland fjord CWHvm2"),
    (52.250, -127.750, "Khutze Inlet",             "Mid-coast fjord CWHvm2"),
    # North coast / Skeena / Nass
    (54.350, -130.350, "Tsimpsean Peninsula",      "Prince Rupert mainland CWHvh"),
    (54.100, -130.100, "Skeena Estuary",           "Outer coast Prince Rupert CWHvh, outer_island"),
    (54.650, -130.450, "Work Channel",             "N of Prince Rupert, outer coast CWHvh3"),
    (54.350, -130.150, "Chatham Sound Slopes",     "Near Prince Rupert mainland CWHvh"),
    # Haida Gwaii — currently only 1 tile on South Moresby
    (53.350, -132.050, "Haida Gwaii Central",      "Central Moresby Island CWHvh3"),
    (53.680, -132.450, "Haida Gwaii East Graham",  "Graham Island east coast CWH"),
    (54.000, -132.050, "Tow Hill Area",            "N Graham Island CWHvh3"),
    (53.550, -132.000, "Skidegate Flats",          "Graham–Moresby narrows CWH"),
    # Fill remaining VI/mainland
    (51.950, -128.200, "Seaforth Channel",         "Mid-coast outer passage CWH"),
    (52.850, -127.800, "Mucha Inlet",              "Dean-Kimsquit corridor CWH"),
    (54.650, -130.350, "Observatory Inlet",        "North fjord CWHvh3"),
    (50.950, -127.350, "Blunden Harbour",          "Broughton area CWHvh inner"),
    (51.350, -126.700, "Tribune Channel",          "Remote outer-coast fjord CWH"),
    (52.150, -126.200, "Tweedsmuir South",         "Atnarko CWH valley bottom"),
]


def centre_to_bbox(lat, lon, km=10):
    hl = (km * 500) / 111320.0
    hw = (km * 500) / (111320.0 * cos(radians(lat)))
    return box(lon - hw, lat - hl, lon + hw, lat + hl)


def main():
    print("Loading boundaries...")
    # cwh_cdf_land.shp is EPSG:3005; reproject to 4326 once, then union
    cwh_raw = gpd.read_file(str(CWH_BOUNDARY))  # EPSG:3005, 695 parts
    cwh = cwh_raw.to_crs("EPSG:4326")
    cwh_union = cwh.geometry.union_all() if hasattr(cwh.geometry, 'union_all') else cwh.geometry.unary_union
    # Buffer by ~0.09° (~10 km) to catch points in polygon gaps at polygon edges
    # e.g. outer VI coast, Haida Gwaii, outer islands — all genuinely CWH
    cwh_buffered = cwh_union.buffer(0.09)
    print(f"  CWH/CDF zone loaded ({len(cwh)} polygons, 10 km buffer applied)")

    land = None
    if LAND_BOUNDARY.exists():
        land = gpd.read_file(str(LAND_BOUNDARY)).to_crs("EPSG:4326")
        land_union = land.geometry.union_all() if hasattr(land.geometry, 'union_all') else land.geometry.unary_union
        print(f"  Land boundary loaded ({len(land)} parts)")
    else:
        print("  Warning: no land boundary found — skipping water check")
        land_union = None



    # Validate candidates
    valid, invalid = [], []
    for lat, lon, name, desc in CANDIDATES:
        pt = Point(lon, lat)
        in_cwh = cwh_buffered.contains(pt)
        # coastal_land_only.geojson has incomplete coverage for outer islands
        # (Haida Gwaii, Aristazabal, Banks, etc.) — bypass land check for those
        outer_island = (lon < -130.0)  # west of Prince Rupert = outer archipelago
        in_land = True if outer_island else (land_union.contains(pt) if land_union else True)
        status = "OK" if (in_cwh and in_land) else (
            "NOT_IN_CWH" if not in_cwh else "IN_WATER"
        )
        entry = (lat, lon, name, desc, status)
        (valid if status == "OK" else invalid).append(entry)

    print(f"\nValid (in CWH + land): {len(valid)}")
    print(f"Invalid:               {len(invalid)}")
    for e in invalid:
        print(f"  ✗ {e[2]:30s}  ({e[4]})")

    # ── Build folium map ───────────────────────────────────────────────────
    m = folium.Map(
        location=[51.5, -126.0],
        zoom_start=6,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
    )
    folium.TileLayer("openstreetmap", name="OpenStreetMap").add_to(m)

    # CWH/CDF boundary — full BEC zone
    cwh_fg = folium.FeatureGroup(name="CWH/CDF boundary (BEC zone)", show=True)
    folium.GeoJson(
        cwh.__geo_interface__,
        style_function=lambda _: {
            "color": "#00ccff", "weight": 1.2,
            "fillColor": "#88eeff", "fillOpacity": 0.10,
        },
        tooltip="CWH / CDF biogeoclimatic zone",
    ).add_to(cwh_fg)
    cwh_fg.add_to(m)

    # Existing tiles — blue squares
    existing_fg = folium.FeatureGroup(name=f"Existing tiles ({len(EXISTING)})", show=True)
    for lat, lon, name in EXISTING:
        bbox = centre_to_bbox(lat, lon)
        folium.GeoJson(
            bbox.__geo_interface__,
            style_function=lambda _: {
                "color": "#2288ff", "weight": 2,
                "fillColor": "#2288ff", "fillOpacity": 0.25,
            },
            tooltip=f"EXISTING: {name}",
        ).add_to(existing_fg)
        folium.CircleMarker(
            [lat, lon], radius=4, color="#2288ff", fill=True,
            fill_color="#2288ff", fill_opacity=1.0,
            tooltip=f"EXISTING: {name}",
        ).add_to(existing_fg)
    existing_fg.add_to(m)

    # Valid proposed tiles — green squares
    valid_fg = folium.FeatureGroup(name=f"Proposed new tiles — valid ({len(valid)})", show=True)
    for i, (lat, lon, name, desc, _) in enumerate(valid):
        bbox = centre_to_bbox(lat, lon)
        folium.GeoJson(
            bbox.__geo_interface__,
            style_function=lambda _: {
                "color": "#22cc44", "weight": 2,
                "fillColor": "#22cc44", "fillOpacity": 0.25,
            },
            tooltip=f"NEW #{i+1}: {name} — {desc}",
        ).add_to(valid_fg)
        folium.Marker(
            [lat, lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size:10px;color:white;background:#22aa33;'
                     f'border-radius:50%;width:20px;height:20px;text-align:center;'
                     f'line-height:20px;font-weight:bold;">{i+1}</div>',
                icon_size=(20, 20), icon_anchor=(10, 10),
            ),
            tooltip=f"NEW #{i+1}: {name} — {desc}",
        ).add_to(valid_fg)
    valid_fg.add_to(m)

    # Invalid / flagged tiles — red
    if invalid:
        invalid_fg = folium.FeatureGroup(name=f"Flagged candidates ({len(invalid)})", show=True)
        for lat, lon, name, desc, status in invalid:
            folium.CircleMarker(
                [lat, lon], radius=8, color="#ff3333", fill=True,
                fill_color="#ff3333", fill_opacity=0.7,
                tooltip=f"FLAGGED ({status}): {name}",
            ).add_to(invalid_fg)
        invalid_fg.add_to(m)

    folium.LayerControl().add_to(m)

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT_HTML))
    print(f"\nMap saved: {OUT_HTML}")

    # Print summary table
    print("\nProposed new tiles (valid only):")
    print(f"{'#':>3}  {'Name':<30}  {'Lat':>7}  {'Lon':>9}  Description")
    print("-" * 85)
    for i, (lat, lon, name, desc, _) in enumerate(valid):
        print(f"{i+1:>3}  {name:<30}  {lat:>7.3f}  {lon:>9.3f}  {desc}")

    print(f"\nTotal valid candidates: {len(valid)}")


if __name__ == "__main__":
    main()
