#!/usr/bin/env python3
"""
Build an interactive Leaflet map of FAIB permanent sample plots (PSPs) that
contain Pacific yew (TW), for browser inspection.

Output: docs/faib_yew_map.html  (open in any browser; uses OSM tiles + CDN Leaflet)

Run:
    conda run -n yew_pytorch python scripts/analysis/build_faib_yew_map.py
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path("/home/jericho/yew_project")
OUT  = ROOT / "docs/faib_yew_map.html"

ZONE_COLOR = {"CWH": "#0072B2", "ICH": "#D55E00", "CDF": "#009E73",
              "IDF": "#E69F00", "OTHER": "#999999"}


def main():
    td = pd.read_csv(ROOT / "data/raw/faib_tree_detail.csv",
                     usecols=["SITE_IDENTIFIER", "CLSTR_ID", "VISIT_NUMBER", "PLOT",
                              "TREE_NO", "SPECIES", "DBH", "LV_D"], low_memory=False)
    h = pd.read_csv(ROOT / "data/raw/faib_header.csv", low_memory=False)

    tw = td[td.SPECIES == "TW"].copy()
    # dedupe to one record per tree (latest visit), matching the population analysis
    tw = (tw.sort_values("VISIT_NUMBER")
            .groupby(["SITE_IDENTIFIER", "PLOT", "TREE_NO"], as_index=False).last())

    per_site = tw.groupby("SITE_IDENTIFIER").agg(
        n_yew=("DBH", "size"),
        n_live=("LV_D", lambda s: (s == "L").sum()),
        max_dbh=("DBH", "max"),
        median_dbh=("DBH", "median"),
        n_mature=("DBH", lambda s: (s >= 10).sum()),
    ).reset_index()

    hh = h[["SITE_IDENTIFIER", "Longitude", "Latitude", "BEC_ZONE", "BEC_SBZ",
            "SAMPLE_ESTABLISHMENT_TYPE"]].drop_duplicates("SITE_IDENTIFIER")
    sites = per_site.merge(hh, on="SITE_IDENTIFIER", how="left").dropna(
        subset=["Longitude", "Latitude"])

    feats = []
    for _, r in sites.iterrows():
        zone = r.BEC_ZONE if r.BEC_ZONE in ZONE_COLOR else "OTHER"
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r.Longitude, r.Latitude]},
            "properties": {
                "site": str(r.SITE_IDENTIFIER), "zone": r.BEC_ZONE,
                "sbz": r.BEC_SBZ, "stype": r.SAMPLE_ESTABLISHMENT_TYPE,
                "n_yew": int(r.n_yew), "n_live": int(r.n_live),
                "n_mature": int(r.n_mature),
                "max_dbh": round(float(r.max_dbh), 1),
                "median_dbh": round(float(r.median_dbh), 1),
                "color": ZONE_COLOR[zone],
            }})
    fc = {"type": "FeatureCollection", "features": feats}

    zone_counts = sites.BEC_ZONE.value_counts().to_dict()
    legend = "".join(
        f'<div><span style="background:{ZONE_COLOR.get(z, ZONE_COLOR["OTHER"])}"></span>'
        f'{z} ({zone_counts.get(z, 0)})</div>'
        for z in ["CWH", "ICH", "CDF", "IDF"])

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>FAIB PSPs with Pacific yew — BC</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body,#map{{height:100%;margin:0}}
  #map{{height:100vh}}
  .info,.legend{{background:#fff;padding:8px 10px;border-radius:6px;
     box-shadow:0 1px 5px rgba(0,0,0,.3);font:13px/1.4 sans-serif}}
  .legend span{{display:inline-block;width:12px;height:12px;margin-right:6px;
     border-radius:50%;vertical-align:middle}}
  .title{{font-weight:bold;margin-bottom:4px}}
</style></head><body><div id="map"></div><script>
var map = L.map('map').setView([51.0, -125.0], 6);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
  {{maxZoom:14, attribution:'© OpenStreetMap'}}).addTo(map);
var data = {json.dumps(fc)};
L.geoJSON(data, {{
  pointToLayer: function(f, latlng) {{
    var p = f.properties;
    var r = 4 + Math.min(8, p.n_yew*0.5);
    return L.circleMarker(latlng, {{radius:r, fillColor:p.color, color:'#222',
      weight:0.8, fillOpacity:0.8}});
  }},
  onEachFeature: function(f, layer) {{
    var p = f.properties;
    layer.bindPopup('<div class="title">FAIB site '+p.site+'</div>'+
      'BEC: <b>'+p.zone+' / '+p.sbz+'</b><br>'+
      'Sample type: '+p.stype+'<br>'+
      'Yew stems: <b>'+p.n_yew+'</b> ('+p.n_live+' live, '+p.n_mature+' \\u226510 cm)<br>'+
      'Max DBH: '+p.max_dbh+' cm &middot; median '+p.median_dbh+' cm');
  }}
}}).addTo(map);
var legend = L.control({{position:'bottomright'}});
legend.onAdd = function() {{
  var d = L.DomUtil.create('div','legend');
  d.innerHTML = '<div class="title">FAIB PSPs with yew (n={len(feats)})</div>'+
    '{legend}'+'<div style="margin-top:4px;color:#666">marker size \\u221d yew stems</div>';
  return d;
}};
legend.addTo(map);
</script></body></html>"""

    OUT.write_text(html)
    print(f"Wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
    print(f"  {len(feats)} yew PSP sites; zones: {zone_counts}")
    print(f"  Open in browser:  file://{OUT}")


if __name__ == "__main__":
    main()
