import json
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
meta_path = ROOT / "data" / "tecmm_campus_meta.json"

with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

bbox = f"{meta['lat_min']:.6f},{meta['lon_min']:.6f},{meta['lat_max']:.6f},{meta['lon_max']:.6f}"
query = f"""[out:json][timeout:15];
(
  way["building"]({bbox});
  way["leisure"]({bbox});
  way["amenity"]({bbox});
);
out body;
>;
out skel qt;"""

url = "https://overpass-api.de/api/interpreter?data=" + urllib.parse.quote(query)
req = urllib.request.Request(url, headers={"User-Agent": "TecMMBuilder/1.0"})

try:
    with urllib.request.urlopen(req, timeout=12) as r:
        d = json.loads(r.read().decode("utf-8"))
        ways = [e for e in d.get("elements", []) if e.get("type") == "way"]
        print(f"Total campus OSM ways found: {len(ways)}")
        nodes = {e["id"]: (e["lat"], e["lon"]) for e in d.get("elements", []) if e.get("type") == "node"}
        for w in ways[:12]:
            w_nodes = [nodes[nid] for nid in w.get("nodes", []) if nid in nodes]
            if w_nodes:
                avg_lat = sum(n[0] for n in w_nodes) / len(w_nodes)
                avg_lon = sum(n[1] for n in w_nodes) / len(w_nodes)
                # Distance in meters from center
                dx = (avg_lon - meta["center_lon"]) * meta["k_lon"]
                dy = (avg_lat - meta["center_lat"]) * meta["k_lat"]
                print(f"  Way {w.get('id')}: {w.get('tags', {}).get('name', w.get('tags', {}).get('building', 'bldg'))} at dx={dx:.1f}m, dy={dy:.1f}m")

        out_file = ROOT / "data" / "tecmm_osm_features.json"
        with open(out_file, "w", encoding="utf-8") as out_f:
            json.dump(d, out_f, indent=2)
except Exception as e:
    print(f"Error fetching OSM: {e}")
