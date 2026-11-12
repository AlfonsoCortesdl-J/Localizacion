"""Fetches real building footprints from OpenStreetMap for TecMM Campus Zapopan."""
import urllib.request, urllib.parse, json, math

QUERY = """[out:json][timeout:30];
(
  way["building"](20.699,-103.479,20.708,-103.470);
  way["leisure"](20.699,-103.479,20.708,-103.470);
  way["amenity"](20.699,-103.479,20.708,-103.470);
  way["sport"](20.699,-103.479,20.708,-103.470);
);
out center tags;"""

MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

# Campus center & scale
CX_LAT = 20.703454
CX_LON = -103.474731
K_LAT  = 110713.0
K_LON  = 104218.0

def gps_to_3d(lat, lon):
    x = (lon - CX_LON) * K_LON
    z = -(lat - CX_LAT) * K_LAT
    return round(x, 1), round(z, 1)

result = None
for mirror in MIRRORS:
    try:
        data = urllib.parse.urlencode({"data": QUERY}).encode()
        req = urllib.request.Request(mirror, data=data, headers={"User-Agent": "TecMM-3D/1.0"})
        with urllib.request.urlopen(req, timeout=25) as r:
            result = json.loads(r.read())
        print(f"OK via {mirror}")
        break
    except Exception as e:
        print(f"FAIL {mirror}: {e}")

if not result:
    print("All mirrors failed.")
else:
    elements = result.get("elements", [])
    print(f"\nTotal OSM elements: {len(elements)}\n")
    buildings = []
    for el in elements:
        tags = el.get("tags", {})
        center = el.get("center", {})
        lat = center.get("lat")
        lon = center.get("lon")
        if lat is None or lon is None:
            continue
        x, z = gps_to_3d(lat, lon)
        name = tags.get("name") or tags.get("amenity") or tags.get("leisure") or tags.get("sport") or tags.get("building") or "?"
        btype = tags.get("building") or tags.get("leisure") or tags.get("amenity") or tags.get("sport") or ""
        levels = tags.get("building:levels", "?")
        height = tags.get("height", "?")
        buildings.append({"name": name, "type": btype, "levels": levels, "height": height,
                          "lat": lat, "lon": lon, "x": x, "z": z})
        print(f"  [{btype:12s}] L:{levels:3s} H:{height:6s} | {name[:35]:35s} | X={x:+7.1f}m  Z={z:+7.1f}m")

    # Save for use in HTML
    with open("data/campus_osm_buildings.json", "w", encoding="utf-8") as f:
        json.dump(buildings, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to data/campus_osm_buildings.json")
