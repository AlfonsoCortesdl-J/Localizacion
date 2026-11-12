import json

with open('data/campus_buildings_detected.json', encoding='utf-8') as f:
    blds = json.load(f)

print('const CAMPUS_BUILDINGS = [')
for b in blds[:30]:
    area = b['area_m2']
    if area > 2000: h = 12
    elif area > 800: h = 9
    elif area > 300: h = 7
    else: h = 5
    w = min(b['w_m'], 150)
    d = min(b['h_m'], 150)
    x = b['x3d']
    z = b['z3d']
    line = "  { x:%.1f, y:%.1f, z:%.1f, w:%.0f, h:%d, d:%.0f, area:%.0f }," % (x, h/2.0, z, w, h, d, area)
    print(line)
print('];')
