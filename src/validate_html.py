content = open('src/web/campus_drone.html', encoding='utf-8').read()
opens = content.count('<script')
closes = content.count('</script>')
print(f'Script tags: {opens} opens, {closes} closes')
if 'type="module"' in content:
    print('ERROR: type=module encontrado!')
else:
    print('OK: sin type=module')
if 'cdn.jsdelivr' in content or 'unpkg.com' in content:
    print('WARNING: CDN externo encontrado!')
else:
    print('OK: sin CDN externo')
lines = content.split('\n')
for i, l in enumerate(lines):
    if 'three.min.js' in l and 'src=' in l:
        print(f'three.min.js en linea {i+1}: {l.strip()}')
    if 'OrbitControls.js' in l and 'src=' in l:
        print(f'OrbitControls.js en linea {i+1}: {l.strip()}')
    if 'THREE.OrbitControls' in l and 'new ' in l:
        print(f'OrbitControls instancia linea {i+1}: {l.strip()}')
print(f'Total lineas: {len(lines)}')
