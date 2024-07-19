# %%

from h3 import h3
import folium

# Polyfill a Geo Json with hexagons
geoJson1 = {
    "type": "Polygon",
    "coordinates": [[[90, -180], [90, 0], [-90, 0], [-90, -180]]],
}
geoJson2 = {
    "type": "Polygon",
    "coordinates": [[[90, 0], [90, 180], [-90, 180], [-90, 0]]],
}
hexagons = list(h3.polyfill(geoJson1, 1)) + list(h3.polyfill(geoJson2, 1))
# %%
polylines = []
for hex in hexagons:
    polygons = h3.h3_set_to_multi_polygon([hex], geo_json=False)
    outlines = [loop for polygon in polygons for loop in polygon]
    polyline = [outline + [outline[0]] for outline in outlines][0]
    polylines.append(polyline)
# %%
base = folium.Map([0, 0], zoom_start=2, tiles="cartodbpositron")
for polyline in polylines:
    m = folium.PolyLine(locations=polyline, weight=1, color="black")
    base.add_child(m)
m.save("test.html")
# %%

from pyogrio import read_dataframe

coastline = read_dataframe()
