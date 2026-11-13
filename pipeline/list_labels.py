"""List every label name in FGVD and how often it appears."""
from unicodedata import name
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

FGVD = Path(__file__).resolve().parent.parent / "data" / "fgvd"

counts = Counter() 

types = Counter()

for xml_file in FGVD.rglob("*.xml"):                      # every XML file
    for obj in ET.parse(xml_file).getroot().iter("object"):  # every box in that file
        name = obj.find("name").text                       # e.g. "scooter_Honda_Activa"
        counts[name] += 1
        types[name.split("_")[0]] += 1                     
# inside the loop, next to counts[...]:
types[name.split("_")[0]] += 1
# at the end:
print(f"{len(counts)} different labels, {sum(counts.values())} boxes in total")

print("\nVehicle types:")
for t, n in types.most_common():
    print(f"{n:6d}  {t}")
print(f"{len(types)} different vehicle types")