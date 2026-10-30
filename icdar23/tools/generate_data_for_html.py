import os, json, subprocess
from pathlib import Path

import numpy as np
import umap


OUTPUT_JSON = "umap_data.json"

# ------------------------------------------------------
# Utility: Convert RGBA → HEX (opacity ignored)
# ------------------------------------------------------
def rgba_to_hex(rgba):
    r, g, b, _ = rgba
    return "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))

# needed: Encoding (N,D), labels (N,), files (N,), colors (N,4)


reducer = umap.UMAP(random_state=42, n_neighbors=12, min_dist=0.1)
reduced = reducer.fit_transform(encodings)





data = []
for i in range(len(encodings)):
    data.append({
        "x": float(reduced[i][0]),
        "y": float(reduced[i][1]),
        "label": labels[i],
        "color": colors[i],
        "img": os.path.join("images", f"{Path(files[i]).stem}.png") # Assuming you have generated thumbnails in this directory
    })

# Save
with open(OUTPUT_JSON, "w") as f:
    json.dump(data, f, indent=2)

print(f"Saved JSON output → {OUTPUT_JSON}")



# call generate_html.py to create HTML with hover preview
import subprocess
subprocess.run(["python3", "create_html.py", 
                OUTPUT_JSON, 
                "WR-umap-hover.html",
                "--title", "Writer Retrieval UMAP Visualization",
                "--max-thumb", "200"
                ])

exit()
