#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body {{ margin:0; overflow:hidden; }}
    #plot {{ width:100vw; height:100vh; }}
    /* Hover preview image */
    #hover-img {{
      position: fixed;
      display: none;
      max-width: {max_px}px;
      max-height: {max_px}px;
      box-shadow: 0 6px 16px rgba(0,0,0,.3);
      border: 1px solid rgba(0,0,0,.2);
      background: #fff;
      pointer-events: none;
      z-index: 9999;
    }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <img id="hover-img" alt="preview" />

  <!-- Inline data so it works via file:// without a server -->
  <script type="application/json" id="tsne-data">{json_data}</script>

  <script>
    // Parse inline JSON
    const raw = document.getElementById('tsne-data').textContent;
    let data = [];
    try {{
      data = JSON.parse(raw);
    }} catch (e) {{
      document.body.innerHTML = '<pre style="color:red">Failed to parse embedded JSON: ' + e + '</pre>';
      throw e;
    }}

    const trace = {{
      x: data.map(d => d.x),
      y: data.map(d => d.y),
      mode: 'markers',
      type: 'scatter',
      text: data.map(d => d.label),
      hoverinfo: 'text', // show label
      marker: {{
        size: 8,
        color: data.map(d => d.color)
      }}
    }};

    const layout = {{
      title: '{title}',
      xaxis: {{ showgrid: false }},
      yaxis: {{ showgrid: false }},
      hovermode: 'closest',
      margin: {{ l:40, r:40, t:40, b:40 }}
    }};

    Plotly.newPlot('plot', [trace], layout).then(gd => {{
      const img = document.getElementById('hover-img');
      const PAD = 14; // px gap from cursor
      const clamp = (v, min, max) => Math.max(min, Math.min(max, v));

      function placeImage(clientX, clientY) {{
        const vw = window.innerWidth, vh = window.innerHeight;
        const iw = img.naturalWidth ? Math.min(img.naturalWidth, {max_px}) : {max_px};
        const ih = img.naturalHeight ? Math.min(img.naturalHeight, {max_px}) : {max_px};

        let left = clientX + PAD;
        let top  = clientY + PAD;
        if (left + iw > vw) left = clientX - PAD - iw;
        if (top  + ih > vh) top  = clientY - PAD - ih;

        left = clamp(left, 0, vw - iw);
        top  = clamp(top,  0, vh - ih);

        img.style.left = left + 'px';
        img.style.top  = top  + 'px';
      }}

      gd.on('plotly_hover', ev => {{
        const p = ev.points[0];
        const d = data[p.pointIndex];
        if (!d || !d.img) return;

        // set src only if changed (avoid flicker/reload)
        const abs = (new URL(d.img, window.location.href)).href;
        if (img.src !== abs) img.src = d.img;

        img.style.display = 'block';
        placeImage(ev.event.clientX, ev.event.clientY);
      }});

      gd.on('plotly_unhover', () => {{
        img.style.display = 'none';
      }});

      // keep the preview near the cursor as it moves (while hovering)
      gd.addEventListener('mousemove', ev => {{
        if (img.style.display === 'block') placeImage(ev.clientX, ev.clientY);
      }});

      // click opens full image
      gd.on('plotly_click', ev => {{
        const idx = ev.points[0].pointIndex;
        const d = data[idx];
        if (d && d.img) window.open(d.img, '_blank');
      }});
    }}).catch(err => {{
      document.body.innerHTML = '<pre style="color:red">' + err + '</pre>';
    }});
  </script>
</body>
</html>
"""

def main():
    parser = argparse.ArgumentParser(
        description="Generate a static t-SNE HTML with hover image preview (no server needed)."
    )
    parser.add_argument("json_path", help="Path to tsne_data.json")
    parser.add_argument("html_out", help="Output HTML file, e.g., index_hover.html")
    parser.add_argument("--title", default="t-SNE Visualization",
                        help="Page and plot title (default: %(default)s)")
    parser.add_argument("--max-thumb", type=int, default=240,
                        help="Max thumbnail size in pixels (default: %(default)s)")
    args = parser.parse_args()

    json_path = Path(args.json_path)
    html_out = Path(args.html_out)

    # Load JSON, then minify for embedding
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    json_min = json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    html = HTML_TEMPLATE.format(
        title=args.title,
        json_data=json_min,
        max_px=args.max_thumb
    )
    html_out.write_text(html, encoding="utf-8")
    print(f"✓ Wrote {html_out} — open it directly (file://) and hover points to preview images.")

if __name__ == "__main__":
    main()

