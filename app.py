# Update the Streamlit app to support labeling features from the shapefile.
# Adds a sidebar checkbox to enable labels and a field selector to choose which attribute to draw.
# Saves back to /mnt/data/app.py

updated_app_code = r"""
import os
import io
import zipfile
import tempfile
import numpy as np
import streamlit as st
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import shapefile  # pyshp
from pyproj import CRS, Transformer

st.set_page_config(page_title="Raster Map Generator", layout="wide")
st.title("🗺️ Raster Map Generator (Classes 1–5)")
st.caption("Upload reclassified TIFF rasters (values 1–5) and a **zipped shapefile** boundary. Get maps with title, legend, north arrow, degree grid, and optional labels.")

# ---------------- Helpers ----------------
def parse_prj_wkt(prj_bytes):
    try:
        wkt = prj_bytes.decode("utf-8")
        return CRS.from_wkt(wkt)
    except Exception:
        return None

def load_boundary_with_attrs_from_zip(shp_zip_bytes, dst_epsg=4326):
    \"\"\"
    Returns:
      segments: list of Nx2 arrays for boundary plotting (in EPSG:4326)
      label_points: list of dicts: {\"x\": float, \"y\": float, \"attrs\": {field:value,...}}
      field_names: list of available attribute field names
    \"\"\"
    # Unzip to memory
    with zipfile.ZipFile(io.BytesIO(shp_zip_bytes), "r") as z:
        names = z.namelist()
        shp_name = None
        for n in names:
            if n.lower().endswith(\".shp\"):
                shp_name = n
                break
        if not shp_name:
            raise ValueError(\"No .shp found in ZIP.\")
        base = os.path.splitext(shp_name)[0]

        shp_bytes = z.read(base + \".shp\")
        shx_bytes = z.read(base + \".shx\") if (base + \".shx\") in names else None
        dbf_bytes = z.read(base + \".dbf\") if (base + \".dbf\") in names else None
        prj_bytes = z.read(base + \".prj\") if (base + \".prj\") in names else None

    shp_io = io.BytesIO(shp_bytes)
    shx_io = io.BytesIO(shx_bytes) if shx_bytes else None
    dbf_io = io.BytesIO(dbf_bytes) if dbf_bytes else None
    r = shapefile.Reader(shp=shp_io, shx=shx_io, dbf=dbf_io)

    # fields (skip DeletionFlag)
    fields = [f[0] for f in r.fields if f[0] != \"DeletionFlag\"]
    recs = r.records() if dbf_bytes else []
    # source CRS
    if prj_bytes:
        src_crs = parse_prj_wkt(prj_bytes)
    else:
        src_crs = CRS.from_epsg(4326)

    transformer = None
    if src_crs and src_crs.to_epsg() != dst_epsg:
        transformer = Transformer.from_crs(src_crs, CRS.from_epsg(dst_epsg), always_xy=True)

    segments = []
    label_points = []  # centroid-ish label positions with attrs
    shapes = r.shapes()
    for idx, s in enumerate(shapes):
        attrs = {}
        if dbf_bytes and idx < len(recs):
            values = list(recs[idx])
            # Map values to fields safely
            for k, v in zip(fields, values):
                attrs[k] = v

        if s.shapeType in (shapefile.POLYGON, shapefile.POLYGONZ, shapefile.POLYGONM):
            pts = np.array(s.points)
            parts = list(s.parts) + [len(pts)]
            # label pos: mean of exterior ring coords
            if len(parts) >= 2:
                ring = pts[parts[0]:parts[1]]
                if ring.shape[0] >= 3:
                    if transformer:
                        x, y = transformer.transform(ring[:,0], ring[:,1])
                        ring = np.column_stack([x, y])
                    # Save label point at centroid approx
                    cx, cy = float(np.mean(ring[:,0])), float(np.mean(ring[:,1]))
                    label_points.append({\"x\": cx, \"y\": cy, \"attrs\": attrs})

            for i in range(len(parts)-1):
                seg = pts[parts[i]:parts[i+1]]
                if seg.shape[0] >= 2:
                    if transformer:
                        x, y = transformer.transform(seg[:,0], seg[:,1])
                        seg = np.column_stack([x, y])
                    segments.append(seg)

        elif s.shapeType in (shapefile.POLYLINE, shapefile.POLYLINEZ, shapefile.POLYLINEM):
            pts = np.array(s.points)
            parts = list(s.parts) + [len(pts)]
            # label pos: mean of line coords
            if len(pts) >= 2:
                arr = pts
                if transformer:
                    x, y = transformer.transform(arr[:,0], arr[:,1])
                    arr = np.column_stack([x, y])
                cx, cy = float(np.mean(arr[:,0])), float(np.mean(arr[:,1]))
                label_points.append({\"x\": cx, \"y\": cy, \"attrs\": attrs})

            for i in range(len(parts)-1):
                seg = pts[parts[i]:parts[i+1]]
                if seg.shape[0] >= 2:
                    if transformer:
                        x, y = transformer.transform(seg[:,0], seg[:,1])
                        seg = np.column_stack([x, y])
                    segments.append(seg)
        else:
            # optionally support points in future
            pass

    return segments, label_points, fields

def reproject_to_epsg4326(src_dataset):
    dst_crs = \"EPSG:4326\"
    transform, width, height = calculate_default_transform(
        src_dataset.crs, dst_crs, src_dataset.width, src_dataset.height, *src_dataset.bounds
    )
    dst_array = np.empty((src_dataset.count, height, width), dtype=src_dataset.dtypes[0])
    for i in range(1, src_dataset.count + 1):
        reproject(
            source=rasterio.band(src_dataset, i),
            destination=dst_array[i - 1],
            src_transform=src_dataset.transform,
            src_crs=src_dataset.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
    return dst_array, transform, dst_crs

def add_north_arrow(ax, x_rel=1.02, y_rel=0.5, length=0.12):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx = x1 - x0
    dy = y1 - y0
    x_anchor = x0 + x_rel * dx
    y_anchor = y0 + y_rel * dy
    arrow = FancyArrowPatch((x_anchor, y_anchor - length*dy/2),
                            (x_anchor, y_anchor + length*dy/2),
                            arrowstyle='-|>', mutation_scale=15, linewidth=1.5)
    ax.add_patch(arrow)
    ax.text(x_anchor, y_anchor + length*dy/2 + 0.01*dy, \"N\",
            ha=\"center\", va=\"bottom\", fontsize=10)

def choose_tick_step(span):
    if span > 5: return 1.0
    if span > 2: return 0.5
    if span > 1: return 0.25
    if span > 0.5: return 0.1
    return 0.05

def plot_one_map(raster_bytes, raster_name, title, boundary_segments, label_points=None, label_field=None, label_fontsize=9):
    tif_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=\".tif\")
    tif_tmp.write(raster_bytes)
    tif_tmp.flush()
    tif_tmp.close()

    with rasterio.open(tif_tmp.name) as src:
        if src.crs is None:
            raise ValueError(f\"'{raster_name}' has no CRS defined. Please set a CRS before upload.\")
        dst_array, dst_transform, _ = reproject_to_epsg4326(src)

    band = dst_array[0].astype(float)
    with rasterio.open(tif_tmp.name) as src_check:
        nodata = src_check.nodata
    if nodata is not None:
        band[band == nodata] = np.nan
    band[band == 0] = np.nan

    height, width = band.shape
    west, north = (dst_transform.c, dst_transform.f)
    east = west + dst_transform.a * width
    south = north + dst_transform.e * height
    extent = [west, east, south, north]

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    mappable = ax.imshow(band, extent=extent, origin=\"upper\", interpolation=\"nearest\")
    cbar = plt.colorbar(mappable, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(\"Class (1=Very Low … 5=Very High)\")
    cbar.set_ticks([1,2,3,4,5])
    cbar.set_ticklabels([\"1 Very Low\",\"2 Low\",\"3 Moderate\",\"4 High\",\"5 Very High\"])

    for seg in boundary_segments:
        ax.plot(seg[:,0], seg[:,1], linewidth=1.2, color=\"black\")

    # Optional labels
    if label_points and label_field:
        for pt in label_points:
            val = pt[\"attrs\"].get(label_field, None)
            if val is None:
                continue
            try:
                txt = str(val)
            except Exception:
                txt = \"\"
            if txt:
                ax.text(pt[\"x\"], pt[\"y\"], txt, fontsize=label_fontsize, ha=\"center\", va=\"center\", bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1.5))

    ax.set_title(title or raster_name, pad=10, fontsize=12)
    ax.set_xlabel(\"Longitude (°)\")
    ax.set_ylabel(\"Latitude (°)\")

    ax.grid(True, linestyle=\"--\", alpha=0.4)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_step = choose_tick_step(abs(x_max - x_min))
    y_step = choose_tick_step(abs(y_max - y_min))
    xticks = np.arange(np.floor(x_min/x_step)*x_step, x_max + x_step, x_step)
    yticks = np.arange(np.floor(y_min/y_step)*y_step, y_max + y_step, y_step)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    add_north_arrow(ax, x_rel=1.02, y_rel=0.5, length=0.12)

    buf = io.BytesIO()
    fig.savefig(buf, format=\"png\", bbox_inches=\"tight\")
    plt.close(fig)
    buf.seek(0)
    return buf

# ---------------- Sidebar ----------------
st.sidebar.header(\"Inputs\")
tif_files = st.sidebar.file_uploader(
    \"Upload reclassified TIFFs (1–5)\",
    type=[\"tif\", \"tiff\"],
    accept_multiple_files=True
)
shp_zip = st.sidebar.file_uploader(
    \"Upload boundary as ZIPPED shapefile (.zip)\",
    type=[\"zip\"],
    accept_multiple_files=False
)

custom_titles = {}
if tif_files:
    st.sidebar.markdown(\"**Optional titles**\")
    for f in tif_files:
        base = os.path.splitext(f.name)[0]
        custom_titles[f.name] = st.sidebar.text_input(f\"Title for {f.name}\", value=base)

# Label controls
show_labels = st.sidebar.checkbox(\"Show labels from shapefile field\", value=False)
label_field = None
label_fontsize = st.sidebar.number_input(\"Label font size\", value=9, min_value=6, max_value=24, step=1)

generate = st.sidebar.button(\"🚀 Generate Maps\")

# ---------------- Main ----------------
if generate:
    try:
        if not tif_files:
            st.error(\"Please upload at least one TIFF.\")
            st.stop()
        if not shp_zip:
            st.error(\"Please upload the boundary shapefile as a zipped .zip file.\")
            st.stop()

        shp_zip_bytes = shp_zip.read()
        segments, label_points, field_names = load_boundary_with_attrs_from_zip(shp_zip_bytes, dst_epsg=4326)
        if not segments:
            st.error(\"No plottable geometry found in the boundary ZIP.\")
            st.stop()

        # If user wants labels, show field selector dynamically
        if show_labels and field_names:
            label_field = st.sidebar.selectbox(\"Choose label field\", field_names, index=0)

        out_pngs = []
        col1, col2 = st.columns(2)
        cols = [col1, col2]
        idx = 0
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, \"w\", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in tif_files:
                try:
                    title = custom_titles.get(f.name) if custom_titles else os.path.splitext(f.name)[0]
                    img_buf = plot_one_map(
                        f.read(), f.name, title,
                        segments,
                        label_points=label_points if show_labels and label_field else None,
                        label_field=label_field,
                        label_fontsize=label_fontsize
                    )
                    out_name = f\"{os.path.splitext(f.name)[0]}.png\"
                    zf.writestr(out_name, img_buf.getvalue())
                    out_pngs.append((out_name, img_buf.getvalue()))
                    with cols[idx % 2]:
                        st.image(img_buf, caption=out_name, use_container_width=True)
                    idx += 1
                except Exception as e:
                    st.error(f\"Failed for {f.name}: {e}\")

        if out_pngs:
            zbuf.seek(0)
            st.success(f\"Generated {len(out_pngs)} map(s).\" )
            st.download_button(
                \"⬇️ Download all maps (ZIP)\",
                data=zbuf,
                file_name=\"generated_maps.zip\",
                mime=\"application/zip\"
            )
        else:
            st.warning(\"No maps were generated.\")
    except Exception as e:
        st.exception(e)

st.caption(\"Tip: If your TIFFs have no CRS, define it before upload. The app reprojects to EPSG:4326 for degree grids.\")
"""

with open("/mnt/data/app.py", "w", encoding="utf-8") as f:
    f.write(updated_app_code)

"/mnt/data/app.py updated with labeling support."
