
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

# ---------- Page Setup ----------
st.set_page_config(page_title="Raster Map Generator", layout="wide")
st.title("🗺️ Raster Map Generator (Classes 1–5)")
st.caption("Upload reclassified TIFF rasters (values 1–5) and a **zipped shapefile** boundary. Get maps with title, legend, north arrow, and degree grid.")

# ---------- Helpers: read boundary from ZIP (robust without GeoPandas) ----------
def load_boundary_segments_from_zip(shp_zip_bytes, dst_epsg=4326):
    import fiona
    from pyproj import Transformer

    # Write the in-memory zip to a temp file for fiona
    tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp_zip.write(shp_zip_bytes.getvalue())
    tmp_zip.flush()
    tmp_zip.close()

    zippath = f"zip://{tmp_zip.name}"

    segments = []
    with fiona.open(zippath) as src:
        src_crs = src.crs
        if src_crs:
            transformer = Transformer.from_crs(src_crs, f"EPSG:{dst_epsg}", always_xy=True)
        else:
            transformer = None

        for feat in src:
            geom = feat["geometry"]
            if geom is None:
                continue
            gtype = geom["type"]
            coords = geom["coordinates"]

            def transform_coords(arr):
                if transformer:
                    x, y = transformer.transform(arr[:, 0], arr[:, 1])
                    return np.column_stack([x, y])
                return arr

            if gtype == "Polygon":
                ext = np.array(coords[0])
                segments.append(transform_coords(ext))
            elif gtype == "MultiPolygon":
                for poly in coords:
                    ext = np.array(poly[0])
                    segments.append(transform_coords(ext))
            elif gtype in ("LineString", "MultiLineString"):
                if gtype == "LineString":
                    arrs = [np.array(coords)]
                else:
                    arrs = [np.array(ls) for ls in coords]
                for arr in arrs:
                    segments.append(transform_coords(arr))
    return segments

def reproject_to_epsg4326(src_dataset):
    dst_crs = "EPSG:4326"
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
            resampling=Resampling.nearest,  # categorical
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
    ax.text(x_anchor, y_anchor + length*dy/2 + 0.01*dy, "N",
            ha="center", va="bottom", fontsize=10)

def choose_tick_step(span):
    if span > 5: return 1.0
    if span > 2: return 0.5
    if span > 1: return 0.25
    if span > 0.5: return 0.1
    return 0.05

def plot_one_map(raster_bytes, raster_name, title, boundary_segments):
    # Write TIFF to temp file for rasterio
    tif_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    tif_tmp.write(raster_bytes.getvalue())
    tif_tmp.flush()
    tif_tmp.close()

    with rasterio.open(tif_tmp.name) as src:
        if src.crs is None:
            raise ValueError(f"'{raster_name}' has no CRS defined. Please set a CRS before upload.")
        # Reproject to EPSG:4326
        dst_array, dst_transform, _ = reproject_to_epsg4326(src)

    band = dst_array[0].astype(float)
    # Treat nodata or zeros as transparent
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
    mappable = ax.imshow(band, extent=extent, origin="upper", interpolation="nearest")
    cbar = plt.colorbar(mappable, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Class (1=Very Low … 5=Very High)")
    cbar.set_ticks([1, 2, 3, 4, 5])
    cbar.set_ticklabels(["1 Very Low", "2 Low", "3 Moderate", "4 High", "5 Very High"])

    # Overlay boundary
    for seg in boundary_segments:
        ax.plot(seg[:,0], seg[:,1], linewidth=1.2, color="black")

    ax.set_title(title or raster_name, pad=10, fontsize=12)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")

    ax.grid(True, linestyle="--", alpha=0.4)
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
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

# ---------- Sidebar Inputs ----------
st.sidebar.header("Inputs")
tif_files = st.sidebar.file_uploader(
    "Upload reclassified TIFFs (1–5)",
    type=["tif", "tiff"],
    accept_multiple_files=True
)
shp_zip = st.sidebar.file_uploader(
    "Upload boundary as ZIPPED shapefile (.zip)",
    type=["zip"],
    accept_multiple_files=False
)

# Optional custom titles for each raster
custom_titles = {}
if tif_files:
    st.sidebar.markdown("**Optional titles**")
    for f in tif_files:
        base = os.path.splitext(f.name)[0]
        custom_titles[f.name] = st.sidebar.text_input(f"Title for {f.name}", value=base)

generate = st.sidebar.button("🚀 Generate Maps")

# ---------- Main Action ----------
if generate:
    if not tif_files:
        st.error("Please upload at least one TIFF.")
        st.stop()
    if not shp_zip:
        st.error("Please upload the boundary shapefile as a zipped .zip file.")
        st.stop()

    # Load boundary
    try:
        boundary_segments = load_boundary_segments_from_zip(io.BytesIO(shp_zip.read()), dst_epsg=4326)
        if not boundary_segments:
            st.error("No plottable geometry found in the boundary ZIP.")
            st.stop()
    except Exception as e:
        st.exception(e)
        st.stop()

    # Process rasters
    out_pngs = []
    col1, col2 = st.columns(2)
    cols = [col1, col2]
    idx = 0
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in tif_files:
            try:
                title = custom_titles.get(f.name) if custom_titles else os.path.splitext(f.name)[0]
                img_buf = plot_one_map(io.BytesIO(f.read()), f.name, title, boundary_segments)
                out_name = f"{os.path.splitext(f.name)[0]}.png"
                zf.writestr(out_name, img_buf.getvalue())
                out_pngs.append((out_name, img_buf.getvalue()))

                # Preview in UI
                with cols[idx % 2]:
                    st.image(img_buf, caption=out_name, use_container_width=True)
                idx += 1
            except Exception as e:
                st.error(f"Failed for {f.name}: {e}")

    if out_pngs:
        zip_buffer.seek(0)
        st.success(f"Generated {len(out_pngs)} map(s).")
        st.download_button(
            "⬇️ Download all maps (ZIP)",
            data=zip_buffer,
            file_name="generated_maps.zip",
            mime="application/zip"
        )
    else:
        st.warning("No maps were generated.")

# ---------- Footer ----------
st.caption("Tip: If your TIFFs have no CRS, define it before upload. The app reprojects to EPSG:4326 for degree grids.")
