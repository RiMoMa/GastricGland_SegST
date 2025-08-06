import os
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, box
import openslide
import cv2
import numpy as np

# ==== CONFIGURACIÓN ====
input_dir = "/home/ricardo/Glands-Urkunina/input_data/images/"
output_img_dir = "./patches_10x/images/"
output_mask_dir = "./patches_20x/masks/"
patch_size = 512
target_magnification = 20

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# ==== FUNCIONES ====
def parse_asap_xml(xml_file):
    gland_polys = []
    region_polys = []
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for ann in root.findall(".//Annotation"):
        name = ann.attrib.get("Name", "").strip().lower()
        coords_tag = ann.find("Coordinates")
        if coords_tag is None:
            continue
        coords = [(float(c.attrib['X']), float(c.attrib['Y'])) for c in coords_tag.findall("Coordinate")]
        if len(coords) < 3:
            continue
        poly = Polygon(coords)
        if name.startswith("gland") or name.startswith("annotation"):
            gland_polys.append(poly)
        elif "region" in name:
            region_polys.append(poly)
    return gland_polys, region_polys

def polygon_in_regions(poly, regions):
    return any(poly.intersects(r) for r in regions)

def patch_fully_inside_region(x, y, patch_size, regions):
    half = patch_size / 2
    patch_poly = box(x - half, y - half, x + half, y + half)
    return any(r.contains(patch_poly) for r in regions)

# ==== CONTADORES ====
annotated_cases = 0
total_glands_in_regions = 0

# ==== PROCESAR TODAS LAS IMÁGENES ====
for file in os.listdir(input_dir):
    if not file.lower().endswith(".xml"):
        continue
    case_id = os.path.splitext(file)[0]
    xml_path = os.path.join(input_dir, file)
    wsi_path = os.path.join(input_dir, case_id + ".svs")
    if not os.path.exists(wsi_path):
        print(f"[!] No se encontró WSI para {case_id}")
        continue

    gland_polys, region_polys = parse_asap_xml(xml_path)
    valid_glands = [g for g in gland_polys if polygon_in_regions(g, region_polys)]

    if len(region_polys) > 0:
        annotated_cases += 1
        total_glands_in_regions += len(valid_glands)

    slide = openslide.OpenSlide(wsi_path)
    base_magnification = float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    scale_factor = base_magnification / target_magnification if base_magnification > target_magnification else 1.0
    adjusted_patch_size = int(patch_size * scale_factor)

    saved_count = 0
    for idx, gland_poly in enumerate(valid_glands):
        cx, cy = gland_poly.centroid.coords[0]
        if not patch_fully_inside_region(cx, cy, adjusted_patch_size, region_polys):
            continue

        x0 = int(cx - adjusted_patch_size / 2)
        y0 = int(cy - adjusted_patch_size / 2)
        img = slide.read_region((x0, y0), 0, (adjusted_patch_size, adjusted_patch_size)).convert("RGB")
        img = np.array(img)[:, :, :3]
        img = cv2.resize(img, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

        mask = np.zeros((adjusted_patch_size, adjusted_patch_size), dtype=np.uint8)
        for poly in valid_glands:
            if poly.intersects(box(x0, y0, x0 + adjusted_patch_size, y0 + adjusted_patch_size)):
                shifted = [(px - x0, py - y0) for px, py in poly.exterior.coords]
                cv2.fillPoly(mask, [np.array(shifted, dtype=np.int32)], 1)

        mask = cv2.resize(mask, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)

        img_filename = f"{case_id}_patch_{idx:04d}.png"
        mask_filename = f"{case_id}_patch_{idx:04d}.png"
        cv2.imwrite(os.path.join(output_img_dir, img_filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_mask_dir, mask_filename), mask * 255)
        saved_count += 1

    print(f"✅ {case_id}: {saved_count} parches con máscara guardados")

print(f"\nResumen: {annotated_cases} casos con anotaciones de experto, {total_glands_in_regions} glándulas dentro de regiones anotadas")
