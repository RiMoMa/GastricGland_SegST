import os
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, box
import openslide
import cv2
import numpy as np
import random
import math

# ==== CONFIGURACIÓN ====
input_dir = "/home/ricardo/Glands-Urkunina/input_data/images/"
output_base = "./patches_20xR/"
patch_size = 510+130
target_magnification = 20

# Control de dataset
max_unlabeled_per_case = 20   # límite máximo de parches positivos por WSI unlabeled
min_distance_px = 500          # distancia mínima entre parches en coordenadas WSI (solo unlabeled)
neg_ratio = 0.2                # proporción de parches negativos extraídos en unlabeled

# Carpetas de salida
for subset in ["labeled/images", "labeled/masks", "unlabeled/images", "unlabeled/masks"]:
    os.makedirs(os.path.join(output_base, subset), exist_ok=True)

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

def filter_by_distance(glands, min_dist):
    selected = []
    for g in glands:
        cx, cy = g.centroid.coords[0]
        if all(math.dist((cx, cy), (sx, sy)) > min_dist for sx, sy in selected):
            selected.append((cx, cy))
    return selected

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

    # Determinar si es labeled o unlabeled
    is_labeled = len(region_polys) > 0
    if is_labeled:
        annotated_cases += 1
        valid_glands = [g for g in gland_polys if polygon_in_regions(g, region_polys)]
        total_glands_in_regions += len(valid_glands)
        gland_centroids = [(g.centroid.x, g.centroid.y) for g in valid_glands]  # sin filtro
    else:
        valid_glands = gland_polys
        gland_centroids = filter_by_distance(valid_glands, min_distance_px)
        if len(gland_centroids) > max_unlabeled_per_case:
            gland_centroids = random.sample(gland_centroids, max_unlabeled_per_case)

    # Abrir la WSI
    slide = openslide.OpenSlide(wsi_path)
    base_magnification = float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    scale_factor = base_magnification / target_magnification if base_magnification > target_magnification else 1.0
    adjusted_patch_size = int(patch_size * scale_factor)

    saved_count = 0
    # Extraer parches positivos
    for idx, (cx, cy) in enumerate(gland_centroids):
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

        subset = "labeled" if is_labeled else "unlabeled"
        img_filename = f"{case_id}_patch_{idx:04d}.png"
        mask_filename = f"{case_id}_patch_{idx:04d}.png"
        cv2.imwrite(os.path.join(output_base, subset, "images", img_filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_base, subset, "masks", mask_filename), mask * 255)
        saved_count += 1

    # Extraer parches negativos para unlabeled
    if not is_labeled and neg_ratio > 0:
        neg_count = int(len(gland_centroids) * neg_ratio)
        wsi_w, wsi_h = slide.dimensions
        neg_patches = 0
        while neg_patches < neg_count:
            cx = random.randint(adjusted_patch_size // 2, wsi_w - adjusted_patch_size // 2)
            cy = random.randint(adjusted_patch_size // 2, wsi_h - adjusted_patch_size // 2)
            # Comprobar que no intersecta con ninguna glándula
            if not any(Polygon(g.exterior).intersects(box(cx - adjusted_patch_size / 2,
                                                         cy - adjusted_patch_size / 2,
                                                         cx + adjusted_patch_size / 2,
                                                         cy + adjusted_patch_size / 2))
                       for g in valid_glands):
                x0 = int(cx - adjusted_patch_size / 2)
                y0 = int(cy - adjusted_patch_size / 2)
                img = slide.read_region((x0, y0), 0, (adjusted_patch_size, adjusted_patch_size)).convert("RGB")
                img = np.array(img)[:, :, :3]
                img = cv2.resize(img, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
                mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
                subset = "unlabeled"
                img_filename = f"{case_id}_neg_{neg_patches:04d}.png"
                mask_filename = f"{case_id}_neg_{neg_patches:04d}.png"
                cv2.imwrite(os.path.join(output_base, subset, "images", img_filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(output_base, subset, "masks", mask_filename), mask)
                neg_patches += 1

    print(f"✅ {case_id} ({'labeled' if is_labeled else 'unlabeled'}): {saved_count} positivos guardados")

print(f"\nResumen: {annotated_cases} casos con anotaciones de experto, {total_glands_in_regions} glándulas dentro de regiones anotadas")
