import logging
import os 
import json
import math
from pathlib import Path
from dataclasses import dataclass
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient

import psycopg
from psycopg.errors import UniqueViolation
from softmaxx.config import AppConfig, DatabaseConfig 
from softmaxx.config import get_logger_config, get_database_config


logger = logging.getLogger("main." + __name__)


@dataclass(frozen=True)
class MapTile:
    x: int
    y: int
    z: int
    packed_id: int
    area_fraction: float

@dataclass(frozen=True)
class ComputationDetail:
    id: int
    name: str
    zoom_level: int

@dataclass(frozen=True)
class PolygonDetail:
    id: int
    name: str
    geometry: str


# Constants for Web Mercator (EPSG:3857) projection math
# EARTH_RADIUS = 6378137.0
# ORIGIN_SHIFT = math.pi * EARTH_RADIUS  # ~20037508.34 meters


def _pack_map_tile(x: int, y: int, z: int) -> int:
    """Packs X, Y, and Z tile coordinates into a single 64-bit integer ID.

    Bit-Packing layout allocation:
    - Z: bits 0-5    (allows zoom levels 0 to 63)
    - X: bits 6-34   (allows X grid indices up to 536,870,911)
    - Y: bits 35-63  (allows Y grid indices up to 536,870,911)

    Returns:
        int: Unique 64-bit packed integer ID
    """
    # 1. Cast inputs to integers to guarantee correct bitwise execution
    x_val = int(x)
    y_val = int(y)
    z_val = int(z)

    # 2. Shift components into their designated bit boundaries and combine them
    packed_id = (y_val << 35) | (x_val << 6) | z_val
    return packed_id


def _unpack_map_tile(packed_id: int) -> tuple[int, int, int]:
    """Decodes a 64-bit packed integer back into its original X, Y, and Z tile coordinates.
    
    Bit-Packing layout mapping:
    - Z: bits 0-5    (mask 63)
    - X: bits 6-34   (mask 536870911)
    - Y: bits 35-63  (mask 536870911)
    
    Returns:
        tuple: (x, y, z) as pure integers
    """
    z = packed_id & 63
    x = (packed_id >> 6) & 536870911
    y = (packed_id >> 35) & 536870911
    return x, y, z


def _orient_polygon(coords: list[list[float]]) -> list[list[float]]:
    """Ensures coordinates are ordered counter-clockwise (positive area)."""
    if not coords or len(coords) < 3:
        return coords
    
    # Close the ring if it isn't closed
    if coords[0] != coords[-1]:
        coords = coords + [coords[0]]
        
    # Calculate signed area to check orientation
    edge_sum = 0.0
    for i in range(len(coords) - 1):
        x1, y1 = coords[i][0], coords[i][1] # lon, lat
        x2, y2 = coords[i+1][0], coords[i+1][1]
        edge_sum += (x2 - x1) * (y2 + y1)
        
    # If edge_sum > 0, it's clockwise. Reverse it to make it CCW.
    if edge_sum > 0:
        return coords[::-1]
    return coords


def _get_wgs84_metric_scales(center_lat: float) -> tuple[float, float]:
    """
    Replicates your JS getGlobalMetricScales using standard WGS-84 
    Ellipsoid constants to find exact meters per degree of Lat and Lon.
    """
    lat_rad = math.radians(center_lat)
    
    a = 6378137.0                # Equatorial radius in meters
    b = 6356752.314245           # Polar radius in meters
    e_sq = 1 - (b * b) / (a * a) # Square of eccentricity

    # Calculate radius of curvature along the meridian (North-South)
    denominator = (1 - e_sq * (math.sin(lat_rad) ** 2)) ** 1.5
    m = (a * (1 - e_sq)) / denominator

    # Calculate radius of curvature along the prime vertical (East-West)
    n = a / math.sqrt(1 - e_sq * (math.sin(lat_rad) ** 2))

    lat_to_meters = (math.pi / 180.0) * m
    lng_to_meters = (math.pi / 180.0) * n * math.cos(lat_rad)
    
    return lat_to_meters, lng_to_meters


def _get_polygon_shoe_lace_area(coords: list[list[float]]) -> float:
    """Calculates true ellipsoidal flat surface area in square meters."""
    if not coords or len(coords) < 3:
        return 0.0
        
    # Close the ring if needed
    if coords[0] != coords[-1]:
        coords = coords + [coords[0]]

    # 1. Compute Centroid/Center Point
    sum_lng = sum(p[0] for p in coords[:-1])
    sum_lat = sum(p[1] for p in coords[:-1])
    n_points = len(coords) - 1
    
    center_lng = sum_lng / n_points
    center_lat = sum_lat / n_points
    
    # 2. Get Ellipsoid Scale Modifiers
    lat_to_meters, lng_to_meters = _get_wgs84_metric_scales(center_lat)
    
    # 3. Project to Local Center Meters
    projected = [
        ((p[0] - center_lng) * lng_to_meters, (p[1] - center_lat) * lat_to_meters)
        for p in coords
    ]
    
    # 4. Shoelace Formula
    area = 0.0
    num_pts = len(projected)
    for i in range(num_pts - 1):
        area += projected[i][0] * projected[i+1][1]
        area -= projected[i+1][0] * projected[i][1]
        
    return abs(area / 2.0)


def _local_project_array(coords: list[list[float]], 
                         c_lng: float, 
                         c_lat: float, 
                         lat_to_meters: float, lng_to_meters: float) -> list[tuple[float, float]]:
    """
    This function takes a set of global geographic coordinates (Latitude/Longitude) 
    and flattens them into a simple, local 2D plane measured in physical meters.
    By converting the complex polygon and grid tile into this temporary, flat metric 
    canvas, Shapely can accurately compute intersection areas using true meters.
    """
    return [
        ((lon - c_lng) * lng_to_meters, (lat - c_lat) * lat_to_meters)
        for lon, lat in coords
    ]



def _get_map_tiles(polygon_coords: list[list[float]], zoom: int) -> list[MapTile]:
    """Finds intersecting tiles and calculates area fractions using raw array mathematics."""
    # 1. Instantiate the raw Shapely polygon
    # Note: Shapely expects coordinate pairs as (longitude, latitude) [X, Y]
    raw_polygon = Polygon(polygon_coords)
    
    # 2. Enforce Counter-Clockwise (CCW) orientation using native orient
    target_polygon = orient(raw_polygon, sign=1.0)
    
    # 3. Extract degree bounding box directly from the corrected shape bounds
    min_lng, min_lat, max_lng, max_lat = target_polygon.bounds
    
    # Extract the freshly oriented coordinate array to pass to your local projector
    # .exterior.coords outputs tuples, convert back to list[list[float]] to match your signature
    corrected_coords = [list(pt) for pt in target_polygon.exterior.coords]

    # Standard tile conversion equations
    def lon2tile(lon, z): return math.floor((lon + 180) / 360 * (2 ** z))
    def lat2tile(lat, z):
        lat = max(min(lat, 85.0511), -85.0511)
        lat_rad = math.radians(lat)
        return math.floor((1 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * (2 ** z))
    
    def tile2lon(x, z): return x / (2 ** z) * 360.0 - 180.0
    def tile2lat(y, z):
        n = math.pi - 2.0 * math.pi * y / (2 ** z)
        return 180.0 / math.pi * math.atan(0.5 * (math.exp(n) - math.exp(-n)))

    # Compute raw grid bounds
    x_min, x_max = sorted([lon2tile(min_lng, zoom), lon2tile(max_lng, zoom)])
    y_min, y_max = sorted([lat2tile(max_lat, zoom), lat2tile(min_lat, zoom)])
    
    tiles = []

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            w, e = tile2lon(x, zoom), tile2lon(x + 1, zoom)
            n, s = tile2lat(y, zoom), tile2lat(y + 1, zoom)

            # Build structural bounding tile geometry
            tile_box_wgs84 = box(min(w, e), min(s, n), max(w, e), max(s, n))

            # Fast bounding intersection check
            if target_polygon.intersects(tile_box_wgs84):
                c_lng = (w + e) / 2.0
                c_lat = (s + n) / 2.0

                lat_to_meters, lng_to_meters = _get_wgs84_metric_scales(c_lat)

                # Project the tile corner geometry array using your math
                tile_array_wgs84 = [[w, n], [e, n], [e, s], [w, s], [w, n]]
                
                # Transform arrays to native Shapely meters shapes using corrected_coords
                local_tile_shape = Polygon(_local_project_array(tile_array_wgs84, c_lng, c_lat, lat_to_meters, lng_to_meters))
                local_poly_shape = Polygon(_local_project_array(corrected_coords, c_lng, c_lat, lat_to_meters, lng_to_meters))

                # Compute the area intersection subset
                intersection_geom = local_poly_shape.intersection(local_tile_shape)
                
                if not intersection_geom.is_empty:
                    area_fraction = intersection_geom.area / local_tile_shape.area
                    area_fraction = round(area_fraction, 4)

                    if area_fraction > 0.0001:
                        packed_id = _pack_map_tile(int(x), int(y), int(zoom))
                        tiles.append(
                            MapTile(x=int(x), y=int(y), z=int(zoom), 
                                    area_fraction=area_fraction, packed_id=packed_id)
                        )
    return tiles



def _store_aoi_polygon(conn: psycopg.Connection, polygon_name, polygon_object):
    with conn.cursor() as cur:
        insert_query = """
            INSERT INTO polygon_master (name, raw_geometry)
            VALUES (%s, %s)
            RETURNING polygon_id;
        """

        # Pass the python dictionary directly.
        # Psycopg automatically intercepts the dict and casts it into Postgres JSONB.
        cur.execute(insert_query, (polygon_name, psycopg.types.json.Jsonb(polygon_object)))
        # Fetch the auto-generated primary key
        polygon_id = cur.fetchone()[0]
        # Commit transaction safely
        conn.commit()
        return polygon_id


def _add_computation(conn: psycopg.Connection, computation_name, zoom_level):
    with conn.cursor() as cur:
        insert_query = """
            INSERT INTO computation_master (name, zoom_level)
            VALUES (%s, %s)
            RETURNING computation_id;
        """

        cur.execute(insert_query, (computation_name, zoom_level))
        computation_id = cur.fetchone()[0]
        # Commit transaction safely
        conn.commit()
        return computation_id


def _create_polygon_computation(conn: psycopg.Connection, polygon_id, computation_id):
    query = """
        INSERT INTO polygon_computation(polygon_id, computation_id)
        VALUES (%s, %s)
        RETURNING polygon_comp_id;
    """
    with conn.cursor() as cur:
        try:
            cur.execute(query, (polygon_id, computation_id))
        except UniqueViolation as e:
            logger.error(f"record already exists for polygon_id={polygon_id} and computation_id={computation_id}")
            raise e


def _create_geo_tile(conn: psycopg.Connection, tile: MapTile) -> int:
    """Finds or creates a tile in the geo_tiles table. """

    with conn.cursor() as cur:
       
        insert_query = """
            INSERT INTO geo_tiles (tile_id, tile_z, tile_x, tile_y)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (tile_id) 
            DO UPDATE SET tile_id = EXCLUDED.tile_id
            RETURNING tile_id;
        """
        cur.execute(insert_query, (tile.packed_id, tile.z, tile.x, tile.y))

        # fetchone will return a tuple 
        # we need to ensure that we return the first item of tuple
        result = cur.fetchone()
        if result:
            return result[0] 

        # 2. If result is None, the tile already existed. Fetch its ID.
        select_query = """
            SELECT tile_id 
            FROM geo_tiles 
            WHERE tile_z = %s AND tile_x = %s AND tile_y = %s;
        """
        cur.execute(select_query, (tile.z, tile.x, tile.y))
        existing_result = cur.fetchone()
        if not existing_result:
            raise ValueError(f"fatal: tile x:{tile.x} y:{tile.y} z:{tile.z} not created or found!")
        return existing_result[0]


def _create_polygon_tile(conn: psycopg.Connection, polygon_id: int, tile_id: int, area_fraction: float) -> int:
    with conn.cursor() as cur:
        # 1. Attempt insertion. 
        # If it already exists, do nothing but keep transaction clean.
        insert_query = """
            INSERT INTO polygon_tiles (polygon_id, tile_id, area_fraction)
            VALUES (%s, %s, %s)
            ON CONFLICT (polygon_id, tile_id) DO NOTHING
        """
        cur.execute(insert_query, (polygon_id, tile_id, area_fraction))


def _create_computation_tile(conn: psycopg.Connection, computation_id: int, tile_id: int) -> int:
    with conn.cursor() as cur:
        # 1. Attempt insertion. 
        # If it already exists, do nothing but keep transaction clean.
        insert_query = """
            INSERT INTO computation_tiles (computation_id, tile_id)
            VALUES (%s, %s)
            ON CONFLICT (computation_id, tile_id) DO NOTHING
        """
        cur.execute(insert_query, (computation_id, tile_id))


def _get_computation_detail(conn: psycopg.Connection, name: str) -> tuple:
    with conn.cursor() as cur:
        select_query = """
            SELECT computation_id, name, zoom_level
            FROM computation_master
            WHERE name = %s;
        """
        cur.execute(select_query, (name,))
        result = cur.fetchone()

        if not result:
            logger.error(f"Computation with name '{name}' not found.")
            raise ValueError(f"Computation '{name}' does not exist.")

        # 2. Instantiate and return the clean object container
        row_id, row_name, row_zoom = result
        return ComputationDetail(name=row_name, zoom_level=int(row_zoom), id = int(row_id))


def _get_polygon_detail(conn: psycopg.Connection, name: str) -> tuple:
    query = """
           SELECT polygon_id, name, raw_geometry 
           FROM polygon_master 
           WHERE name = %s;
    """
   
    with conn.cursor() as cur:
        cur.execute(query, (name,))
        result = cur.fetchone()
        if not result:
            logger.error(f"polygon with name '{name}' not found.")
            raise ValueError(f"polygon '{name}' does not exist.")
        row_id, row_name, row_geometry = result 
        return PolygonDetail(id=int(row_id), geometry=row_geometry, name=row_name)




# ###########################
# 
# Public methods 
# 
# ############################# 


def link_computation_to_aoi(computation_name: str, aoi_name: str) -> int:
    """
    To link a computation to an AOI polygon,
    (1) we find the tiles that define the polygon at computation zoom level. 
    (2) we store each tile as independent GEO_TILE
    (3) Link GEO_TILE to AOI_POLYGON 
    (4) Link GEO_TILE to Computation 
    After this linking, running a computation means doing computation over 
    linked GEO_TILES.

    """
    logger = logging.getLogger("main." + __name__)
    db_config:DatabaseConfig = get_database_config()    
    with psycopg.connect(**db_config.get_map()) as conn:
        try:
            polygon_detail: PolygonDetail = _get_polygon_detail(conn, aoi_name)
            computation_detail: ComputationDetail = _get_computation_detail(conn, computation_name)

            logger.info(f"polygon id is {polygon_detail.id}")
            logger.info(f"computation id is {computation_detail.id}")
            logger.info(f"computation z-level is {computation_detail.zoom_level}")
             
            _create_polygon_computation(conn, polygon_detail.id, computation_detail.id)
            raw_coordinates = polygon_detail.geometry["coordinates"]
            aoi_coordinates = raw_coordinates[0]
            logger.info(f"AOI polygon coordinates are {aoi_coordinates}")
            aoi_tiles = _get_map_tiles(aoi_coordinates, computation_detail.zoom_level)

            for tile in aoi_tiles:
                logger.info(f"insert tile x: {tile.x}, y:{tile.y}, {tile.z}")
                tile_id = _create_geo_tile(conn, tile)
                _create_polygon_tile(conn, polygon_detail.id, tile_id, tile.area_fraction)
                _create_computation_tile(conn, computation_detail.id, tile_id)
                
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.exception(f"error happened. database changes are rolled back.")
            raise e 
            
    return 


def add_computation(computation_name, zoom_level):
    """
    add a a new computation with native resolution = zoom_level 
    The zoom_level here is the same as google map zoom level
    We have to find the native resolution or zoom level for each 
    computation
    """
    db_config:DatabaseConfig = get_database_config()
    with psycopg.connect(**db_config.get_map()) as conn:
        try:
            computation_id = _add_computation(conn, computation_name, zoom_level) 
            logger.info("computation inserted with id: " + str(computation_id))
        except UniqueViolation:
            logger.info("A computation with the name {0} already exists!".format(computation_name))


def add_aoi_polygon(aoi_name, polygon_file):
    """
    add area of interest polygon with a name in the database 
    The AOI (area of interest) is a polygon of geo points. 
    We store each AOI with a name. 
    """
    file_path = Path(polygon_file)
    if not file_path.exists() or not file_path.is_file():
        logger.error(f"Target GeoJSON source path does not exist: {file_path}")
        raise FileNotFoundError(f"Missing file: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        geometry = json.load(file)
    
    # 3. Validate geometry type and extract coordinates
    geom_type = geometry.get("type")
    if geom_type not in ["Polygon", "MultiPolygon"]:
        raise ValueError(f"Unsupported geometry type: {geom_type}. Expected Polygon or MultiPolygon.")


    db_config:DatabaseConfig = get_database_config()
    with psycopg.connect(**db_config.get_map()) as conn:
        try:
            polygon_id = _store_aoi_polygon(conn, aoi_name, geometry) 
            logger.info("polygon inserted with id: " + str(polygon_id))
        except UniqueViolation:
            logger.info("A polygon with the name {0} already exists!".format(aoi_name))


def show_polygon_tiles(polygon_file, zoom_level):
    """
    Get a polygon or MultiPolygon points in a file
    The method prints the x,y,z tiles at supplied zoom levels 
    along with intersection of polygon and tile as a fraction 
    of tile area. 
    """
    file_path = Path(polygon_file)
    if not file_path.exists() or not file_path.is_file():
        logger.error(f"Target GeoJSON source path does not exist: {file_path}")
        raise FileNotFoundError(f"Missing file: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        geometry = json.load(file)
    
    # 3. Validate geometry type and extract coordinates
    geom_type = geometry.get("type")
    if geom_type not in ["Polygon", "MultiPolygon"]:
        raise ValueError(f"Unsupported geometry type: {geom_type}. Expected Polygon or MultiPolygon.")

    raw_coordinates = geometry["coordinates"]
    coordinates = raw_coordinates[0]
    tiles = _get_map_tiles(coordinates, zoom_level)
    for tile in tiles:
        print(f"x: {tile.x}, y: {tile.y}, z: {tile.z}, fraction: {tile.area_fraction}, packed_id:{tile.packed_id}")

    

def start_worker():
    print(f"start geo polygon process under PID: {os.getpid()}...")
    AppConfig.load()
    log_config = get_logger_config("local")
    AppConfig.init_logging(log_file=log_config.log_file, log_level=log_config.log_level)
    logger.info(f"xboa sdk config loaded...")
    show_polygon_tiles("polygon.json", 14)
    # add_aoi_polygon("bihta_block", "polygon.json")
    # add_computation("RAIN", 10)
    # link_computation_to_aoi("RAIN", "bihta_block")
   


if __name__ == "__main__":
    start_worker()