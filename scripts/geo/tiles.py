import logging
import os 
import json
import psycopg
from pathlib import Path
from dataclasses import dataclass
from psycopg.errors import UniqueViolation

from softmaxx.config import AppConfig, DatabaseConfig 
from softmaxx.config import get_logger_config, get_database_config
from softmaxx.geo.tiles import get_polygon_intersection_tiles, TileMatrix, WebMapTile


logger = logging.getLogger("main." + __name__)

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


def _create_geo_tile(conn: psycopg.Connection, tile: WebMapTile) -> int:
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
            aoi_tiles = get_polygon_intersection_tiles(aoi_coordinates, computation_detail.zoom_level)

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




def start_worker():
    print(f"start geo polygon process under PID: {os.getpid()}...")
    AppConfig.load()
    log_config = get_logger_config("local")
    AppConfig.init_logging(log_file=log_config.log_file, log_level=log_config.log_level)
    
    logger.info(f"xboa sdk config loaded...")
    tiles: list[WebMapTile]= get_polygon_intersection_tiles("polygon.json", 14, tile_matrix=TileMatrix.WEB_MAP)
    for tile in tiles:
        print(tile)

    # add_aoi_polygon("bihta_block", "polygon.json")
    # add_computation("RAIN", 10)
    # link_computation_to_aoi("RAIN", "bihta_block")
   


if __name__ == "__main__":
    start_worker()