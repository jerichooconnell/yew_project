from geotessera import GeoTessera

# Initialize the client
gt = GeoTessera()

# Method 1: Fetch a single tile
embedding, crs, transform = gt.fetch_embedding(lon=0.15, lat=52.05, year=2024)
print(f"Shape: {embedding.shape}")  # e.g., (1200, 1200, 128)
print(f"CRS: {crs}")  # Coordinate reference system from landmask

# Method 2: Fetch all tiles in a bounding box
bbox = (-0.2, 51.4, 0.1, 51.6)  # (min_lon, min_lat, max_lon, max_lat)
embeddings = gt.fetch_embeddings(bbox, year=2024)

for tile_lon, tile_lat, embedding_array, crs, transform in embeddings:
    print(f"Tile ({tile_lat}, {tile_lon}): {embedding_array.shape}")