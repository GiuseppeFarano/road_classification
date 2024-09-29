import requests
from PIL import Image
from io import BytesIO


def convert_to_decimal(degrees, minutes=0, seconds=0, direction='N'):
    decimal = degrees + (minutes / 60) + (seconds / 3600)
    if direction in ['S', 'W']:
        decimal = -decimal
    return decimal


# Configurazione
MAPBOX_API_KEY = 'pk.eyJ1IjoiZ2l1c2VwcGVmYXJhbm8iLCJhIjoiY20xbmMwYnpwMG9xMzJtczh1bmE4OXphayJ9.Uw7UK4gmfncTBBsNHzjlVw'
lon_deg = 9
lon_min = 11
lon_sec = 22.1
lon_dir = 'E'

lat_deg = 45
lat_min = 30
lat_sec = 09.7
lat_dir = 'N'

# Conversione
lon = convert_to_decimal(lon_deg, lon_min, lon_sec, lon_dir)
lat = convert_to_decimal(lat_deg, lat_min, lat_sec, lat_dir)

zoom = 19
width, height = 256, 256

# URL dell'API per l'immagine satellitare statica
url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},{zoom}/{width}x{height}?access_token={MAPBOX_API_KEY}"

# Effettua la richiesta per ottenere l'immagine satellitare
try:
    response = requests.get(url)
    response.raise_for_status()  # Alza un'eccezione per codici di stato HTTP 4xx/5xx
    # Carica l'immagine in memoria e la salva in locale
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image.save("satellite_image.png")
    print("Immagine salvata come 'satellite_image.png'")
    image.show()  # Mostra l'immagine
except requests.exceptions.HTTPError as err:
    print(f"Errore durante il download dell'immagine: {err}")
except Exception as e:
    print(f"Si è verificato un errore: {e}")
