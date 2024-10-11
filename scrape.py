import requests
import requests
import csv

# API URL for GeoJSON data
url = 'https://www.azstat.gov.az/webmap/geojson/inzibati_rayonlar_2024.geojson'

# Headers
headers = {
    'referer': 'https://www.azstat.gov.az/webmap/index.php?geolevel=rayonlar&v=off&year=2015&indicator=00360&section=11&colorFrom=edf8fb&colorTo=810f7c&cc=5&ms=method_Q',
    'sec-ch-ua': '"Google Chrome";v="129", "Not=A?Brand";v="8", "Chromium";v="129"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
    'dnt': '1'
}

# Send GET request
response = requests.get(url, headers=headers)

# Check for success
if response.status_code == 200:
    data = response.json()  # Parse GeoJSON data

    # Extract relevant data (for example, properties and years between 2015 and 2023)
    features = data['features']

    # Open CSV file for writing
    with open('scraped_data_2015_2023.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write header (adapt based on actual fields)
        writer.writerow(['id', 'name', 'year', 'indicator_value'])

        # Iterate through features and save relevant data
        for feature in features:
            properties = feature['properties']
            # Assuming years and indicators are available in properties
            for year in range(2015, 2024):
                # Filter data for the desired years (update the logic based on the actual structure)
                indicator_value = properties.get(f'indicator_{year}', None)
                if indicator_value is not None:
                    writer.writerow([properties.get('id'), properties.get('name'), year, indicator_value])

    print("Data successfully saved to 'scraped_data_2015_2023.csv'.")
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")
