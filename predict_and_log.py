import pandas as pd
import datetime
import ee
import joblib
import os
import json

# ------------------------------
# Authenticate Earth Engine
# ------------------------------
SERVICE_ACCOUNT = os.getenv("SERVICE_ACCOUNT")
KEY_JSON = os.getenv("KEY_JSON")

credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, key_data=KEY_JSON)
ee.Initialize(credentials, project='rainfall-functionality-predict')

# ------------------------------
# Load model and well locations
# ------------------------------
model = joblib.load("model.pkl")
wells_df = pd.read_csv("ASDF_Wells_Cleaned.csv")
wells_df[['Longitude', 'Latitude']] = wells_df['coords'].str.strip("()").str.split(",", expand=True).astype(float)
wells_df['coords'] = list(zip(wells_df['Longitude'], wells_df['Latitude']))
wells = wells_df.drop_duplicates(subset='coords')

# ------------------------------
# Define date range: past 7 days
# ------------------------------
today = datetime.date.today()
start_date = today - datetime.timedelta(days=7)
end_date = today

# ------------------------------
# Fetch rainfall + predict risk
# ------------------------------
def fetch_rainfall(twp_id, name, coords):
    try:
        point = ee.Geometry.Point(coords)
        chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
            .filterDate(str(start_date), str(end_date)) \
            .select('precipitation')

        def extract(image):
            date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
            value = image.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point,
                scale=5000
            ).get('precipitation')
            return ee.Feature(None, {'date': date, 'precipitation': value})

        features = chirps.map(extract).getInfo()
        if 'features' not in features or not features['features']:
            return None  # No data found

        data = []
        for f in features['features']:
            props = f.get('properties', {})
            data.append({
                'date': props.get('date'),
                'rainfall': props.get('precipitation')
            })

        df = pd.DataFrame(data)
        df['rainfall'] = pd.to_numeric(df['rainfall'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'])
        df['twp_id'] = twp_id
        df['name'] = name
        df['lon'] = coords[0]
        df['lat'] = coords[1]
        df['rolling_7d'] = df['rainfall'].rolling(7).sum()
        return df.tail(1)  # latest record only

    except Exception as e:
        print(f"Error fetching for {name}: {e}")
        return None

# ------------------------------
# Collect data from all wells
# ------------------------------
all_data = []
for _, row in wells.iterrows():
    df = fetch_rainfall(row['twp_id'], row['name'], row['coords'])
    if df is not None:
        all_data.append(df)

# ------------------------------
# Predict and save
# ------------------------------
if all_data:
    df = pd.concat(all_data)
    df['failure_risk'] = model.predict_proba(df[['rolling_7d']].fillna(0))[:, 1]

    def label_risk(r):
        if r >= 0.75: return 'High'
        elif r >= 0.5: return 'Medium'
        return 'Low'

    df['risk_level'] = df['failure_risk'].apply(label_risk)
    df['date_fetched'] = pd.Timestamp.now().normalize()

    if os.path.exists("risk_history.csv"):
        existing = pd.read_csv("risk_history.csv")
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv("risk_history.csv", index=False)
    print("✅ risk_history.csv updated")
else:
    print("⚠️ No rainfall data fetched.")
