import streamlit as st
import pandas as pd
import ee
import datetime

# ----------------------------------------
# Earth Engine authentication
# ----------------------------------------
import json

SERVICE_ACCOUNT = st.secrets["SERVICE_ACCOUNT"]
KEY_JSON = st.secrets["KEY_JSON"]

credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, key_data=json.loads(KEY_JSON))
ee.Initialize(credentials)

# ----------------------------------------
# Load Cleaned ASDF Wells (222 unique)
# ----------------------------------------
@st.cache_data
def load_location_records():
    df = pd.read_csv("ASDF_Wells_Cleaned.csv")
    df[['Longitude', 'Latitude']] = df['coords'].str.strip("()") \
        .str.split(",", expand=True).astype(float)
    df['coords'] = list(zip(df['Longitude'], df['Latitude']))
    df = df.drop_duplicates(subset='coords')
    return [
        {'twp_id': row['twp_id'], 'name': row['name'], 'coords': row['coords']}
        for _, row in df.iterrows()
    ]

location_records = load_location_records()

# ----------------------------------------
# Streamlit UI Controls
# ----------------------------------------
st.sidebar.title("Rainfall Fetch Controls")
start_date = st.sidebar.date_input("Start Date", datetime.date(2024, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date(2024, 3, 31))
run_fetch = st.sidebar.button("Fetch Rainfall")

st.title("Rainfall Predictor for ASDF Wells")
st.markdown(f"222 unique wells loaded. Date range: **{start_date}** to **{end_date}**.")

# ----------------------------------------
# Fetch CHIRPS rainfall data
# ----------------------------------------
def fetch_rainfall_for_record(record, start_date, end_date):
    try:
        point = ee.Geometry.Point(record['coords'])
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
        data = [{'date': f['properties']['date'], 'rainfall': f['properties']['precipitation']} 
                for f in features['features']]

        df = pd.DataFrame(data)
        df['rainfall'] = pd.to_numeric(df['rainfall'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'])
        df['twp_id'] = record['twp_id']
        df['name'] = record['name']
        df['rolling_7d_rainfall'] = df['rainfall'].rolling(7).sum()
        return df
    except Exception as e:
        print(f"Error fetching data for {record['name']}: {e}")
        return None

# ----------------------------------------
# Run fetch when user clicks button
# ----------------------------------------
if run_fetch:
    st.info("Fetching rainfall data. Please wait...")
    all_data = []
    for i, record in enumerate(location_records):
        st.write(f"Fetching {i+1} of {len(location_records)}: {record['name']}")
        df = fetch_rainfall_for_record(record, start_date, end_date)
        if df is not None:
            all_data.append(df)

    if all_data:
        full_df = pd.concat(all_data)
        st.success("Rainfall data fetch complete!")
        st.dataframe(full_df.head())

        selected_well = st.selectbox("Select a well to visualize", full_df['name'].unique())
        chart_df = full_df[full_df['name'] == selected_well].set_index('date')
        st.line_chart(chart_df[['rainfall', 'rolling_7d_rainfall']])

        csv = full_df.to_csv(index=False)
        st.download_button("Download CSV", csv, "rainfall_data.csv")
    else:
        st.error("No data fetched. Check logs or try again.")

