import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import defaultdict, deque
from datetime import datetime
import threading
from azure.cosmos import CosmosClient
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from joblib import load
import pandas as pd
import warnings
import tensorflow as tf

warnings.filterwarnings("ignore", module='sklearn')

url = {url}
key = {key}
database_name = {db_name}
container_name = {container_name}



client = CosmosClient(url, credential=key)
database = client.get_database_client(database_name)
container = database.get_container_client(container_name)

print(f"TensorFlow version: {tf.__version__}")
custom_objects = {'mse': MeanSquaredError()}

try:
    humidity_model = load_model('autoencoder_model_humidity.h5', custom_objects=custom_objects)
    temp_model = load_model('autoencoder_model_temp.h5', custom_objects=custom_objects)
    print("Autoencoder models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

try:
    scaler_humidity = load('scaler_humidity.joblib')
    scaler_temp = load('scaler_temp.joblib')
    print("MinMaxScalers loaded successfully.")
    if not isinstance(scaler_humidity, MinMaxScaler) or not hasattr(scaler_humidity, 'data_min_'):
        raise TypeError("Loaded humidity scaler is not a fitted MinMaxScaler instance.")
    if not isinstance(scaler_temp, MinMaxScaler) or not hasattr(scaler_temp, 'data_max_'):
        raise TypeError("Loaded temperature scaler is not a fitted MinMaxScaler instance.")
    print(f"Humidity Scaler: Min={scaler_humidity.data_min_}, Max={scaler_humidity.data_max_}")
    print(f"Temperature Scaler: Min={scaler_temp.data_min_}, Max={scaler_temp.data_max_}")
except Exception as e:
    print(f"Error loading scalers: {e}")
    exit()

# Configuration
humidity_threshold = 0.342
temp_threshold = 0.1221
MAX_POINTS = 100  # Maximum points to display on each graph

sensor_data = defaultdict(lambda: {
    'timestamps': deque(maxlen=MAX_POINTS),
    'humidity_values': deque(maxlen=MAX_POINTS),
    'temp_values': deque(maxlen=MAX_POINTS),
    'humidity_anomalies': deque(maxlen=MAX_POINTS),
    'temp_anomalies': deque(maxlen=MAX_POINTS),
    'humidity_errors': deque(maxlen=MAX_POINTS),
    'temp_errors': deque(maxlen=MAX_POINTS)
})

data_lock = threading.Lock()
last_item = None

def get_sensor_id(item):

    possible_sensor_keys = ['uniqueid']
    
    for key in possible_sensor_keys:
        if key in item:
            return str(item[key])
  
    if 'device_name' in item:
        return str(item['device_name'])
    
    return f"sensor_{hash(str(item.get('_rid', 'unknown'))) % 1000}"

def data_collection_thread():

    global last_item
    
    while True:
        try:
            query = "SELECT TOP 1 * FROM c ORDER BY c.received_dttime DESC"
            items = list(container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            
            if items:
                latest_item = items[0]
                
                if last_item != latest_item:
                    print("New or Updated Document:")
                    print(latest_item)
                    
             
                    sensor_id = get_sensor_id(latest_item)
                    
                    # Extract data
                    possible_temp_keys = ['Temperature', 'temperature', 'temp']
                    possible_humidity_keys = ['Humidity', 'humidity']

                    humidity_data = None
                    for key in possible_humidity_keys:
                        if key in latest_item:
                            humidity_data = latest_item.get(key)
                            break

                    temp_data = None
                    for key in possible_temp_keys:
                        if key in latest_item:
                            temp_data = latest_item.get(key)
                            break
                            
                    timestamp = latest_item.get('received_dttime')
                    
                    if humidity_data is not None and temp_data is not None:
                        print(f"  Sensor {sensor_id}: Humidity={humidity_data}, Temp={temp_data}")
                        
                        # Process with models
                        humidity_data_scaled = scaler_humidity.transform(np.array([[humidity_data]]))
                        temp_data_scaled = scaler_temp.transform(np.array([[temp_data]]))

                        humidity_reconstruction = humidity_model.predict(humidity_data_scaled.reshape(1, 1))[0][0]
                        temp_reconstruction = temp_model.predict(temp_data_scaled.reshape(1, 1))[0][0]
                        
                        humidity_error = np.square(humidity_data_scaled[0][0] - humidity_reconstruction)
                        temp_error = np.square(temp_data_scaled[0][0] - temp_reconstruction)

                        is_humidity_anomaly = humidity_error > humidity_threshold
                        is_temp_anomaly = temp_error > temp_threshold

                        # Store data thread-safely
                        with data_lock:
                            sensor_data[sensor_id]['timestamps'].append(datetime.now())
                            sensor_data[sensor_id]['humidity_values'].append(humidity_data)
                            sensor_data[sensor_id]['temp_values'].append(temp_data)
                            sensor_data[sensor_id]['humidity_anomalies'].append(is_humidity_anomaly)
                            sensor_data[sensor_id]['temp_anomalies'].append(is_temp_anomaly)
                            sensor_data[sensor_id]['humidity_errors'].append(humidity_error)
                            sensor_data[sensor_id]['temp_errors'].append(temp_error)

                        print(f"  Reconstruction Errors: Humidity={humidity_error:.4f}, Temp={temp_error:.4f}")
                        
                        if is_humidity_anomaly:
                            print(f"  ALERT: Humidity anomaly detected! Error ({humidity_error:.4f}) > Threshold ({humidity_threshold}) at {timestamp}")
                        if is_temp_anomaly:
                            print(f"  ALERT: Temperature anomaly detected! Error ({temp_error:.4f}) > Threshold ({temp_threshold}) at {timestamp}")
                    
                    last_item = latest_item
                    
        except Exception as e:
            print(f"Error in data collection: {e}")
        
        time.sleep(10)

def create_dashboard():
  
    plt.style.use('dark_background')
    ax_map={}
    def update_plots(frame):
        with data_lock:

            current_sensors = list(sensor_data.keys())
            
            if not current_sensors:
                return
       
            fig.clear()
            ax_map.clear()
            
      
            n_sensors = len(current_sensors)
            cols = min(3, n_sensors)  # Max 3 columns
            rows = (n_sensors + cols - 1) // cols
            
            # Create subplots for each sensor 
            for i, sensor_id in enumerate(current_sensors):
                data = sensor_data[sensor_id]
                
                if not data['timestamps']:
                    continue
                
                # Convert to lists for plotting
                times = list(data['timestamps'])
                humidity_vals = list(data['humidity_values'])
                temp_vals = list(data['temp_values'])
                humidity_anomalies = list(data['humidity_anomalies'])
                temp_anomalies = list(data['temp_anomalies'])
                
                # Humidity subplot
                ax1 = fig.add_subplot(rows * 2, cols, i * 2 + 1)
                ax2 = fig.add_subplot(rows * 2, cols, i * 2 + 2)
                # Plot normal points in blue, anomalies in red
                normal_times = [t for t, is_anom in zip(times, humidity_anomalies) if not is_anom]
                normal_vals = [v for v, is_anom in zip(humidity_vals, humidity_anomalies) if not is_anom]
                anomaly_times = [t for t, is_anom in zip(times, humidity_anomalies) if is_anom]
                anomaly_vals = [v for v, is_anom in zip(humidity_vals, humidity_anomalies) if is_anom]
                
                if normal_times:
                    ax1.plot(normal_times, normal_vals, 'b-', alpha=0.7, linewidth=1)
                    ax1.scatter(normal_times, normal_vals, c='blue', s=20, alpha=0.8)
                
                if anomaly_times:
                    ax1.scatter(anomaly_times, anomaly_vals, c='red', s=40, alpha=0.9, marker='x')
                
                ax1.set_title(f'Sensor {sensor_id} - Humidity', fontsize=10, color='white')
                ax1.set_ylabel('Humidity (%)', fontsize=8)
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(axis='x', rotation=45, labelsize=6)
                ax1.tick_params(axis='y', labelsize=6)
                
                # Temperature subplot
                ax2 = fig.add_subplot(rows * 2, cols, i * 2 + 2)
                
                normal_times = [t for t, is_anom in zip(times, temp_anomalies) if not is_anom]
                normal_vals = [v for v, is_anom in zip(temp_vals, temp_anomalies) if not is_anom]
                anomaly_times = [t for t, is_anom in zip(times, temp_anomalies) if is_anom]
                anomaly_vals = [v for v, is_anom in zip(temp_vals, temp_anomalies) if is_anom]
                
                if normal_times:
                    ax2.plot(normal_times, normal_vals, 'g-', alpha=0.7, linewidth=1)
                    ax2.scatter(normal_times, normal_vals, c='green', s=20, alpha=0.8)
                
                if anomaly_times:
                    ax2.scatter(anomaly_times, anomaly_vals, c='red', s=40, alpha=0.9, marker='x')
                
                ax2.set_title(f'Sensor {sensor_id} - Temperature', fontsize=10, color='white')
                ax2.set_ylabel('Temperature (°C)', fontsize=8)
                ax2.set_xlabel('Time', fontsize=8)
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(axis='x', rotation=45, labelsize=6)
                ax2.tick_params(axis='y', labelsize=6)

                ax_map[ax1] = (sensor_id, 'humidity')
                ax_map[ax2] = (sensor_id, 'temperature')
        
        plt.tight_layout()

    # Create the figure
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Live Sensor Anomaly Detection Dashboard', fontsize=16, color='white')
    def on_click(event):
        if event.inaxes in ax_map:
            sensor_id, data_type = ax_map[event.inaxes]
            with data_lock:
                data = sensor_data[sensor_id]
                times = list(data['timestamps'])
                values = list(data['humidity_values'] if data_type == 'humidity' else data['temp_values'])
                anomalies = list(data['humidity_anomalies'] if data_type == 'humidity' else data['temp_anomalies'])

            normal_times = [t for t, is_anom in zip(times, anomalies) if not is_anom]
            normal_vals = [v for v, is_anom in zip(values, anomalies) if not is_anom]
            anomaly_times = [t for t, is_anom in zip(times, anomalies) if is_anom]
            anomaly_vals = [v for v, is_anom in zip(values, anomalies) if is_anom]

            # Open new figure
            detail_fig, detail_ax = plt.subplots(figsize=(10, 5))
            detail_ax.plot(normal_times, normal_vals, 'b-' if data_type == 'humidity' else 'g-', alpha=0.7, linewidth=1)
            detail_ax.scatter(normal_times, normal_vals, c='blue' if data_type == 'humidity' else 'green', s=20)
            detail_ax.scatter(anomaly_times, anomaly_vals, c='red', s=40, marker='x')

            detail_ax.set_title(f'Expanded View: {sensor_id} - {data_type.capitalize()}')
            detail_ax.set_xlabel('Time')
            detail_ax.set_ylabel('Humidity (%)' if data_type == 'humidity' else 'Temperature (°C)')
            detail_ax.grid(True)
            detail_ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.show()

    fig.canvas.mpl_connect('button_press_event', on_click)

    ani = FuncAnimation(fig, update_plots, interval=5000, cache_frame_data=False)
    return fig, ani  
    # Start animation
    

if __name__ == "__main__":

    data_thread = threading.Thread(target=data_collection_thread, daemon=True)
    data_thread.start()
    
    print("Starting live dashboard...")
    print("Data collection thread started. Waiting for data...")
 
    time.sleep(15)
    
    # Create and show dashboard
    fig, ani = create_dashboard()
    plt.show()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Dashboard stopped by user")
