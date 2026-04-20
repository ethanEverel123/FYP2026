import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns


scaler = None
X_scaled = None
benign_clusters = None
dbscan_model = None
dbscan_labels = None
data_read = False
clustered = False


def read_data():
    global scaler, X_scaled, data_read
    file_to_read = input("please enter the file you want to read")
    #validation
    if(file_to_read[-4:] != ".csv"):
        print("invalid file format")
        return None, None
    
    
    print(f"reading: {file_to_read}")
    df = pd.read_csv(f"{file_to_read}")
    df.columns = df.columns.str.strip()
    

    df.head(20)
    #cleaning
    df = df.dropna()  # Remove missing values
    X = df.select_dtypes(include=['float64', 'int64'])  # Select numeric columns
    X = X.drop(columns=['Flow ID', 'Source IP', 'Destination IP', 'Timestamp'], errors='ignore')
    

    # Replace infinity values with NaN and drop rows with NaN
    X.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
    X.dropna(inplace=True)

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #data samplling

    # Sample 10% of data for faster computation
    sample_size = min(10000, len(X_scaled))  # Use max 10k samples
    sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)

    df_sample = df.iloc[sample_indices].copy()
    X_sample = X_scaled[sample_indices]   
    
    data_read = True
    return X_sample, df_sample
        
def cluster(X_sample):
    global dbscan_labels, benign_clusters, clustered
    dbscan = DBSCAN(eps=1, min_samples=10)
    dbscan_labels = dbscan.fit_predict(X_sample)

    score = silhouette_score(X_sample, dbscan_labels)
    print("Silhouette Score (DBSCAN):", score)
    
    df_sample['Cluster_DBSCAN'] = dbscan_labels
    df_sample['Label'] = df_sample['Label'].astype('category')
    
    
    cluster_purity = (
        df_sample.groupby('Cluster_DBSCAN')['Label']
        .apply(lambda x: (x == 'BENIGN').mean())
        .rename('benign_purity')
    )
    
    benign_clusters = cluster_purity[cluster_purity >= 0.60].index
    clustered = True
    
def visualise(df_sample):
    global benign_clusters, dbscan_labels
    
    df_sample['Cluster_DBSCAN'] = dbscan_labels
    df_sample['Label'] = df_sample['Label'].astype('category')

    ct = pd.crosstab(df_sample['Cluster_DBSCAN'], df_sample['Label'])
    print(ct)
    ct_norm = pd.crosstab(df_sample['Cluster_DBSCAN'], df_sample['Label'], normalize='index').round(3)
    print(ct_norm)


    print(f"\nClusters classed as Normal (≥60% BENIGN): {list(benign_clusters)}")

    #map each sample to predicted Normal/Anomaly 
    df_sample['y_pred'] = np.where(
        df_sample['Cluster_DBSCAN'].isin(benign_clusters), 
        'Normal', 
        'Anomaly'
    )

    #get if data is normal or Anomaly 
    df_sample['y_true'] = np.where(
        df_sample['Label'] == 'BENIGN', 
        'Normal', 
        'Anomaly'
    )

    #Confusion matrix: 
    labels_order = ['Normal', 'Anomaly']
    cm = confusion_matrix(df_sample['y_true'], df_sample['y_pred'], labels=labels_order)

    ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        cmap='Blues',
        xticklabels=labels_order,
        yticklabels=labels_order,
        ax=ax
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('DBSCAN Anomaly Detection — Confusion Matrix\n(cluster ≥50% BENIGN → Normal)')
    plt.tight_layout()
    plt.show() 
    


    # show normal clusters
    print(classification_report(df_sample['y_true'], df_sample['y_pred'], target_names=labels_order))
    
def guess(raw_input: str):
    global scaler, X_scaled, benign_clusters, dbscan_labels

    try:
        values = [float(v.strip()) for v in raw_input.split(",")]
    except ValueError:
        print(f"Could not parse input. Expected {len(dbscan_labels)} comma-separated numbers.")
        return
    
    #scale using the fitted scaler
    new_point = np.array(values).reshape(1, -1)
    new_point_scaled = scaler.transform(new_point)


    #get the nearest neighbour to inputed data
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X_scaled[:len(dbscan_labels)])
    distance, idx = nn.kneighbors(new_point_scaled)
    
    assigned_cluster = dbscan_labels[idx[0][0]]
    nearest_distance = distance[0][0]


    if assigned_cluster == -1:
        # noise point
        verdict = "Anomaly (nearest neighbour was a noise point)"
    elif assigned_cluster in benign_clusters:
        verdict = "Normal"
    else:
        verdict = "Anomaly"

    print(f"\nGuess Result")
    print(f"  Assigned cluster : {assigned_cluster}")
    print(f"  Distance to NN   : {nearest_distance}")
    print(f"  Verdict          : {verdict}")
 
 

while True:

    
    print("what would you like to do?")
    print("1. read data")
    print("2. cluster data")
    print("3. visualise data")
    print("4. guess new data")
    print("5. exit")
    choice = input("Enter your choice: ")
    
    if choice == "1":            
        X_sample, df_sample = read_data()
        
    elif choice == "2":
        if(data_read == True):
            if('X_sample' in globals()):  
                print("clustering data")
                cluster(X_sample)     
        else:
            print("data hasn't been read")

            
    elif choice == "3":   
        if(data_read == True and clustered == True):
            if('df_sample' in globals()):
                print("visualising data")
                visualise(df_sample)
        else:
            print("Error")     

        
    elif choice == "4":
        if('X_sample' in globals() and  clustered == True):
            raw = input("Data: ")
            guess(raw)
 
        else:
            print("clustering not run")
        
    elif choice == "5":               
        print("exiting") 
        break
    
    else:
        print("invalid choice")