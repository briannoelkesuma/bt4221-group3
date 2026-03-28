import pandas as pd

def download_data():
    url1 = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
    url2 = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet"
    print("Downloading Jan data...")
    df1 = pd.read_parquet(url1)
    df1.head(1000).to_csv("taxi_jan.csv", index=False)
    
    print("Downloading Feb data...")
    df2 = pd.read_parquet(url2)
    df2.head(1000).to_csv("taxi_feb.csv", index=False)
    
    print("Done generating mock data!")

if __name__ == "__main__":
    download_data()
