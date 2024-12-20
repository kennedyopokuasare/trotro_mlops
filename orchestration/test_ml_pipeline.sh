export PYTHONPATH=$(dirname "$(pwd)"):$PYTHONPATH

echo "Testing prepare data pipeline" 

python scripts/prepare_data.py \
    --train_data_path ../data/yellow_tripdata_2024-01.parquet \
    --validation_data_path ../data/yellow_tripdata_2024-02.parquet \
    --features_train_path ../data \
    --features_validation_path ../data

echo "Testing train pipeline"

python scripts/train_model.py\
    --train_data_path ../data \
    --validation_data_path ../data \
    --scoring_data ../data/yellow_tripdata_2024-03.parquet