# Trotro MLops


An MLOPS for predicting when a taxi will arrive at a trip destination.


## The dataset 

Downloaded from New York City tax and limousine Commision. We use the [January 2024 data](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet) for training and [February 2024 data](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet) for validating the model. 



# Modeling

Comparing the prediction of an xgboot regressor on a training and validation set

| Training dataset prediction | Validation dataset prediction |
|----------|----------|
| ![Training set prediction](./modeling/img/train_dist.png)  | ![Training set prediction](./modeling/img/val_dist.png) |

# Experiment tracking with MlFlow


