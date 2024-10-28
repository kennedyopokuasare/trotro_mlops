
lint:
	black .

clean:
	rm -rf build dist trotro.egg-info __pycache__

mflow_server:
	mlflow server --backend-store-uri sqlite:///mlflow.db

all: lint clean
