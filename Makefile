install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

lint:
	@echo "#### Formatting and Linting... ####"
	isort orchestration
	black .
	pylint  --disable=R,C orchestration

test:
	@echo "#### Testing... ####"
	pytest orchestration

clean:
	@echo "#### Cleaning up... ####"
	rm -rf build dist trotro.egg-info __pycache__ 


start_mlflow_server:
	@echo "#### Starting MLFlow server... ####"
	mlflow server --backend-store-uri sqlite:///./data/mlflow.db --host 127.0.0.1 --port 8080

all:  clean test lint
