install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

lint:
	@echo "#### Formatting and Linting... ####"
	isort .
	black .
	pylint  --disable=R,C .

test:
	@echo "#### Testing... ####"
	pytest orchestration

clean:
	@echo "#### Cleaning up... ####"
	rm -rf build dist trotro.egg-info __pycache__

start_mlflow_server:
	@echo "#### Starting MLFlow server... ####"
	mlflow server --backend-store-uri sqlite:///./data/mlflow.db --host 127.0.0.1 --port 8080

build_deployment_container:
	@echo "#### Building deployment container image ####"
	docker build -t trotro-duration-prediction-service:v1 ./deployment

run_deployment_container: build_deployment_container
	@echo "#### Running deployment container image ####"
	docker run -d --rm -p 9696:9696 trotro-duration-prediction-service:v1


pre-commit: clean test lint
	git add .

all:  clean install test lint
