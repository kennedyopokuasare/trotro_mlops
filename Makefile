
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

commit_changes:
	@echo "#### saving lint changes"
	git commit -a -m "Automated lint"
mflow_server:
	@echo "#### Starting MLFlow server... ####"
	mlflow server --backend-store-uri sqlite:///mlflow.db

all: clean test lint
