
lint:
	black .

clean:
	rm -rf build dist trotro.egg-info __pycache__

all: lint clean
