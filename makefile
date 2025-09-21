.PHONY: build-cpu build-gpu run-cpu run-gpu jupyter test clean

build-cpu:
	docker compose build biomedical-cpu

build-gpu:
	docker compose build biomedical-gpu

run-cpu:
	docker compose run --rm biomedical-cpu

run-gpu:
	docker compose run --rm biomedical-gpu

jupyter:
	docker compose up jupyter

test:
	docker compose run --rm biomedical-cpu python -m pytest tests/ -v

clean:
	docker compose down --rmi all --volumes
