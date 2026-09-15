NB ?= notebooks/hello_world.ipynb

.PHONY: lab verify shell clean

lab:
	docker compose run --rm --service-ports sinergym jupyter lab --ip=0.0.0.0 --no-browser --allow-root

verify:
	docker compose run --rm sinergym jupyter nbconvert --to notebook --execute --inplace $(NB)

shell:
	docker compose run --rm sinergym bash

clean:
	docker compose run --rm sinergym find /workspace -type d -name 'Eplus-*-res*' -exec rm -rf {} +
