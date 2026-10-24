NB ?= notebooks/cadence_hvac_dqn_ppo_v2.ipynb

.PHONY: build lab run shell clean

build:
	docker compose build

lab:
	docker compose run --rm --service-ports sinergym jupyter lab --ip=0.0.0.0 --no-browser --allow-root

# Training runs for hours: no per-cell timeout, executed outputs are saved back into the notebook
run:
	docker compose run --rm sinergym jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=-1 \
		--ExecutePreprocessor.startup_timeout=300 $(NB)

shell:
	docker compose run --rm sinergym bash

clean:
	docker compose run --rm sinergym find /workspace -type d -name 'Eplus-*-res*' -exec rm -rf {} +
