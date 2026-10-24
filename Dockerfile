FROM sailugr/sinergym:v3.12.2

RUN pip install --no-cache-dir jupyterlab ipykernel matplotlib pyarrow tensorboard

WORKDIR /workspace
