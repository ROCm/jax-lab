# JAX-MAD

This folder uses [MadEngine](https://github.com/ROCm/madengine) to build and run the model sources defined in `models.json`.

## Usage

You can use the jax-mad CI workflow or, for a local run from this directory:

```bash
git clone https://github.com/ROCm/jax-lab.git && cd jax-lab/jax-mad
git clone --depth 1 https://github.com/ROCm/madengine.git
python3 -m venv myenv
source myenv/bin/activate
cd madengine && pip install -e .
cd ..

madengine run --tags <model_name> --live-output \
  --additional-context "{\"docker_build_arg\":{\"BASE_DOCKER\":\"<image>\"}}" \
  --keep-alive