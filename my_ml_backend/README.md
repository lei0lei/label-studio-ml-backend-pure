This guide describes the simplest way to start using ML backend with Label Studio.

## Running with Docker (Recommended)

1. Start Machine Learning backend on `http://localhost:9090` with prebuilt image:

```bash
docker-compose up
```

2. Validate that backend is running

```bash
$ curl http://localhost:9090/
{"status":"UP"}
```

3. Connect to the backend from Label Studio running on the same host: go to your project `Settings -> Machine Learning -> Add Model` and specify `http://localhost:9090` as a URL.


## Building from source (Advanced)

To build the ML backend from source, you have to clone the repository and build the Docker image:

```bash
docker-compose build
```

## Running without Docker (Advanced)

To run the ML backend without Docker, you have to clone the repository and install all dependencies using pip:

```bash
python -m venv ml-backend
source ml-backend/bin/activate
pip install -r requirements.txt
```

Then you can start the ML backend:

```bash
label-studio-ml start ./dir_with_your_model
```

# Configuration
Parameters can be set in `docker-compose.yml` before running the container.

## Model routing parameters

You can route inference by passing these parameters in ML backend requests:

- `model_family`: model family, e.g. `yolov8`, `yolo11`, `yolo26`, `sam2`
- `model_task`: task type, e.g. `detect`, `segment`, `obb`
- `model_name`: model file name without extension, e.g. `best`, `obb_640`

### SAM2 integration

- Use `model_family=sam2` and `model_task=segment`.
- The backend uses the SAM backend path and does **not** force YOLO `imgsz` arguments.
- If Label Studio interaction context provides `keypointlabels` and/or `rectanglelabels`, those prompts are forwarded to SAM as points/boxes.
- If no prompts are provided, SAM runs segmentation with default model behavior.

Example request params:

```json
{
	"model_family": "sam2",
	"model_task": "segment",
	"model_name": "best"
}
```


The following common parameters are available:
- `BASIC_AUTH_USER` - specify the basic auth user for the model server
- `BASIC_AUTH_PASS` - specify the basic auth password for the model server
- `LOG_LEVEL` - set the log level for the model server
- `WORKERS` - specify the number of workers for the model server
- `THREADS` - specify the number of threads for the model server

# Customization

The ML backend can be customized by adding your own models and logic inside the `./dir_with_your_model` directory. 