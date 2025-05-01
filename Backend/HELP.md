#Cloud-AI-Backend

#### This is the latest Backend version.

## Configuration
to configure please edit the entries in the application-prod.yml file.

## Build an Image from the Docker File
`docker build -t NAME:TAG .` e.g. `docker build -t cloud-ai-api:1.0.0 .`

## Build a container from the docker image

`docker run -d -p 9090:9090 --name CONTAINER_NAME CONTAINER_IAMAGE:CONTAINER_TG` e.g.
`docker run -d -p 9090:9090 --name cloud-ai-api cloud-ai-api:1.0.0`

## Env
SPRING_PROFILE=prod