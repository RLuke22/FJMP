# build docker image
docker build -t fjmp .
# run docker container
docker run --rm -it -u root -v [path/to/FJMP]:/FJMP --gpus=all --name fjmp_docker fjmp