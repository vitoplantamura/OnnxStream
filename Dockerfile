#
# Command to build the image:
#   sudo docker build -t sd .
#
# Example command to start the container (weights and result are saved in the current directory):
#   sudo docker run -t -i --init -v .:/current sd --models-path /current --output /current/result.png --steps 1 --turbo
#

ARG UBUNTU_VERSION=22.04
FROM ubuntu:$UBUNTU_VERSION AS build
RUN apt-get update && apt-get install -y build-essential git cmake python3
WORKDIR OnnxStream
COPY src .
RUN mkdir build \
    && cd build \
    && cmake .. \
    && cmake --build . --config Release
FROM ubuntu:$UBUNTU_VERSION AS runtime
RUN apt-get update && apt-get install -y curl
COPY --from=build /OnnxStream/build/sd /sd
ENTRYPOINT [ "/sd" ]
