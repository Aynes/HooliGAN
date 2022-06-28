FROM ubuntu:20.04

WORKDIR /usr/src/app
RUN apt-get update -y && apt-get upgrade -y && apt-get install -y --no-install-recommends python3 python3-dev python3-pip git 

ENTRYPOINT /bin/bash