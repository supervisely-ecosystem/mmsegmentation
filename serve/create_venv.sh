#!/bin/bash

# learn more in documentation
# Official python docs: https://docs.python.org/3/library/venv.html
# Superviely developer portal: https://developer.supervisely.com/getting-started/installation#venv

if [ -d ".venv" ]; then
    echo "VENV already exists, will be removed"
    rm -rf .venv
fi

echo "VENV will be created" && \
python3 -m venv .venv && \
source .venv/bin/activate && \

echo "Install requirements..." && \
pip3 install -r serve/dev_requirements.txt && \

arch=$(uname -m)
ARCHFLAGS="-arch x86_64" 
if [[ $arch == arm* ]]; then
  echo "------> Running on MAC ..."
fi

echo "Requirements have been successfully installed" && \

echo "Testing imports, please wait a minute ..." && \
python -c "import supervisely as sly" && \
echo "Success!" && \

deactivate
