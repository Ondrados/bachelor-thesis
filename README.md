# Cell detection using convolutional neural networks

## Instalation

1. clone this repo and cd to it
    ```
    git clone https://github.com/Ondrados/cnn-cells.git
    cd cnn-cells
    ```

2. create virtual enviroment and activate it
    ```
    virtualenv env
    source env/bin/activate
    ```

3. install dependencies
    ```
    pip install -r requirements_tiny.txt
    ```

4. download weights from releases and extract it inside cnn-cells folder
    ```
    wget https://github.com/Ondrados/cnn-cells/releases/download/v1.0/models.zip
    unzip models.zip
    ```

5. export python path for this repository to your enviroment
    ```
    export PYTHONPATH="${PYTHONPATH}:/path/to/cnn-cells"
    ```
6. run demo script which detect cells on 5 images
    ```
    python demo.py
    ```