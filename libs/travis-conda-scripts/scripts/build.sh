mkdir -p "$CONDA_BLD_PATH"
conda config --set anaconda_upload no
conda build . $EXTRA_CONDA_CHANNELS --python "$PACKAGE" --dirty
