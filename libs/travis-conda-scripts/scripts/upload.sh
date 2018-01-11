if [ -z "$1" ]; then
    echo "ERROR: No channel provided"
    echo "Usage: upload.sh <channel> [label]"
    exit
fi
if [ $TRAVIS_TEST_RESULT -eq 0 ]; then
    cd packages
    LABEL=${2:-main}

    for TARBALL in $BUILD_OUTPUT/{noarch}-*/*.tar.bz2; do
        echo "Uploading $TARBALL to anaconda with anaconda upload..."
        set +x # hide token
        anaconda -t "$CONDA_UPLOAD_TOKEN" upload -u "$1" --force "$TARBALL" -l "$LABEL"
        set -x
    done
    cd ..
    echo "Upload command complete!"
else
    echo "Upload cancelled due to failed test."
fi
