# Modified version to parallelize portabilizing

# Get the directory of the current script
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)

for dir in */; do
    {
        echo "Current directory: $dir"
        cd $dir
        mkdir -p bin
        cp raw/modules.log bin/modules.log
        python2 "${SCRIPT_DIR}/portabilize_trace.py" .
        cp bin/modules.log raw/modules.log
        ${RESOURCES_DIR}/scarab/src/build/opt/deps/dynamorio/clients/bin64/drraw2trace -indir ./raw/
        rm -rf ./raw
        cd -
    } &
done
wait
