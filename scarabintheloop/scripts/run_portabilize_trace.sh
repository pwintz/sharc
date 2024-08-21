# Modified version to parallelize portabilizing. 
# This must be run within 

# Get the directory of the current script
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)

for dir in */; do
    {
        echo "Current directory: $dir"
        cd $dir
        mkdir -p bin
        cp raw/modules.log bin/modules.log
        portabilize_trace.py .
        cp bin/modules.log raw/modules.log
        ${SCARAB_ROOT}/src/build/opt/deps/dynamorio/clients/bin64/drraw2trace -indir ./raw/
        rm -rf ./raw
        cd -
    } &
done
wait
