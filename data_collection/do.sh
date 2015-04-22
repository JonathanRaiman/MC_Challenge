#!/bin/sh
#!/bin/bash

# Stop on error
set -e
# Stop when undefined variable is ecountered
set -u
# Easier to debug errors
set -o pipefail

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

action=${1:-}
if [[ -z "$action" ]]
then
    echo "Usage $0 [decrypt|encrypt]"
    exit 1
fi
pushd $SCRIPT_DIR > /dev/null 2>&1
if [[ "$action" == "encrypt" ]]
then
    tar -c secret | gpg --symmetric > private.gpg
elif [[ "$action" == "decrypt" ]]
then
    gpg --decrypt private.gpg | tar xf -
else
    echo "Usage $0 [decrypt|encrypt]"
    exit 1
fi
popd > /dev/null 2>&1

