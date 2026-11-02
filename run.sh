#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

if [ -d "/Applications/GigaDictate.app" ]; then
    open /Applications/GigaDictate.app
elif [ -d "$DIR/GigaDictate.app" ]; then
    open "$DIR/GigaDictate.app"
else
    echo "GigaDictate.app not found. Running build script first..."
    bash "$DIR/scripts/create_app.sh"
    open /Applications/GigaDictate.app
fi
