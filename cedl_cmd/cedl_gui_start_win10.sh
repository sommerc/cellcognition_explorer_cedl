#!/bin/bash
cd "/c/Program Files (x86)/Xming/"
"/c/Program Files (x86)/Xming/Xming.exe" :0 -clipboard -multiwindow -ac &

export DISPLAY=`ipconfig | grep -m 1 IPv4 | grep -oE '((1?[0-9][0-9]?|2[0-4][0-9]|25[0-5])\.){3}(1?[0-9][0-9]?|2[0-4][0-9]|25[0-5])'`
winpty docker run -e DISPLAY=$DISPLAY:0 --rm -it sommerc/cedl-gui


