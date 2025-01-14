#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "You must enter exactly 2 argument (interface and log filename)"
    exit 1
fi

interface=$1
log_filename=$2


sudo bmon -p $interface -o format:fmt='$(element:name) $(attr:rxrate:bytes) $(attr:txrate:bytes) $(attr:rxrate:packets) $(attr:txrate:packets) \n' | while read line; do echo "`date +%s` $line"; done &>> $log_filename
