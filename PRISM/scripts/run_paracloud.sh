#!/bin/bash

if [ ! -f "$1" ]; then
	echo "File $1 does not exists."
fi

echo "Submitting $1"
"$1"
