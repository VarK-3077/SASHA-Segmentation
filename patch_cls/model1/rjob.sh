#!/bin/bash
# Launch a detached job on the server and drop a sentinel when it exits, so the caller can
# poll for completion by file rather than by pgrep (which matches its own ssh wrapper).
#   ./rjob.sh <tag> <remote command...>
set -e
TAG=$1; shift
ssh -n vivi "rm -f ~/logs/$TAG.done; setsid bash -c 'cd ~/SASHA-Segmentation/patch_cls/model1 && $* ; echo \$? > ~/logs/$TAG.done' > ~/logs/$TAG.log 2>&1 < /dev/null &"
echo "launched $TAG"
