#!/bin/sh

USER=mgrabner,cudatemplates
HOST=shell.sourceforge.net
HTDOCS=/home/groups/c/cu/cudatemplates/htdocs

rsync --verbose --recursive --rsh="ssh -l $USER" html $USER@$HOST:$HTDOCS/doc
