#!/bin/sh

USER=mgrabner,cudatemplates
HOST=shell.sourceforge.net
HTDOCS=/home/groups/c/cu/cudatemplates/htdocs

# create:
#ssh -t $USER@$HOST create

# copy:
rsync --verbose --recursive --delete --rsh="ssh -l $USER" html $USER@$HOST:$HTDOCS/doc
