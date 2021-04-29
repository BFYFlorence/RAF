#!/bin/bash

ls rep.c* | while read line
do
stride ./${line} >>./SASA/sasa_${line}
done