#!/bin/bash
while true
do

i=1
first=first
second=second

while read -r line
do
s=$(echo "${line}" | cut -d ' ' -f 1)

if [ ${i} -eq 1 ]
then
first=${s}
fi

if [ ${i} -eq 2 ]
then
second=${s}
fi

i=$((i + 1))

if [ ${i} -eq 3 ]
then
break
fi
done <<< "$(tail text.log -n 13)"

if [ "${first}" = "Step" ]
then
echo "${second}"

fi


sleep 5
done