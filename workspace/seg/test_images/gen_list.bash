#!/bin/bash

rm list.txt

for jpg in `ls *.jpg`
do
    echo "${jpg}" >> list.txt
done

for jpeg in `ls *.jpeg`
do
    echo "${jpeg}" >> list.txt
done

for png in `ls *.png`
do
    echo "${png}" >> list.txt
done

