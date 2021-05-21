#!/bin/bash

declare -a timeins
declare -a timeouts
declare -a archs

timeins=(100,200)
timeouts=(10,20,50,100)
archs=(1,2,3)
for timeout in 20 50 100
do
for arch in 1 2 3
do 
     python3 correct-ms-mc.py "100" $timeout $arch
done
done

for timeout in 10 20 50 100
do
for arch in 1 2 3
do 
     python3 correct-ms-mc.py "200" $timeout $arch
done
done