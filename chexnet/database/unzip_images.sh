#!/bin/bash

for i in {1..9}
do
	mkdir images_00${i}
	tar -xvzf images_0${i}.tar.gz -C images_00${i}
	mv images_00${i}/images/* images_00${i}/.

done

for i in {10..12}
do
	mkdir images_0${i}
	tar -xvzf images_${i}.tar.gz -C images_0${i}
	mv images_0${i}/images/* images_0${i}/.
done


