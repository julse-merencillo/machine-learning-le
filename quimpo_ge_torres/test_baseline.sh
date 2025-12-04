#!/usr/bin/env bash

for i in {1..50}; do
	echo "Running iteration $i"
	sumo -c quimpo_blvd.sumocfg --tripinfo-output baseline_trip_$i.xml --seed $i
done
