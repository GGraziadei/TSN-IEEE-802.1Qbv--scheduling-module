#!/bin/bash
echo "Validation script..."

# for all instances in input
for var in $@
do
    echo "$var Instance started"
    python3 validation_heuristic.py -i "instance_generator/instances/$var/" &

done

# Wait for all tasks to complete
wait
echo "All tasks completed."