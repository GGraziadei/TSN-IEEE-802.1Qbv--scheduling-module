#!/bin/bash

echo "Update the repository..."
git pull

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

for var in $@
do
    echo "$var Instance Post-processing"
    python3 instance_generator/stats_to_csv.py -f "instance_generator/instances/$var/out_heuristic"
    python3 instance_generator/generate_graph.py -f "instance_generator/instances/$var/out_heuristic"
done



echo "Push results to the repository..."
git add *
git commit -m "Validation results"
git push
