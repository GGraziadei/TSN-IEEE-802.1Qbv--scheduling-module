#!/bin/bash


for var in $@
do
    echo "$var Instance Post-processing"
    python3 instance_generator/stats_to_csv.py -f "instance_generator/instances/$var/out_heuristic" >> "instance_generator/instances/$var/stats.txt"
    python3 instance_generator/generate_graph.py -f "instance_generator/instances/$var/out_heuristic"
    python3 instance_generator/traffic_analyzer.py -f "instance_generator/instances/$var/out_heuristic_troughput" 
done

echo "Push results to the repository..."
git add *
git commit -m "Validation results - Autocommit"
git push
