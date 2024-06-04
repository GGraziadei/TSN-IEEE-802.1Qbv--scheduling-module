#!/bin/bash
echo "Validation script..."

python3 validation_heuristic.py -i instance_generator/instances/wifi2wifi_express/ &
python3 validation_heuristic.py -i instance_generator/instances/wifi2wifi_storeAndForward/ &
python3 validation_heuristic.py -i instance_generator/instances/wifi2dc_express/ &
python3 validation_heuristic.py -i instance_generator/instances/wifi2dc_storeAndForward/ &

# Wait for all tasks to complete
wait
echo "All tasks completed."