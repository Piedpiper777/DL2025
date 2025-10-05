# kill 5001/8500/8501/8502/8602/8702
ports=(5001 7687 8500 8501 8502 8602 8702 8801 8802 8803 8901 9001 9002 9003 9101 9201)
for port in "${ports[@]}"
do
kill $(lsof -i:$port | awk '{print $2}' | awk 'NR==2{print}')
done