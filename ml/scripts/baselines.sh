for n in 0 25 50 75 100 150 200 300 400 500 600 700 800 900
do 
    python ../main.py --type processed --seed 1 --num_states 60 --normalize --importance --save --model all --load_only $n --name md09
done
