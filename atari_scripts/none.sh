
game=$1
export OMP_NUM_THREADS=1

if  [ -n $2 ] ; then
    export CUDA_VISIBLE_DEVICES=$2
fi
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES

for seed in 1 2 3
do
env_name=$game/none.$seed
echo $env_name
	python main.py \
		--id=$env_name \
		--count-base-bonus=0.01 \
		--env-type=atari \
			--seed=$seed \
		--game=$game \
		--checkpoint-interval=100000
done