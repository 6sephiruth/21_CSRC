
# hw

MODEL := cifar10_CNN

# cifar10 전용

GPU := 3
SEED := 0
EPOCHS := 30
BATCH := 32

OPTIMIZER := adam



MODEL_ARG := --model $(MODEL) --gpu $(GPU) --seed $(SEED) --epochs $(EPOCHS) --batch $(BATCH) --optimizer $(OPTIMIZER)


base : *.py
		@python3 -u base.py $(MODEL_ARG)