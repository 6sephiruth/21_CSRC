
# hw

MODEL := cifar10_KAGGLE

# cifar10 전용

GPU := 2
SEED := 0
EPOCHS := 100
BATCH := 32

XAI := shap



MODEL_ARG := --model $(MODEL) --gpu $(GPU) --seed $(SEED) --epochs $(EPOCHS) --batch $(BATCH) --xai $(XAI)


base : *.py
		@python3 -u base.py $(MODEL_ARG)