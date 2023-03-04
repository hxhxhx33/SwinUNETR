include .env
include .env.local
export

BASE_ARGS=\
	--data_root "$(SWINUNETR_DATA_ROOT)" \
	--fold_map "./resource/folds.json"


train:
	ARGS_FILES="arg/model.txt arg/runtime.txt arg/train.txt" \
	python -m cmd.train $(BASE_ARGS) \
		--folds 1 2 3 4 \
		--checkpoint_dir "$(SWINUNETR_WORKSPACE)/checkpoint"

predict:
	ARGS_FILES="arg/model.txt arg/runtime.txt arg/predict.txt" \
	python -m cmd.predict $(BASE_ARGS) \
		--folds 0 \
		--checkpoint $(SWINUNETR_WORKSPACE)/checkpoint/model-60.pt \
		--output_dir $(SWINUNETR_WORKSPACE)/prediction

evaluate:
	python -m cmd.evaluate $(BASE_ARGS) \
		--folds 0 \
		--prediction_dir $(SWINUNETR_WORKSPACE)/prediction
