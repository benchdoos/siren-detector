# Load deployment config from .env (gitignored).
# Copy .env.example to .env and fill in your values.
-include .env

RPI         ?= user@192.168.1.100
RPI_BASE    ?= ~/Projects/esphome
CONTAINER   ?= siren-detector-engine
WEB_UI      ?= http://192.168.1.100:8080
LABELS      = siren background motorcycle explosion unknown

# ── Deploy ────────────────────────────────────────────────────────────────────

## Full rebuild: copy detector + compose + rebuild Docker container on RPi
deploy:
	rsync -av --delete --exclude='__pycache__' detector/ $(RPI):$(RPI_BASE)/detector/
	scp hub/docker-compose.yml $(RPI):$(RPI_BASE)/hub/
	@test -f hub/.env && scp hub/.env $(RPI):$(RPI_BASE)/hub/.env || true
	ssh $(RPI) "cd $(RPI_BASE)/hub && docker compose up --build -d detector && docker image prune -f"

## Deploy audio_source to RPi and rebuild
deploy-audio:
	rsync -av --delete --exclude='__pycache__' audio_source/ $(RPI):$(RPI_BASE)/audio_source/
	ssh $(RPI) "cd $(RPI_BASE)/audio_source && docker compose up --build -d"

## Deploy only frontend (no rebuild needed)
deploy-ui:
	rsync -av --exclude='.DS_Store' detector/static/ $(RPI):$(RPI_BASE)/detector/static/
	ssh $(RPI) "docker cp $(RPI_BASE)/detector/static/. $(CONTAINER):/app/detector/static/"

ESP_YAML    ?= esp32/siren-detector.yaml
ESP_DEVICE  ?= siren-detector.local

## Copy, validate, compile, and flash ESP32 firmware via OTA
deploy-firmware:
	scp $(ESP_YAML) $(RPI):$(RPI_BASE)/esphome/siren-detector.yaml
	ssh $(RPI) "cd $(RPI_BASE) && docker compose run --rm esphome config /config/siren-detector.yaml"
	ssh $(RPI) "cd $(RPI_BASE) && docker compose run --rm esphome compile /config/siren-detector.yaml"
	ssh $(RPI) "cd $(RPI_BASE) && docker compose run --rm esphome upload /config/siren-detector.yaml --device $(ESP_DEVICE)"

## Stream ESP32 logs via ESPHome
logs-esp:
	ssh $(RPI) "cd $(RPI_BASE) && docker compose run --rm esphome logs /config/siren-detector.yaml --device $(ESP_DEVICE)"

## Deploy docker-compose.yml to RPi
deploy-compose:
	scp hub/docker-compose.yml $(RPI):$(RPI_BASE)/hub/

## Restart detector container (no rebuild)
restart:
	ssh $(RPI) "docker restart $(CONTAINER)"

## View detector logs on RPi
logs:
	ssh $(RPI) "docker logs -f $(CONTAINER)"

# ── ML / Retraining ──────────────────────────────────────────────────────────

## Sync labeled dataset from RPi to local Mac
sync-dataset:
	rsync -av $(RPI):$(RPI_BASE)/dataset/ dataset/

## Train MLP classifier on local dataset
train:
	ml/.venv/bin/python ml/train_classifier.py

## Train DSP pre-filter on local dataset
DSP_TRAIN_ARGS ?=
train-dsp:
	ml/.venv/bin/python ml/train_dsp_classifier.py $(DSP_TRAIN_ARGS)

## Deploy trained model weights and refresh detector compose configuration on RPi
deploy-model:
	scp ml/results/classifier_weights.npz ml/results/classifier_labels.json \
		ml/results/val_metrics.json \
		ml/results/model_card.json \
		$(RPI):$(RPI_BASE)/ml/results/
	scp hub/docker-compose.yml $(RPI):$(RPI_BASE)/hub/docker-compose.yml
	@if [ -f ml/results/dsp_metrics.json ]; then \
		scp ml/results/dsp_metrics.json $(RPI):$(RPI_BASE)/ml/results/; \
	fi
	@if ls ml/results/dsp_classifier_f32.tflite ml/results/dsp_norm_stats.npz ml/results/dsp_classifier_labels.json >/dev/null 2>&1; then \
		echo "Deploying DSP float32 model..."; \
		scp ml/results/dsp_classifier_f32.tflite ml/results/dsp_norm_stats.npz \
			ml/results/dsp_classifier_labels.json \
			$(RPI):$(RPI_BASE)/ml/results/; \
	fi
	@if [ -d ml/results/audits ]; then \
		ssh $(RPI) "mkdir -p $(RPI_BASE)/ml/results/audits"; \
		scp ml/results/audits/audit_*.json $(RPI):$(RPI_BASE)/ml/results/audits/; \
	fi
	ssh $(RPI) "cd $(RPI_BASE)/hub && docker compose up -d detector"

## Full retrain cycle: merge recorder → sync dataset → train → audit → clean synced → deploy model (restart)
retrain:
	$(MAKE) _merge-recorder
	$(MAKE) _wait-calibration
	$(MAKE) sync-dataset
	$(MAKE) augment
	$(MAKE) train-dsp
	$(MAKE) train
	$(MAKE) audit
	-$(MAKE) audit-compare
	$(MAKE) _clean-trained
	$(MAKE) deploy-model

## Move labeled WAVs from dataset/recorder/ into main dataset/ on RPi
_merge-recorder:
	@for label in $(LABELS); do \
		ssh $(RPI) "mkdir -p $(RPI_BASE)/dataset/$$label && \
			if ls $(RPI_BASE)/dataset/recorder/$$label/*.wav >/dev/null 2>&1; then \
				sudo chown \$$(id -u):\$$(id -g) $(RPI_BASE)/dataset/recorder/$$label/ $(RPI_BASE)/dataset/recorder/$$label/*.wav && \
				mv $(RPI_BASE)/dataset/recorder/$$label/*.wav $(RPI_BASE)/dataset/$$label/ && \
				echo \"Moved $$label WAVs from recorder to dataset\"; \
			fi"; \
	done

## Wait for calibration to finish (if active), then stop & save it
_wait-calibration:
	@_token=$${ADMIN_TOKEN:-$$(grep '^ADMIN_TOKEN=' hub/.env 2>/dev/null | cut -d= -f2)}; \
	_url="$(WEB_UI)/api/calibrate"; \
	_status=$$(curl -sf -H "Authorization: Bearer $$_token" "$$_url" 2>/dev/null) || { echo "Detector API unreachable, skipping calibration check"; exit 0; }; \
	_active=$$(echo "$$_status" | python3 -c "import sys,json; print(json.load(sys.stdin).get('active',False))"); \
	if [ "$$_active" != "True" ]; then \
		echo "No calibration in progress"; \
		exit 0; \
	fi; \
	echo "Calibration in progress — waiting for it to finish..."; \
	while true; do \
		_status=$$(curl -sf -H "Authorization: Bearer $$_token" "$$_url"); \
		_done=$$(echo "$$_status" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('done',False) or not d.get('active',False))"); \
		if [ "$$_done" = "True" ]; then break; fi; \
		_progress=$$(echo "$$_status" | python3 -c "import sys,json; print(json.load(sys.stdin).get('progress',0))"); \
		printf "\r  progress: %s%%" "$$_progress"; \
		sleep 2; \
	done; \
	echo ""; \
	_status=$$(curl -sf -H "Authorization: Bearer $$_token" "$$_url"); \
	_still_active=$$(echo "$$_status" | python3 -c "import sys,json; print(json.load(sys.stdin).get('active',False))"); \
	if [ "$$_still_active" = "True" ]; then \
		echo "Calibration done — saving recording..."; \
		curl -sf -X POST -H "Authorization: Bearer $$_token" "$(WEB_UI)/api/calibrate/stop" > /dev/null; \
		echo "Calibration saved"; \
	fi

## Clean from RPi only files we have locally (= synced + trained on). Safe for new arrivals.
_clean-trained:
	@count=0; \
	for label in $(LABELS); do \
		if ls dataset/$$label/*.wav >/dev/null 2>&1; then \
			NAMES=$$(cd dataset/$$label && ls *.wav | tr '\n' ' '); \
			ssh $(RPI) "cd $(RPI_BASE)/dataset/$$label && rm -f $$NAMES" && \
			count=$$((count + $$(echo $$NAMES | wc -w))); \
		fi; \
		for subdir in dataset/$$label/*/; do \
			[ -d "$$subdir" ] || continue; \
			subname=$$(basename "$$subdir"); \
			if [ "$$label" = "background" ]; then continue; fi; \
			if ls "$$subdir"*.wav >/dev/null 2>&1; then \
				NAMES=$$(cd "$$subdir" && ls *.wav | tr '\n' ' '); \
				ssh $(RPI) "cd $(RPI_BASE)/dataset/$$label/$$subname && rm -f $$NAMES" && \
				count=$$((count + $$(echo $$NAMES | wc -w))); \
			fi; \
		done; \
	done; \
	ssh $(RPI) "docker exec $(CONTAINER) sh -c 'rm -f /clips/*.wav'" && \
	echo "Cleaned $$count dataset files + all clips from RPi"

## Clean saved clips on RPi (all)
clean-clips:
	ssh $(RPI) "docker exec $(CONTAINER) find /clips -name '*.wav' -delete"

## Clean dataset on RPi (all — sync first!)
clean-dataset:
	ssh $(RPI) "rm -rf $(RPI_BASE)/dataset/siren/* $(RPI_BASE)/dataset/background/* $(RPI_BASE)/dataset/motorcycle/* $(RPI_BASE)/dataset/explosion/* $(RPI_BASE)/dataset/unknown/*"

## Clean Docker build cache and dangling images on RPi (preserves volumes)
clean-docker:
	ssh $(RPI) "docker system prune -a -f"

# ── Dataset ───────────────────────────────────────────────────────────────────

## Download and extract ESC-50 dataset
fetch-esc50:
	python3 dataset/fetch_esc50.py

## Remove all augmented files before regenerating
clean-aug:
	find dataset/siren dataset/explosion dataset/motorcycle -name 'aug_*' -delete 2>/dev/null || true
	@echo "Cleaned aug files"

## Augment siren data (aug ≤ 2× real; 86 real → max 172)
NOISE_PROFILE := dataset/background/noise-profile-inmp441.wav

augment-siren:
	ml/.venv/bin/python dataset/augment.py siren --count 172 --noise-profile $(NOISE_PROFILE)

## Augment explosion data (258 real → max 516, using 150)
augment-explosion:
	ml/.venv/bin/python dataset/augment.py explosion --count 150 --noise-profile $(NOISE_PROFILE)

## Augment motorcycle data (aug ≤ 2× real; 80 real → max 160: 120 normal + 40 quiet)
augment-motorcycle:
	ml/.venv/bin/python dataset/augment.py motorcycle --count 120 --noise-profile $(NOISE_PROFILE)
	ml/.venv/bin/python dataset/augment.py motorcycle --count 40 --quiet --noise-profile $(NOISE_PROFILE)

## Augment all safety-critical classes (cleans old aug first)
augment:
	$(MAKE) clean-aug
	$(MAKE) augment-siren
	$(MAKE) augment-explosion
	$(MAKE) augment-motorcycle

# ── Audit ────────────────────────────────────────────────────────────────────

## Run dataset & model audit, save snapshot
audit:
	ml/.venv/bin/python ml/audit.py

## Compare latest audit with previous (or: make audit-compare OLD=file NEW=file)
audit-compare:
ifdef OLD
	ml/.venv/bin/python ml/audit_compare.py $(OLD) $(NEW)
else
	ml/.venv/bin/python ml/audit_compare.py
endif

# ── Utilities ─────────────────────────────────────────────────────────────────

## Dump current runtime settings from detector (JSON, pipe to jq for pretty-print)
settings:
	@_token=$${ADMIN_TOKEN:-$$(grep '^ADMIN_TOKEN=' hub/.env 2>/dev/null | cut -d= -f2)}; \
	curl -sf -H "Authorization: Bearer $$_token" "$(WEB_UI)/api/settings" | python3 -m json.tool

## Open Web UI in browser
open:
	open $(WEB_UI)

## Unlock SSH key for 10 minutes
unlock:
	ssh-add -t 10m

## Show available commands
help:
	@echo ""
	@echo "  Deploy"
	@echo "    make deploy          Full rebuild on RPi"
	@echo "    make deploy-audio    Deploy audio_source to RPi"
	@echo "    make deploy-ui       Frontend only (no rebuild)"
	@echo "    make deploy-firmware Validate + compile + flash ESP32 OTA"
	@echo "    make logs-esp        Stream ESP32 logs"
	@echo "    make deploy-compose  Push docker-compose.yml"
	@echo "    make restart         Restart container"
	@echo "    make logs            Tail container logs"
	@echo ""
	@echo "  ML"
	@echo "    make sync-dataset    Sync labeled data from RPi"
	@echo "    make train           Train MLP classifier"
	@echo "    make train-dsp       Train DSP pre-filter"
	@echo "    make deploy-model    Push weights to RPi + restart"
	@echo "    make retrain         Full cycle: sync → train DSP + MLP → deploy"
	@echo "    make clean-clips     Delete saved clips on RPi"
	@echo "    make clean-dataset   Delete labeled WAVs on RPi"
	@echo "    make clean-docker    Prune Docker images/cache (keeps volumes)"
	@echo ""
	@echo "  Other"
	@echo "    make settings        Dump current runtime settings (JSON)"
	@echo "    make unlock          Unlock SSH key for 10 min"
	@echo "    make open            Open Web UI in browser"
	@echo "    make fetch-esc50     Download ESC-50 dataset
	@echo "    make augment         Augment siren + explosion data"
	@echo "    make augment-siren   Augment siren only"
	@echo "    make augment-explosion Augment explosion only"
	@echo "    make augment-motorcycle Augment motorcycle (normal + quiet)"
	@echo "    make audit           Run dataset/model audit snapshot"
	@echo "    make audit-compare   Compare latest vs previous audit"
	@echo ""

.PHONY: deploy deploy-audio deploy-ui deploy-firmware logs-esp deploy-compose restart logs settings \
        sync-dataset train train-dsp deploy-model retrain _merge-recorder _wait-calibration _clean-trained \
        clean-clips clean-dataset clean-docker \
        augment augment-siren augment-explosion augment-motorcycle \
        audit audit-compare \
        fetch-esc50 open unlock help
