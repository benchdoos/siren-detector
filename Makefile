RPI         = doos@192.168.1.100
RPI_BASE    = ~/Projects/esphome
CONTAINER   = siren-detector-engine
WEB_UI      = http://192.168.1.100:8080
LABELS      = siren background motorcycle explosion unknown

# ── Deploy ────────────────────────────────────────────────────────────────────

## Full rebuild: copy detector + compose + rebuild Docker container on RPi
deploy:
	scp -r detector/ $(RPI):$(RPI_BASE)/
	scp hub/docker-compose.yml $(RPI):$(RPI_BASE)/hub/
	ssh $(RPI) "cd $(RPI_BASE)/hub && docker compose up --build -d detector && docker image prune -f"

## Deploy only frontend (no rebuild needed)
deploy-ui:
	scp detector/static/index.html $(RPI):$(RPI_BASE)/detector/static/
	ssh $(RPI) "docker cp $(RPI_BASE)/detector/static/index.html $(CONTAINER):/app/static/index.html"

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

## Deploy trained model weights to RPi and restart detector
deploy-model:
	scp ml/results/classifier_weights.npz ml/results/classifier_labels.json \
		ml/results/val_metrics.json \
		$(RPI):$(RPI_BASE)/ml/results/
	ssh $(RPI) "docker restart $(CONTAINER)"

## Full retrain cycle: merge recorder → sync dataset → train → clean synced → deploy model (restart)
retrain:
	$(MAKE) _merge-recorder
	$(MAKE) sync-dataset
	$(MAKE) augment
	$(MAKE) train
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

## Clean from RPi only files we have locally (= synced + trained on). Safe for new arrivals.
_clean-trained:
	@count=0; \
	for label in $(LABELS); do \
		if ls dataset/$$label/*.wav >/dev/null 2>&1; then \
			NAMES=$$(cd dataset/$$label && ls *.wav | tr '\n' ' '); \
			ssh $(RPI) "cd $(RPI_BASE)/dataset/$$label && rm -f $$NAMES" && \
			count=$$((count + $$(echo $$NAMES | wc -w))); \
		fi; \
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

## Augment siren data (200 samples by default)
augment-siren:
	ml/.venv/bin/python dataset/augment.py siren --count 200

## Augment explosion data (150 samples by default)
augment-explosion:
	ml/.venv/bin/python dataset/augment.py explosion --count 150

## Augment motorcycle data (200 normal + 100 quiet)
augment-motorcycle:
	ml/.venv/bin/python dataset/augment.py motorcycle --count 200
	ml/.venv/bin/python dataset/augment.py motorcycle --count 100 --quiet

## Augment all safety-critical classes
augment:
	$(MAKE) augment-siren
	$(MAKE) augment-explosion
	$(MAKE) augment-motorcycle

# ── Utilities ─────────────────────────────────────────────────────────────────

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
	@echo "    make deploy-ui       Frontend only (no rebuild)"
	@echo "    make deploy-compose  Push docker-compose.yml"
	@echo "    make restart         Restart container"
	@echo "    make logs            Tail container logs"
	@echo ""
	@echo "  ML"
	@echo "    make sync-dataset    Sync labeled data from RPi"
	@echo "    make train           Train MLP classifier"
	@echo "    make deploy-model    Push weights to RPi + restart"
	@echo "    make retrain         Full cycle: sync → train → deploy"
	@echo "    make clean-clips     Delete saved clips on RPi"
	@echo "    make clean-dataset   Delete labeled WAVs on RPi"
	@echo "    make clean-docker    Prune Docker images/cache (keeps volumes)"
	@echo ""
	@echo "  Other"
	@echo "    make unlock          Unlock SSH key for 10 min"
	@echo "    make open            Open Web UI in browser"
	@echo "    make fetch-esc50     Download ESC-50 dataset
	@echo "    make augment         Augment siren + explosion data"
	@echo "    make augment-siren   Augment siren only"
	@echo "    make augment-explosion Augment explosion only"
	@echo "    make augment-motorcycle Augment motorcycle (normal + quiet)"
	@echo ""

.PHONY: deploy deploy-ui deploy-compose restart logs \
        sync-dataset train deploy-model retrain _merge-recorder _clean-trained \
        clean-clips clean-dataset clean-docker \
        augment augment-siren augment-explosion augment-motorcycle \
        fetch-esc50 open unlock help
