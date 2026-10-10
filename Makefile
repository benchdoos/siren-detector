RPI         = doos@192.168.1.100
RPI_BASE    = ~/Projects/esphome
CONTAINER   = siren-detector-engine
WEB_UI      = http://192.168.1.100:8080

# ── Deploy ────────────────────────────────────────────────────────────────────

## Full rebuild: copy detector + rebuild Docker container on RPi
deploy:
	scp -r detector/ $(RPI):$(RPI_BASE)/
	ssh $(RPI) "cd $(RPI_BASE)/hub && docker compose up --build -d detector"

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
		$(RPI):$(RPI_BASE)/ml/results/
	ssh $(RPI) "docker restart $(CONTAINER)"

## Full retrain cycle: sync dataset → train → deploy model → restart
retrain: sync-dataset train deploy-model

# ── Dataset ───────────────────────────────────────────────────────────────────

## Download and extract ESC-50 dataset
fetch-esc50:
	python3 dataset/fetch_esc50.py

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
	@echo ""
	@echo "  Other"
	@echo "    make unlock          Unlock SSH key for 10 min"
	@echo "    make open            Open Web UI in browser"
	@echo "    make fetch-esc50     Download ESC-50 dataset"
	@echo ""

.PHONY: deploy deploy-ui deploy-compose restart logs \
        sync-dataset train deploy-model retrain \
        fetch-esc50 open unlock help
