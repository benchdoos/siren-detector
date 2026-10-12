RPI         = doos@192.168.1.100
RPI_BASE    = ~/Projects/esphome
CONTAINER   = siren-detector-engine
WEB_UI      = http://192.168.1.100:8080

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

## Full retrain cycle: sync dataset → train → deploy model → clean synced files
retrain:
	$(MAKE) sync-dataset
	$(MAKE) _snapshot-synced
	$(MAKE) train
	$(MAKE) deploy-model
	$(MAKE) _clean-synced

## Snapshot synced filenames for safe cleanup (only delete what we actually trained on)
_snapshot-synced:
	@for label in siren background motorcycle explosion unknown; do \
		if [ -d "dataset/$$label" ]; then \
			ls "dataset/$$label/" 2>/dev/null | sort > "/tmp/retrain-$$label.list"; \
		else \
			> "/tmp/retrain-$$label.list"; \
		fi; \
	done
	@ssh $(RPI) "docker exec $(CONTAINER) find /clips -name '*.wav' -printf '%f\n'" | sort > /tmp/retrain-clips.list
	@echo "Snapshot: $$(cat /tmp/retrain-*.list | wc -l | tr -d ' ') files recorded for cleanup"

## Clean only files that were synced/trained on (safe against new arrivals)
_clean-synced:
	@for label in siren background motorcycle explosion unknown; do \
		if [ -s "/tmp/retrain-$$label.list" ]; then \
			cat "/tmp/retrain-$$label.list" | while read f; do \
				ssh $(RPI) "rm -f $(RPI_BASE)/dataset/$$label/$$f"; \
			done; \
		fi; \
	done
	@if [ -s "/tmp/retrain-clips.list" ]; then \
		cat /tmp/retrain-clips.list | while read f; do \
			ssh $(RPI) "docker exec $(CONTAINER) rm -f /clips/$$f"; \
		done; \
	fi
	@rm -f /tmp/retrain-*.list
	@echo "Cleaned synced files from RPi"

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
	@echo "    make fetch-esc50     Download ESC-50 dataset"
	@echo ""

.PHONY: deploy deploy-ui deploy-compose restart logs \
        sync-dataset train deploy-model retrain \
        _snapshot-synced _clean-synced \
        clean-clips clean-dataset clean-docker \
        fetch-esc50 open unlock help
