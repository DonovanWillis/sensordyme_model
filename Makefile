.PHONY: dev dev-debug dev-max bench accuracy report deploy test

VENV := .venv/bin

dev:
	$(VENV)/python main.py

dev-debug:
	$(VENV)/python main.py --debug

dev-max:
	$(VENV)/python main.py --profile mac_max --debug

dev-openpilot: # comma.ai openpilot's driver monitoring, verbatim (model + policy)
	$(VENV)/python main.py --module driver_attention_openpilot --debug

dev-skinnemoen: # erlendskinnemoen's MediaPipe FaceMesh DMS (github course project), verbatim
	$(VENV)/python main.py --module driver_attention_skinnemoen --debug

dev-dashsentinel: # DashSentinel (github.com/alec-kr/DashSentinel), vendored under third_party/
	$(VENV)/python main.py --module driver_attention_dashsentinel --debug

bench:
	$(VENV)/python benchmarks/run_benchmarks.py

accuracy: # run the timed scenario script (make dev-max must already be running + calibrated)
	$(VENV)/python benchmarks/accuracy_protocol.py run

report:
	$(VENV)/python -c "from core.reports import generate_weekly_report; print(generate_weekly_report('data/events.csv', 'data/reports'))"

# config.yaml is DEVICE STATE, not code: it holds this unit's device_key, its
# cloud address, its camera orientation and its .hef path. Copying the repo's
# copy over it silently unpairs the device from the backend and un-rotates the
# camera, and the failure shows up much later as "the app stopped seeing it".
# Config changes ship as explicit migrations (scripts/migrate_config.py), never
# as a side effect of deploying code.
deploy:
	rsync -av --exclude='.venv' --exclude='data' --exclude='.git' --exclude='__pycache__' \
		--exclude='config.yaml' --exclude='*.log' --exclude='.secrets' \
		./ pi@sensordyme.local:~/sensordyme/
	ssh pi@sensordyme.local 'cd ~/sensordyme && python3 scripts/migrate_config.py config.yaml'
	ssh pi@sensordyme.local 'cd ~/sensordyme && sudo systemctl restart sensordyme.service'

test:
	$(VENV)/python -m pytest -q
