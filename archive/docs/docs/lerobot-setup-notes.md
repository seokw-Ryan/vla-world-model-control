# LeRobot Setup Notes

## Environment
- Conda env: `lerobot` (Python 3.12, upgraded from 3.11)
- LeRobot v0.5.1 installed as editable from `./lerobot/` (cloned from GitHub)
- Extra dependency installed: `feetech-servo-sdk` (provides `scservo_sdk`)

## What we did

1. **Fixed broken editable install** — lerobot was previously installed as editable pointing to `/home/rocket/Projects/compass-cli/lerobot` which no longer exists. Cloned fresh from GitHub into this repo and reinstalled.

2. **Upgraded Python** — lerobot 0.5.x requires Python >=3.12. Ran `conda install -n lerobot python=3.12`.

3. **Installed missing servo SDK** — `scservo_sdk` (Feetech servo library) was missing. Installed via `pip install feetech-servo-sdk`.

4. **Ran `lerobot-find-port`** — Detected `/dev/ttyACM0` as the motor bus port.

5. **Ran `lerobot-setup-motors`** — Hit "Motor not found" error (see below).

## Setup Motors Command

```bash
conda activate lerobot
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM0
```

## Current Issue: Motor Not Found

```
RuntimeError: Motor 'gripper' (model 'sts3215') was not found. Make sure it is connected.
```

### Things to check
- **Only one motor connected** — the script expects exactly the gripper motor plugged in, no others
- **Motor is powered** — controller board needs power (battery/USB) in addition to data connection
- **Correct robot type** — if it's an SO-100 (not SO-101), use `--robot.type=so100_follower`
- **Motor model** — SO-101 expects STS3215 servos; if yours are SCS series it won't match
- **Baudrate** — if motor baudrate was changed from factory default, the scan may miss it
