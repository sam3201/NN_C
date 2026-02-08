import os
import sys
import time
import json
import signal
import multiprocessing as mp
from pathlib import Path

import requests


def run_server():
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    log_path = repo_root / "logs" / "soak_groupchat.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = log_file
    sys.stderr = log_file
    from complete_sam_unified import UnifiedSAMSystem
    system = UnifiedSAMSystem()
    system.run()


def wait_for_health(url: str, timeout_s: int = 60) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                return True
        except Exception:
            time.sleep(1)
    return False


def run_groupchat(messages):
    try:
        import socketio
    except Exception as exc:
        raise RuntimeError(f"python-socketio not available: {exc}")

    sio = socketio.Client(logger=False, engineio_logger=False)
    received = []

    @sio.on('message_received')
    def on_message(data):
        received.append(data)

    sio.connect('http://localhost:5004')
    user_id = sio.sid
    room_id = 'soak-room'
    sio.emit('join_room', {
        'user_id': user_id,
        'room_id': room_id,
        'room_name': 'Soak Room',
        'agent_type': 'sam'
    })

    for msg in messages:
        sio.emit('send_group_message', {
            'user_id': user_id,
            'room_id': room_id,
            'message': msg
        })
        time.sleep(3)

    time.sleep(3)
    sio.disconnect()
    return received


def main():
    os.environ.setdefault('SAM_TEACHER_POOL_ENABLED', '1')
    os.environ.setdefault('SAM_TEACHER_POOL', 'ollama:mistral:latest')
    os.environ.setdefault('SAM_TEACHER_N_PER', '1')
    os.environ.setdefault('SAM_TEACHER_MIN_SIM', '0.72')
    os.environ.setdefault('SAM_TEACHER_MIN_VOTES', '1')
    os.environ.setdefault('SAM_DISTILL_PATH', 'training/distilled/groupchat.jsonl')

    server_proc = mp.Process(target=run_server)
    server_proc.start()

    try:
        if not wait_for_health('http://localhost:5004/api/health', timeout_s=90):
            raise RuntimeError('Server did not become healthy within timeout')

        messages = [
            'Soak test message one: confirm consensus and distillation.',
            'Soak test message two: verify groupchat teacher pool response.'
        ]
        received = run_groupchat(messages)

        distill_path = Path(os.environ['SAM_DISTILL_PATH'])
        latest_line = None
        if distill_path.exists():
            with distill_path.open('r', encoding='utf-8') as handle:
                lines = handle.readlines()
                if lines:
                    latest_line = lines[-1].strip()

        print(json.dumps({
            'received_messages': len(received),
            'latest_distill_record': latest_line,
        }, indent=2))
    finally:
        if server_proc.is_alive():
            server_proc.terminate()
            server_proc.join(timeout=10)


if __name__ == '__main__':
    main()
