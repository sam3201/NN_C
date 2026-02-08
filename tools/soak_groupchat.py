import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import requests


def run_server(env: dict):
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    os.environ.update(env)
    log_path = repo_root / "logs" / "soak_groupchat.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    os.dup2(log_file.fileno(), 1)
    os.dup2(log_file.fileno(), 2)
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


def run_groupchat(messages, wait_s: int = 45):
    try:
        import socketio
    except Exception as exc:
        raise RuntimeError(f"python-socketio not available: {exc}")

    sio = socketio.Client(logger=False, engineio_logger=False, reconnection=True)
    received = []
    agent_messages = []
    user_info = {"id": None}
    joined = {"ok": False}

    @sio.on('user_connected')
    def on_user_connected(data):
        user = data.get('user') if isinstance(data, dict) else None
        if user:
            user_info["id"] = user.get('id')

    @sio.on('message_received')
    def on_message(data):
        received.append(data)
        if isinstance(data, dict) and data.get("message_type") == "agent":
            agent_messages.append(data)

    @sio.on('joined_room')
    def on_joined_room(_data):
        joined["ok"] = True

    sio.connect('http://localhost:5004', wait_timeout=20, transports=['polling'])

    deadline = time.time() + 10
    while time.time() < deadline and not user_info["id"]:
        time.sleep(0.1)
    if not user_info["id"]:
        user_info["id"] = sio.sid

    user_id = user_info["id"]
    room_id = f"soak-room-{int(time.time())}"
    sio.emit('join_room', {
        'user_id': user_id,
        'room_id': room_id,
        'room_name': 'Soak Room',
        'agent_type': 'sam'
    })

    deadline = time.time() + 5
    while time.time() < deadline and not joined["ok"]:
        time.sleep(0.1)

    for msg in messages:
        sio.emit('send_group_message', {
            'user_id': user_id,
            'room_id': room_id,
            'message': msg
        })
        time.sleep(3)
    deadline = time.time() + wait_s
    expected_agents = max(1, len(messages))
    while time.time() < deadline and len(agent_messages) < expected_agents:
        time.sleep(0.25)
    sio.disconnect()
    return room_id, received, agent_messages


def main():
    parser = argparse.ArgumentParser(description="Live groupchat soak test with distillation verification")
    parser.add_argument("--teacher", default=os.getenv("SAM_TEACHER_POOL", "ollama:qwen2.5-coder:7b"))
    parser.add_argument("--distill-path", default=os.getenv("SAM_DISTILL_PATH", "training/distilled/groupchat.jsonl"))
    parser.add_argument("--wait", type=int, default=60)
    parser.add_argument("--messages", type=int, default=2)
    args = parser.parse_args()

    env = os.environ.copy()
    env['SAM_TEACHER_POOL_ENABLED'] = '1'
    env['SAM_TEACHER_POOL'] = args.teacher
    env['SAM_TEACHER_N_PER'] = env.get('SAM_TEACHER_N_PER', '1')
    env['SAM_TEACHER_MIN_SIM'] = env.get('SAM_TEACHER_MIN_SIM', '0.72')
    env['SAM_TEACHER_MIN_VOTES'] = env.get('SAM_TEACHER_MIN_VOTES', '1')
    env['SAM_DISTILL_PATH'] = args.distill_path
    env['SAM_AUTONOMOUS_ENABLED'] = '0'

    server_proc = mp.Process(target=run_server, args=(env,))
    server_proc.start()

    try:
        if not wait_for_health('http://localhost:5004/api/status', timeout_s=90):
            raise RuntimeError('Server did not become healthy within timeout')

        messages = [
            f"Soak test message {idx + 1}: confirm consensus and distillation."
            for idx in range(args.messages)
        ]
        room_id, received, agent_messages = run_groupchat(messages, wait_s=args.wait)

        distill_path = Path(args.distill_path)
        latest_line = None
        matching_record = None
        if distill_path.exists():
            with distill_path.open('r', encoding='utf-8') as handle:
                lines = handle.readlines()[-50:]
                if lines:
                    latest_line = lines[-1].strip()
                for line in reversed(lines):
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    task_id = payload.get("task_id", "")
                    if task_id.startswith(f"groupchat:{room_id}:"):
                        matching_record = payload
                        break

        print(json.dumps({
            'room_id': room_id,
            'received_messages': len(received),
            'agent_messages': len(agent_messages),
            'distill_path': str(distill_path),
            'latest_distill_record': latest_line,
            'matching_distill_record': matching_record,
        }, indent=2))
    finally:
        if server_proc.is_alive():
            server_proc.terminate()
            server_proc.join(timeout=10)


if __name__ == '__main__':
    main()
