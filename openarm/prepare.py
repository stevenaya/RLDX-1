"""OpenArm adaptation: inspect read-only inputs and fetch pinned public assets.

Execute only in the private model environment, never the credentialed controller.
"""
import json
import os
import time
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

pa.set_cpu_count(2)
pa.set_io_thread_count(2)


def main():
    assert os.environ['MODEL_LAB_STAGE_KIND'] == 'none'
    assert int(os.environ['MODEL_LAB_MAX_STEPS']) == 0
    root = Path('/data/pillow-0702-filtered')
    scratch = Path(os.environ['RLDX_SCRATCH'])
    output = Path(os.environ['RLDX_OUTPUT'])
    output.mkdir(parents=True, exist_ok=True)
    info = json.loads((root / 'meta/info.json').read_text())
    episodes = [json.loads(x) for x in (root / 'meta/episodes.jsonl').read_text().splitlines()]
    assert len(episodes) == info['total_episodes'] == 2418
    assert sum(x['length'] for x in episodes) == info['total_frames']
    assert info['splits'] == {'train': '0:1934', 'val': '1934:2418'}
    cameras = ['wrist_right', 'wrist_left', 'ceiling', 'head_left']
    report = dict(source=str(root), info=info, actual_episodes=len(episodes), samples=[], cameras=cameras)
    warmup = {}
    for episode in (0, 966, 1933, 1934, 2417):
        path = root / info['data_path'].format(episode_chunk=episode // 1000, episode_index=episode)
        table = pq.read_table(path)
        state = np.stack(table['observation.state'].to_pylist())
        action = np.stack(table['action'].to_pylist())
        assert state.shape == action.shape and state.shape[1] == 16
        assert np.isfinite(state).all() and np.isfinite(action).all()
        report['samples'].append(dict(episode=episode, rows=len(table), columns=table.column_names,
            state_min=state.min(0).tolist(), state_max=state.max(0).tolist(),
            action_min=action.min(0).tolist(), action_max=action.max(0).tolist(),
            same_frame_action_minus_state_mean_abs=np.abs(action-state).mean(0).tolist(),
            arrow_metadata={str(k):str(v) for k,v in (table.schema.metadata or {}).items()}))
        if episode == 0:
            warmup['qpos'] = state[0].astype(np.float32)
            task_id = table['task_index'][0].as_py()
            tasks = {x['task_index']: x['task'] for x in map(json.loads, (root/'meta/tasks.jsonl').read_text().splitlines())}
            warmup['prompt'] = np.asarray(tasks[task_id])
    report['video_probe'] = []
    for camera in cameras:
        path = root / info['video_path'].format(episode_chunk=0, episode_index=0, video_key='observation.images.'+camera)
        start = time.monotonic()
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 2
            stream.thread_type = 'SLICE'
            frame = next(container.decode(stream)).to_ndarray(format='rgb24')
            report['video_probe'].append(dict(camera=camera, shape=list(frame.shape), fps=str(stream.average_rate), seconds=time.monotonic()-start))
            warmup['camera_'+camera] = frame
    np.savez_compressed(output/'warmup-private.npz', **warmup)
    (output/'data-audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('actual_episodes','cameras','video_probe')}, indent=2), flush=True)
    # These are public ungated repos. Never read the user's Hub credentials.
    os.environ.update(HF_HUB_OFFLINE='0', HF_DATASETS_OFFLINE='0', TRANSFORMERS_OFFLINE='0', HF_HUB_DISABLE_XET='1')
    from huggingface_hub import snapshot_download
    assets = {}
    for repo, revision, folder, patterns in [
        ('RLWRLD/RLDX-1-PT-IMG','d67fc642a7e768049ba41ff8cf0dfe0ff17f556b','base', ['*.json','*.safetensors','processor/*','LICENSE.md']),
        ('RLWRLD/RLDX-1-VLM','4b9f870d1287e0d38d7eb1445e6d8c60afe66dd7','vlm', ['*.json','*.txt','*.jinja','LICENSE.md']),
    ]:
        target = scratch/'assets'/folder
        snapshot_download(repo, revision=revision, local_dir=target, allow_patterns=patterns, max_workers=2, token=False)
        assets[repo] = dict(revision=revision,path=str(target),files=[dict(path=str(p.relative_to(target)),bytes=p.stat().st_size) for p in sorted(target.rglob('*')) if p.is_file() and '.cache' not in p.parts])
    (output/'assets.json').write_text(json.dumps(assets,indent=2)+'\n')
    print('Prepared pinned public assets and private recorded warmup input.', flush=True)


if __name__ == '__main__':
    main()
