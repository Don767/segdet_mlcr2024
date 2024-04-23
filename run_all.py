import argparse
import collections
import multiprocessing as mp
import os
import pathlib
import re
import subprocess

import requests
from tqdm import tqdm


def clone_repo(repo_dir):
    print('Cloning mmdetection repo...')
    os.system(f'git clone https://github.com/open-mmlab/mmdetection {repo_dir}')


def parse_readme(readme: pathlib.Path) -> [(pathlib.Path, str, str)]:
    config_re = re.compile(r'\[[cC]onfig\]\([\.\/]*([^\)]*)\)')
    weights_re = re.compile(r'((\[model)|(\[reid))\]\(([^\)]*)\)')

    out = []
    with open(readme, 'r') as f:
        lines = f.readlines()
    for line in lines:
        config = config_re.findall(line)
        model = weights_re.findall(line)

        if len(config) == len(model) and len(config) > 0:
            c = config[0]
            m = model[0][3]
            fname = readme.parent.name
            if (m != '<>'
                    and not fname.startswith('groie')
                    and not fname.startswith('ocsort')
                    and not fname.startswith('mm_grounding_dino')
                    and '/' not in c):
                out.append((readme.parent, c, m))
    return out


def download_weights(args):
    (path, config, model), out_dir = args
    out_dir.mkdir(parents=True, exist_ok=True)

    if config.startswith('http'):
        response = requests.get(config)
        c = config.split('/')[-1]
        with open(out_dir / c, 'wb') as f:
            f.write(response.content)
    else:
        c = config.replace('.py', '.pth')

    pth_name = c.replace('.py', '.pth')
    output_file = out_dir / pth_name

    if output_file.exists():
        return

    try:
        response = requests.get(model)
        with open(output_file, 'wb') as f:
            f.write(response.content)
    except Exception as e:
        print(f'Failed to download {model} {e}')


def test_network(path, conf, pth, gpu):
    conf_path = pathlib.Path(path) / conf
    pth_path = pathlib.Path('data/mmdetection_weights') / conf.replace('.py', '.pth')
    print(f'Testing {conf_path} {pth_path} on GPU {gpu}...')
    subprocess.run(
        ['python', 'tools/metrics.py', '--conf', conf_path, '--weights', pth_path, '--gpu', str(gpu), '--mmdet'],
        capture_output=True)


def test_process(jobs, gpu):
    for i, job in enumerate(jobs):
        test_network(*job, gpu)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--gpus", default='0', help="GPU id for evaluation")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    repo_dir = pathlib.Path('data/mmdetection')

    if not repo_dir.exists():
        clone_repo(repo_dir)

    config_dir = repo_dir / 'configs'
    base_exclude = {'_base_', 'misc', 'common'}
    model_readmes = sorted(
        [conf / 'README.md' for conf in config_dir.iterdir() if conf.name not in base_exclude])

    print(f'Parsing {len(model_readmes)} readmes')
    to_run = []
    for readme in model_readmes:
        to_run.extend(parse_readme(readme))

    print(f'Downloading {len(to_run)} weights')
    out_dir = pathlib.Path('data/mmdetection_weights')
    with mp.Pool(20) as p:
        it = tqdm(p.imap(download_weights, [(r, out_dir) for r in to_run]), total=len(to_run))
        collections.deque(it, maxlen=0)

    selected_models = {'cascade_rcnn': '7,16,4', 'cascade_rpn': '1', 'centernet': '0', 'conditional_detr': '0',
        'convnext': '0,2',
        'dab_detr': '0',
        'deformable_detr': '0,2',
        'detr': '0',
        'dino': '1,3',
        'efficientnet': '0',
        'faster_rcnn': '30',
        'grounding_dino': '0,3',
        'mask2former': '2,7',
        'maskformer': '0,1',
        'rtmdet': '3,1,12',
        'ssd': '0,1,2',
        'swin': '1,3',
        'yolo': '0,4',
        'yolof': '0',
        'yolox': '0,1,2,3'
    }
    selected_models = {k: [int(i.strip()) for i in v.split(',')] for k, v in selected_models.items()}
    to_run = sorted([r for r in to_run if any([m in r[0].name for m in selected_models.keys()])])

    jobs = []
    for model_name, conf_indices in selected_models.items():
        all_model_configs = [r for r in to_run if r[0].name == model_name]
        for ci in conf_indices:
            jobs.append(all_model_configs[ci])

    processes = []
    gpus = [int(g) for g in args.gpus.split(',')] if ',' in args.gpus else [int(args.gpus)]
    ngpu = len(gpus)
    print(f'Testing {len(jobs)} models on {ngpu} GPUs')
    for i in gpus:
        process = mp.Process(target=test_process, args=(jobs[i::ngpu], i))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
