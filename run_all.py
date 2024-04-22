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
        ['python', 'tools/metrics.py', '--conf', conf_path, '--weights', pth_path, '--gpu', str(gpu), '--mmdet'])
    exit(123)


def test_process(jobs, gpu):
    for job in jobs:
        test_network(*job, gpu)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--gpus", type=int, nargs='+', default=[0], help="GPU id for evaluation")
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

    selected_models = ['detr', 'rtmdet', 'yolo', 'convnext', 'swin', 'mask2former', 'maskformer', 'swin', 'faster_rcnn',
                       'cascade_rcnn', 'cascade_rpn', 'centernet', 'dino', 'efficientnet', 'yolof', 'yolox', 'ssd']
    # to_run = [r for r in to_run if any([m in r[0].name for m in selected_models])]
    models = set()
    for r in to_run:
        p = r[0]
        models.add(p)
    for r in sorted(models):
        print(r)
    print(len(to_run))
    exit(1)

    processes = []
    gpus = args.gpus
    print(f'Testing {len(to_run)} models on {len(gpus)} GPUs')
    for i in gpus:
        process = mp.Process(target=test_process, args=(to_run[i::4], i))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
