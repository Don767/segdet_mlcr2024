import collections
import multiprocessing as mp
import os
import pathlib
import re

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
    print(f'python tools/speed_benchmark.py --conf {conf_path} --weights {pth_path} --gpu {gpu}')

def test_process(jobs, gpu):
    for job in jobs:
        test_network(*job, gpu)

if __name__ == '__main__':
    repo_dir = pathlib.Path('data/mmdetection')

    if not repo_dir.exists():
        clone_repo(repo_dir)

    config_dir = repo_dir / 'configs'
    model_readmes = sorted(
        [conf / 'README.md' for conf in config_dir.iterdir() if conf.name not in {'_base_', 'misc', 'common'}])

    print(f'Parsing {len(model_readmes)} readmes')
    to_run = []
    for readme in model_readmes:
        to_run.extend(parse_readme(readme))

    print(f'Downloading {len(to_run)} weights')
    out_dir = pathlib.Path('data/mmdetection_weights')
    with mp.Pool(20) as p:
        it = tqdm(p.imap(download_weights, [(r, out_dir) for r in to_run]), total=len(to_run))
        collections.deque(it, maxlen=0)

    processes = []
    for i in range(4):
        process = mp.Process(target=test_process, args=(to_run[i::4], i))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
