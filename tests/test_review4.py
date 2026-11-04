"""Review probes against the real public sample, with no GPU or model calls."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).parent / '..'
OUT = Path('/tmp')
SAMPLE = ROOT / 'tests/fixtures/neighbour_chair_episode'
sys.path.insert(0, str(ROOT / 'src'))
from scout.env.scoring import score_episode
from scout.env.agent import canonical_answer


def test_real_sample_does_not_reward_a_neighbouring_chair():
    man = json.loads((SAMPLE/'manifest.json').read_text())
    log, ev = [json.loads((SAMPLE/'unchanged'/(f+'.json')).read_text()) for f in ['log','evaluator']]
    a = [q for q in log['qa'] if q['kind']=='answer'][-1]
    same = [o for o in ev['objects_start'] if o['category']==man['target']['category']]
    distances = {o['id']: float(np.linalg.norm(np.array(o['xy'])-a['anchor_xy'])) for o in same}
    nearest = min(distances, key=distances.get)
    target = man['target']['instance']
    fresh = [{int(i) for i in v['visible']} for v in ev['visibility'] if v['frame'] in a['now_frames']]
    score = score_episode(str(SAMPLE))['unchanged']['verify']
    print('Frozen target', target, 'distance', distances[target], 'nearest instance', nearest, 'distance', distances[nearest])
    print('Actual fresh-frame visible instances', fresh, 'score', score)
    assert nearest != target  # fixture proves the test concerns different instances
    assert all(target not in v for v in fresh)
    assert score['outcome'] != 'correct', 'Public sample checks the neighbouring chair but receives credit'


def test_real_recall_card_colour_agrees_with_its_score():
    source = ROOT/'scripts/env_video.py'
    spec = importlib.util.spec_from_file_location('review_env_video', source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tree = ast.parse(source.read_text())
    compose = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='compose')
    loop = next(n for n in compose.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id=='q')
    labels = []

    class RecordingDraw:
        def __init__(self, image):
            self.draw = ImageDraw.Draw(image)

        def __getattr__(self, name):
            return getattr(self.draw, name)

        def text(self, xy, text, **kw):
            labels.append((text, kw.get('fill')))
            return self.draw.text(xy, text, **kw)

    writer = imageio.get_writer(str(OUT/'real-sample-card.mp4'), fps=15, codec='libx264', quality=8)
    ns = dict(Image=Image, ImageDraw=SimpleNamespace(Draw=RecordingDraw), np=np, W=1280, H=720,
              st=mod.State(str(SAMPLE/'unchanged')), adir=str(SAMPLE/'unchanged'), arm='unchanged',
              brain='llama4-maverick', big=mod.font(26), mid=mod.font(20), small=mod.font(16), w=writer)
    try:
        exec(compile(ast.Module(body=[loop], type_ignores=[]), str(source), 'exec'), ns)
    finally:
        writer.close()
    ns['card'].save(OUT/'real-sample-card.png')
    actual = next(col for label,col in labels if label.startswith('Without looking'))
    score = score_episode(str(SAMPLE))['unchanged']['verify_recall']
    print('Real recall scorer:', score, 'rendered colour:', actual, 'green=(120,230,140)')
    assert score['outcome'] == 'wrong target'
    assert actual != (120,230,140), 'Renderer paints a wrong-target answer green'


def test_current_canonical_card_metadata_does_not_overlap_image_labels():
    """Real saved images, with current canonical wording, expose the final3 layout path."""
    source = ROOT/'scripts/env_video.py'
    spec = importlib.util.spec_from_file_location('review_canonical_video', source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tree = ast.parse(source.read_text())
    compose = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='compose')
    loop = next(n for n in compose.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id=='q')
    st = mod.State(str(SAMPLE/'moved'))
    # This is a layout probe, not a new episode or a changed score.
    st.log['qa'][-1]['say'] = canonical_answer('not found','chair',236.)
    st.log['recall'][-1]['say'] = 'From memory, without looking: the chair should still be there (last seen at 236 s).'
    labels = []

    class RecordingDraw:
        def __init__(self, image):
            self.draw = ImageDraw.Draw(image)

        def __getattr__(self, name):
            return getattr(self.draw, name)

        def text(self, xy, text, **kw):
            labels.append((text, xy))
            return self.draw.text(xy, text, **kw)

    writer = imageio.get_writer(str(OUT/'canonical-card-layout.mp4'),fps=15,codec='libx264',quality=8)
    ns = dict(Image=Image,ImageDraw=SimpleNamespace(Draw=RecordingDraw),np=np,W=1280,H=720,
              st=st,adir=str(SAMPLE/'moved'),arm='moved',brain='llama4-maverick',
              big=mod.font(26),mid=mod.font(20),small=mod.font(16),w=writer)
    try:
        exec(compile(ast.Module(body=[loop],type_ignores=[]),str(source),'exec'),ns)
    finally:
        writer.close()
    ns['card'].save(OUT/'canonical-card-layout.png')
    truth_y = next(xy[1] for text,xy in labels if text.startswith('truth:'))
    image_label_y = next(xy[1] for text,xy in labels if text.startswith('what it remembered:'))
    print('Canonical-card truth line y=',truth_y,'image label y=',image_label_y)
    assert truth_y+20 <= image_label_y, 'Canonical reply pushes truth text onto image labels'
