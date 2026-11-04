"""Record a command's actual output as a terminal video (adapted from the independent reviewer's recorder)."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from PIL import Image, ImageDraw, ImageFont

p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
p.add_argument('--cwd', required=True)
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
cmd = a.command[1:] if a.command[:1] == ['--'] else a.command
base = Path(a.out)
base.parent.mkdir(parents=True, exist_ok=True)
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 17)
enc = subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x800',
                        '-r','5','-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','25',
                        '-pix_fmt','yuv420p',str(base.with_suffix('.mp4'))], stdin=subprocess.PIPE)
lines = []
def frame(line, repeat=2):
    for start in range(0, max(len(line), 1), 117):
        lines.append(line[start:start+117])
    im = Image.new('RGB',(1280,800),(13,18,25)); d=ImageDraw.Draw(im)
    d.text((20,15),'Scout replay check | actual process output (GPU run, AWS)',font=font,fill=(130,210,255))
    d.text((20,42),'Playback paced for legibility. Real MuJoCo physics + Motivo policy on the GPU.',font=font,fill=(200,200,200))
    for i,l in enumerate(lines[-32:]): d.text((20,85+21*i),l,font=font,fill=(232,232,232))
    for _ in range(repeat): enc.stdin.write(im.tobytes())

t0=time.time(); started=datetime.datetime.now(datetime.timezone.utc).isoformat()
frame('$ '+shlex.join(cmd),5)
env=os.environ.copy(); env['PYTHONUNBUFFERED']='1'; env['PYTHONDONTWRITEBYTECODE']='1'; env['TMPDIR']='/dev/shm'
proc=subprocess.Popen(cmd,cwd=a.cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
with base.with_suffix('.txt').open('w') as f:
    f.write('Started '+started+'\nCommand '+shlex.join(cmd)+'\n')
    for line in proc.stdout:
        f.write(line); f.flush(); print(line,end='',flush=True); frame(line.rstrip())
    rc=proc.wait(); elapsed=time.time()-t0
    end=f'PROCESS EXIT {rc} | recording wall time {elapsed:.2f}s'
    f.write(end+'\n'); print(end,flush=True); frame(end,25)
enc.stdin.close(); enc_rc=enc.wait()
base.with_suffix('.json').write_text(json.dumps({'started':started,'cwd':a.cwd,'command':cmd,'exit_code':rc,
    'encoder_exit_code':enc_rc,'elapsed_s':elapsed,'video_kind':'paced actual terminal output, not physics'},indent=2))
raise SystemExit(rc if rc else enc_rc)
