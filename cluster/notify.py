#!/usr/bin/env python3
"""Email updates for FrMedQA cluster jobs.

  python cluster/notify.py test                       # send a test mail
  python cluster/notify.py send --subject "..." [--tail 40]
  python cluster/notify.py heartbeat --job NAME       # progress mail every HEARTBEAT_HOURS

Settings come from .env (loaded by cluster/common.sh): MAIL_TO, SMTP_HOST,
SMTP_PORT, SMTP_USER, SMTP_PASS, MAIL_FROM, HEARTBEAT_HOURS, STALL_HOURS.
Delivery order: SMTP account → local `sendmail` → local `mail`. Never raises:
a mail problem must not kill a training job.
"""
import argparse, getpass, json, os, re, shutil, smtplib, socket, subprocess, sys, time
from email.message import EmailMessage

START = time.time()


def _log_path():
    return os.environ.get("JOB_LOG", "")


def log_tail(n=40):
    """Last n meaningful lines of the job log (progress-bar redraws collapsed)."""
    p = _log_path()
    if not p or not os.path.exists(p):
        return "(no log yet)"
    with open(p, "rb") as f:
        f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 200_000))
        text = f.read().decode("utf-8", "replace")
    lines = []
    for raw in text.split("\n"):
        seg = [s for s in raw.split("\r") if s.strip()]
        if seg:
            lines.append(seg[-1].rstrip())
    out = []
    for ln in lines:                       # drop consecutive progress-bar duplicates
        key = re.sub(r"\d+", "#", ln)[:60]
        if out and re.sub(r"\d+", "#", out[-1])[:60] == key and "%|" in ln:
            out[-1] = ln
        else:
            out.append(ln)
    return "\n".join(out[-n:])


def progress():
    """Current notebook and cell, from the markers written by run_nb.sbatch / papermill."""
    p = _log_path()
    if not p or not os.path.exists(p):
        return "starting"
    with open(p, "rb") as f:
        f.seek(0, 2); f.seek(max(0, f.tell() - 2_000_000))
        text = f.read().decode("utf-8", "replace")
    cells = re.findall(r"Executing Cell (\d+)", text)
    nb = "?"
    try:   # markers sit at the start of each notebook's output: scan the whole file
        out = subprocess.run(["grep", "-a", "-o", "NOTEBOOK [^ ]* (", p], capture_output=True, text=True, timeout=60).stdout
        hits = re.findall(r"NOTEBOOK (\S+) \(", out)
        nb = hits[-1] if hits else "?"
    except Exception:
        pass
    total = "?"
    try:
        path = os.path.join(os.environ.get("PROJECT_DIR", "."), "notebooks", nb + ".ipynb")
        total = len(json.load(open(path))["cells"])
    except Exception:
        pass
    return f"{nb} · cell {cells[-1] if cells else '?'}/{total}"


def training_progress():
    """Step / loss / ETA from the newest trainer checkpoint written during this job.

    Notebook training bars render as HTML, so batch logs stay silent while a
    model trains; checkpoints (every 10% of training) carry the real progress."""
    import glob
    base = os.environ.get("FRMEDQA_BASE", "")
    states = []
    for fp in glob.glob(os.path.join(base, "MODEL_TRAINING", "*", "checkpoint-*", "trainer_state.json")):
        try:
            mt = os.path.getmtime(fp)
            if mt < START - 60:                      # older run: ignore
                continue
            j = json.load(open(fp))
            states.append((mt, os.path.dirname(os.path.dirname(fp)), j))
        except Exception:
            pass
    if not states:
        return ""
    states.sort(key=lambda t: t[0])
    mt, run_dir, j = states[-1]
    step, total = j.get("global_step", 0), j.get("max_steps", 0) or 0
    losses = [h["loss"] for h in j.get("log_history", []) if "loss" in h]
    txt = f"{os.path.basename(run_dir)}: step {step}/{total}"
    if total:
        txt += f" ({100 * step / total:.0f}%)"
    if losses:
        txt += f", loss {losses[-1]:.3f}"
    same = [(m, d["global_step"]) for m, r, d in states if r == run_dir and "global_step" in d]
    if len(same) >= 2 and total:
        (m0, s0), (m1, s1) = same[-2], same[-1]
        if s1 > s0 and m1 > m0:
            sec_per_step = (m1 - m0) / (s1 - s0)
            txt += f", ~{(total - step) * sec_per_step / 86400:.1f} days left"
    return txt


def _gpu_ids():
    """GPU index(es) SLURM gave this job; nvidia-smi on this cluster sees all GPUs of the node."""
    ids = os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS") or os.environ.get("CUDA_VISIBLE_DEVICES") or ""
    ids = ",".join(x for x in ids.split(",") if x.strip().isdigit())
    return ["-i", ids] if ids else []


def gpu_util():
    """Mean GPU utilisation (%) over a few samples, or None if unavailable."""
    vals = []
    for _ in range(3):
        try:
            out = subprocess.run(["nvidia-smi", *_gpu_ids(), "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=20).stdout.split()
            vals += [float(v) for v in out if v.strip().replace(".", "", 1).isdigit()]
        except Exception:
            return None
        time.sleep(5)
    return sum(vals) / len(vals) if vals else None


def gpu():
    try:
        q = subprocess.run(["nvidia-smi", *_gpu_ids(), "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        return q or "n/a"
    except Exception:
        return "n/a"


def hours(sec):
    return f"{sec / 3600:.1f} h"


def send(subject, body):
    to = os.environ.get("MAIL_TO", "").strip()
    if not to:
        print("[notify] MAIL_TO not set; skipping mail:", subject); return False
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["To"] = to
    msg["From"] = (os.environ.get("MAIL_FROM") or os.environ.get("SMTP_USER")
                   or f"{getpass.getuser()}@{socket.gethostname()}")
    msg.set_content(body)
    host = os.environ.get("SMTP_HOST", "").strip()
    if host:
        port = int(os.environ.get("SMTP_PORT", "587"))
        try:
            s = smtplib.SMTP_SSL(host, port, timeout=30) if port == 465 else smtplib.SMTP(host, port, timeout=30)
            if port not in (25, 465):
                s.starttls()
            if os.environ.get("SMTP_USER"):
                s.login(os.environ["SMTP_USER"], os.environ.get("SMTP_PASS", ""))
            s.send_message(msg); s.quit()
            print("[notify] sent via SMTP:", subject); return True
        except Exception as e:
            print(f"[notify] SMTP failed ({e}); trying local mail tools")
    if shutil.which("sendmail"):
        r = subprocess.run(["sendmail", "-t"], input=msg.as_bytes())
        if r.returncode == 0:
            print("[notify] sent via sendmail:", subject); return True
    if shutil.which("mail"):
        r = subprocess.run(["mail", "-s", subject, to], input=body.encode())
        if r.returncode == 0:
            print("[notify] sent via mail:", subject); return True
    print("[notify] no working mail route; subject was:", subject)
    return False


def header():
    return (f"job      : {os.environ.get('SLURM_JOB_NAME', '?')} (id {os.environ.get('SLURM_JOB_ID', '?')})\n"
            f"node     : {socket.gethostname()}\n"
            f"progress : {progress()}\n"
            + (f"training : {training_progress()}\n" if training_progress() else "") +
f"GPU      : {gpu()}\n"
            f"run dir  : {os.environ.get('FRMEDQA_BASE', '?')}\n"
            f"log      : {_log_path()}\n")


def cmd_send(a):
    body = header() + f"\n── last lines of the log ──\n{log_tail(a.tail)}\n"
    send(a.subject, body)


def cmd_heartbeat(a):
    every = float(os.environ.get("HEARTBEAT_HOURS", "6")) * 3600
    stall = float(os.environ.get("STALL_HOURS", "3")) * 3600
    last_beat, warned = time.time(), False
    while True:
        time.sleep(300)
        p = _log_path()
        idle = time.time() - os.path.getmtime(p) if p and os.path.exists(p) else 0
        if idle > stall and not warned:
            util = gpu_util()
            busy = util is not None and util >= 10
            if not busy:                         # silent log + idle GPU = probably stuck
                send(f"[frmedqa] ⚠ {a.job}: no log output for {hours(idle)}, GPU {util if util is not None else '?'}% busy",
                     header() + f"\nThe log has not changed for {hours(idle)} and the GPU is idle. "
                     f"The job may be stuck.\nCheck with: bash cluster/status.sh\n\n── last lines ──\n{log_tail(30)}\n")
                warned = True
        if idle < 600:
            warned = False
        if time.time() - last_beat >= every:
            tp = training_progress()
            send(f"[frmedqa] {a.job} · {hours(time.time() - START)} · {tp.split(': ', 1)[-1] if tp else progress()}",
                 header() + f"\n── last lines of the log ──\n{log_tail(30)}\n")
            last_beat = time.time()


def cmd_test(a):
    ok = send("[frmedqa] test mail from the cluster",
              f"If you read this, job emails work.\n\nhost: {socket.gethostname()}\n"
              f"route: {'SMTP ' + os.environ.get('SMTP_HOST', '') if os.environ.get('SMTP_HOST') else 'local mail tools'}\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("send"); s.add_argument("--subject", required=True); s.add_argument("--tail", type=int, default=40)
    h = sub.add_parser("heartbeat"); h.add_argument("--job", default=os.environ.get("SLURM_JOB_NAME", "job"))
    sub.add_parser("test")
    a = ap.parse_args()
    {"send": cmd_send, "heartbeat": cmd_heartbeat, "test": cmd_test}[a.cmd](a)
