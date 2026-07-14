#!/usr/bin/env python3
"""
CT Pipeline GUI — minimal tkinter interface.
Run with any Python that has tkinter (system Python or conda base):
    python gui/pipeline_gui.py
"""
import json
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUTS    = REPO_ROOT / "outputs"
STL_DIR    = REPO_ROOT / "DATACREATION" / "STL"
STATE_FILE = OUTPUTS / "seq_pipeline_state.json"
SEQ_SCRIPT = REPO_ROOT / "scripts" / "sequential_train_pipeline.py"
FULL_SH    = REPO_ROOT / "run_full_pipeline.sh"
INFER_SCRIPT = REPO_ROOT / "scripts" / "pure_dl" / "03_inference.py"
CONDA_ENV  = "ct_pipeline"

# Use conda run so pipeline scripts get all ML packages
def conda_python():
    return ["conda", "run", "-n", CONDA_ENV, "python"]


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CT Pipeline")
        self.root.geometry("1000x640")
        self.root.resizable(True, True)
        self.proc = None
        self._build()
        self._refresh_stl()

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------
    def _build(self):
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=4)
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left = tk.Frame(paned, width=270)
        paned.add(left, minsize=240)

        right = tk.Frame(paned)
        paned.add(right, minsize=400)

        self._build_left(left)
        self._build_right(right)

    def _build_left(self, parent):
        def section(text):
            tk.Label(parent, text=text, font=("TkDefaultFont", 9, "bold"),
                     anchor="w").pack(fill=tk.X, pady=(8, 0))
            ttk.Separator(parent).pack(fill=tk.X)

        # ---- Pipeline mode ----
        section("Pipeline Mode")
        self.mode = tk.StringVar(value="sequential")
        tk.Radiobutton(parent, text="Sequential  (one STL at a time)",
                       variable=self.mode, value="sequential").pack(anchor=tk.W)
        tk.Radiobutton(parent, text="Main  (run_full_pipeline.sh)",
                       variable=self.mode, value="main").pack(anchor=tk.W)

        # ---- STL Status ----
        section("STL Status")
        self.stl_frame = tk.Frame(parent)
        self.stl_frame.pack(fill=tk.X)
        tk.Button(parent, text="↻ Refresh", command=self._refresh_stl).pack(anchor=tk.W, pady=2)

        # ---- Options ----
        section("Options")

        row = tk.Frame(parent); row.pack(fill=tk.X, pady=1)
        tk.Label(row, text="Epochs", width=10, anchor="w").pack(side=tk.LEFT)
        self.epochs = tk.StringVar(value="8")
        tk.Entry(row, textvariable=self.epochs, width=6).pack(side=tk.LEFT)

        row2 = tk.Frame(parent); row2.pack(fill=tk.X, pady=1)
        tk.Label(row2, text="Batch size", width=10, anchor="w").pack(side=tk.LEFT)
        self.batch_size = tk.StringVar(value="2")
        tk.Entry(row2, textvariable=self.batch_size, width=6).pack(side=tk.LEFT)

        row3 = tk.Frame(parent); row3.pack(fill=tk.X, pady=1)
        tk.Label(row3, text="Scan method", width=10, anchor="w").pack(side=tk.LEFT)
        self.scan_method = tk.StringVar(value="auto")
        ttk.Combobox(row3, textvariable=self.scan_method,
                     values=["auto", "centered", "offset"], width=8,
                     state="readonly").pack(side=tk.LEFT)

        self.flag_dry    = tk.BooleanVar()
        self.flag_infer  = tk.BooleanVar()
        self.flag_quick  = tk.BooleanVar()
        tk.Checkbutton(parent, text="--dry-run  (preview, no execution)",
                       variable=self.flag_dry).pack(anchor=tk.W)
        tk.Checkbutton(parent, text="--run-inference  (after each STL)",
                       variable=self.flag_infer).pack(anchor=tk.W)
        tk.Checkbutton(parent, text="--quick-test  (1 STL · 2 epochs)",
                       variable=self.flag_quick).pack(anchor=tk.W)

        # ---- Run / Stop ----
        section("Run")
        self.run_btn = tk.Button(parent, text="▶  Run Pipeline",
                                 command=self._run, width=24)
        self.run_btn.pack(pady=3)
        self.stop_btn = tk.Button(parent, text="■  Stop",
                                  command=self._stop, state=tk.DISABLED, width=24)
        self.stop_btn.pack(pady=2)

        # ---- Results ----
        section("Results")
        tk.Button(parent, text="Open DL Reconstruction",
                  command=self._open_dl, width=24).pack(pady=2)
        tk.Button(parent, text="Open FDK Reconstruction",
                  command=self._open_fdk, width=24).pack(pady=2)

        # ---- State management ----
        section("State")
        tk.Button(parent, text="View Pipeline State",
                  command=self._view_state, width=24).pack(pady=2)
        tk.Button(parent, text="Reset State  (retrain all)",
                  command=self._reset_state, width=24).pack(pady=2)

    def _build_right(self, parent):
        hdr = tk.Frame(parent)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="Log Output", font=("TkDefaultFont", 9, "bold")).pack(side=tk.LEFT)
        tk.Button(hdr, text="Clear", command=self._clear_log).pack(side=tk.RIGHT)

        self.log = scrolledtext.ScrolledText(
            parent, state=tk.DISABLED,
            font=("Courier", 9), wrap=tk.WORD,
            bg="#1e1e1e", fg="#d4d4d4",
            insertbackground="white",
        )
        self.log.pack(fill=tk.BOTH, expand=True, pady=(2, 0))

        # colour tags
        self.log.tag_config("ok",   foreground="#4ec9b0")
        self.log.tag_config("err",  foreground="#f48771")
        self.log.tag_config("info", foreground="#9cdcfe")

    # ------------------------------------------------------------------
    # STL list
    # ------------------------------------------------------------------
    def _refresh_stl(self):
        for w in self.stl_frame.winfo_children():
            w.destroy()
        stls = sorted(STL_DIR.glob("*.stl"))
        completed = set()
        if STATE_FILE.exists():
            try:
                completed = set(json.loads(STATE_FILE.read_text()).get("completed_stls", []))
            except Exception:
                pass
        if not stls:
            tk.Label(self.stl_frame, text="  (none found)", fg="gray").pack(anchor=tk.W)
            return
        for stl in stls:
            done   = stl.name in completed
            symbol = "✓" if done else "○"
            color  = "#2d7a2d" if done else "#555"
            tk.Label(self.stl_frame, text=f"  {symbol}  {stl.name}",
                     fg=color, anchor="w").pack(fill=tk.X)

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------
    def _log(self, text, tag=None):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text, tag or "")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _clear_log(self):
        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Build command
    # ------------------------------------------------------------------
    def _build_cmd(self):
        if self.mode.get() == "main":
            return ["bash", str(FULL_SH)]

        cmd = conda_python() + [str(SEQ_SCRIPT),
              "--epochs",      self.epochs.get(),
              "--batch-size",  self.batch_size.get(),
              "--scan-method", self.scan_method.get()]
        if self.flag_dry.get():   cmd.append("--dry-run")
        if self.flag_infer.get(): cmd.append("--run-inference")
        if self.flag_quick.get(): cmd.append("--quick-test")
        return cmd

    # ------------------------------------------------------------------
    # Run / Stop
    # ------------------------------------------------------------------
    def _run(self):
        cmd = self._build_cmd()
        self._log(f"$ {' '.join(str(c) for c in cmd)}\n\n", "info")
        self.run_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)

        def worker():
            try:
                self.proc = subprocess.Popen(
                    [str(c) for c in cmd],
                    cwd=str(REPO_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                for line in self.proc.stdout:
                    tag = "err" if any(w in line.lower() for w in ("error", "traceback", "failed")) \
                          else "ok" if any(w in line for w in ("✅", "Complete", "saved", "OK")) \
                          else None
                    self.root.after(0, self._log, line, tag)
                self.proc.wait()
                rc = self.proc.returncode
                tag = "ok" if rc == 0 else "err"
                self.root.after(0, self._log, f"\n─── exit code {rc} ───\n", tag)
            except Exception as e:
                self.root.after(0, self._log, f"\n[ERROR] {e}\n", "err")
            finally:
                self.root.after(0, self._done)

        threading.Thread(target=worker, daemon=True).start()

    def _stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self._log("\n[Stopped by user]\n", "err")

    def _done(self):
        self.proc = None
        self.run_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self._refresh_stl()

    # ------------------------------------------------------------------
    # Open results
    # ------------------------------------------------------------------
    def _open(self, path: Path):
        if not path.exists():
            messagebox.showinfo("Not found", str(path))
            return
        subprocess.Popen(["xdg-open", str(path)])

    def _open_dl(self):
        self._open(OUTPUTS / "dl_reconstruction" / "dl_volume.tif")

    def _open_fdk(self):
        hits = sorted(OUTPUTS.rglob("fdk_volume.tif"))
        if not hits:
            messagebox.showinfo("Not found", "No fdk_volume.tif found in outputs/")
            return
        # Let user pick if multiple
        if len(hits) == 1:
            self._open(hits[0])
            return
        win = tk.Toplevel(self.root)
        win.title("Choose FDK volume")
        tk.Label(win, text="Multiple FDK volumes found. Click to open:").pack(padx=8, pady=4)
        for h in hits:
            label = str(h.relative_to(REPO_ROOT))
            tk.Button(win, text=label, anchor="w",
                      command=lambda p=h: [self._open(p), win.destroy()]).pack(fill=tk.X, padx=8, pady=1)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def _view_state(self):
        if not STATE_FILE.exists():
            messagebox.showinfo("No state", "State file not found.\nRun the pipeline first.")
            return
        win = tk.Toplevel(self.root)
        win.title("Pipeline State")
        txt = scrolledtext.ScrolledText(win, width=56, height=16,
                                        font=("Courier", 9))
        txt.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        txt.insert(tk.END, STATE_FILE.read_text())
        txt.configure(state=tk.DISABLED)

    def _reset_state(self):
        if not messagebox.askyesno(
            "Reset state?",
            "This marks all STLs as untrained.\n"
            "Next run will retrain from scratch.\n\nContinue?"
        ):
            return
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        self._refresh_stl()
        self._log("[State reset — all STLs marked as pending]\n", "info")


# ------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
