#!/usr/bin/env python3
"""
CT Pipeline GUI — tkinter interface with full descriptions and tooltips.
Run with any Python that has tkinter (system Python or conda base):
    python gui/pipeline_gui.py
"""
import os
import json
import subprocess
import sys
import time
import threading
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog

REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUTS    = REPO_ROOT / "outputs"
STL_DIR    = REPO_ROOT / "DATACREATION" / "STL"
STATE_FILE = OUTPUTS / "seq_pipeline_state.json"
SEQ_SCRIPT = REPO_ROOT / "scripts" / "sequential_train_pipeline.py"
FULL_SH    = REPO_ROOT / "run_full_pipeline.sh"
INFER_SCRIPT = REPO_ROOT / "scripts" / "pure_dl" / "03_inference.py"
CAD_EXPORT = REPO_ROOT / "ct_recon" / "volume_to_cad.py"
CONDA_ENV  = "ct_pipeline"

def conda_python():
    return [
        "bash", "-c",
        f'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate {CONDA_ENV} && exec python -u "$@"',
        "--"
    ]


# ──────────────────────────────────────────────────────────────────────
# Tooltip helper
# ──────────────────────────────────────────────────────────────────────
class ToolTip:
    """Hover tooltip for any tkinter widget."""
    def __init__(self, widget, text, delay=400):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tip_window = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)

    def _schedule(self, event=None):
        self._after_id = self.widget.after(self.delay, self._show)

    def _show(self):
        if self.tip_window:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", foreground="#333",
                         relief=tk.SOLID, borderwidth=1,
                         font=("TkDefaultFont", 10), wraplength=320,
                         padx=6, pady=4)
        label.pack()

    def _hide(self, event=None):
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CT Reconstruction Pipeline")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        self.proc = None
        
        self.infer_model_path = tk.StringVar(value="No model selected")
        self.infer_sample_dir = tk.StringVar(value="No projection folder selected")
        
        self._build()
        self._refresh_stl()
        self._toggle_ui()

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------
    def _build(self):
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=4)
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left panel with scrollbar
        left_outer = tk.Frame(paned, width=380)
        paned.add(left_outer, minsize=360)
        
        canvas = tk.Canvas(left_outer)
        scrollbar = ttk.Scrollbar(left_outer, orient="vertical", command=canvas.yview)
        left = tk.Frame(canvas)
        
        left.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=left, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        right = tk.Frame(paned)
        paned.add(right, minsize=400)

        self._build_left(left)
        self._build_right(right)

    def _build_left(self, parent):
        def section(text, description=None):
            tk.Label(parent, text=text, font=("TkDefaultFont", 10, "bold"),
                     anchor="w").pack(fill=tk.X, pady=(10, 0))
            ttk.Separator(parent).pack(fill=tk.X)
            if description:
                tk.Label(parent, text=description, fg="#666",
                         wraplength=340, justify=tk.LEFT, anchor="w",
                         font=("TkDefaultFont", 9)).pack(fill=tk.X, padx=4, pady=(2, 4))

        # ── Pipeline Mode ──
        section("Pipeline Mode",
                "Choose how to run the CT reconstruction pipeline.")
        self.mode = tk.StringVar(value="sequential")

        modes = [
            ("Sequential Training",
             "sequential",
             "Train the DL model on each STL file one-by-one.\n"
             "Best for limited disk space — cleans up after each STL."),
            ("Main Batch Pipeline",
             "main",
             "Runs the full pipeline shell script.\n"
             "Processes all datasets at once in batch mode."),
            ("Standalone Inference",
             "inference",
             "Load a trained model and reconstruct a 3D volume\n"
             "from raw projections. No training — inference only."),
        ]
        for label, value, tip in modes:
            rb = tk.Radiobutton(parent, text=label, variable=self.mode,
                                value=value, command=self._toggle_ui)
            rb.pack(anchor=tk.W, padx=8)
            ToolTip(rb, tip)

        # ── Dynamic Container ──
        self.dynamic_frame = tk.Frame(parent)
        self.dynamic_frame.pack(fill=tk.X, pady=5)
        
        # === Frame 1: Training Options ===
        self.train_frame = tk.Frame(self.dynamic_frame)
        
        tk.Label(self.train_frame, text="Training Options",
                 font=("TkDefaultFont", 10, "bold"), anchor="w").pack(fill=tk.X, pady=(8, 0))
        ttk.Separator(self.train_frame).pack(fill=tk.X)
        tk.Label(self.train_frame,
                 text="Configure training hyperparameters for the neural network.",
                 fg="#666", wraplength=340, justify=tk.LEFT, anchor="w",
                 font=("TkDefaultFont", 9)).pack(fill=tk.X, padx=4, pady=(2, 4))

        # Epochs
        row = tk.Frame(self.train_frame); row.pack(fill=tk.X, pady=1, padx=4)
        lbl = tk.Label(row, text="Epochs", width=14, anchor="w"); lbl.pack(side=tk.LEFT)
        ToolTip(lbl, "Number of complete passes through the training dataset.\nMore epochs = better quality but longer training time.\nRecommended: 30–50 for publishable results.")
        self.epochs = tk.StringVar(value="30")
        tk.Entry(row, textvariable=self.epochs, width=6).pack(side=tk.LEFT)

        # Batch size
        row2 = tk.Frame(self.train_frame); row2.pack(fill=tk.X, pady=1, padx=4)
        lbl2 = tk.Label(row2, text="Batch Size", width=14, anchor="w"); lbl2.pack(side=tk.LEFT)
        ToolTip(lbl2, "Number of sinogram slices processed simultaneously.\nLarger = faster training but more GPU memory.\nReduce to 1 if you get Out-Of-Memory errors.")
        self.batch_size = tk.StringVar(value="2")
        tk.Entry(row2, textvariable=self.batch_size, width=6).pack(side=tk.LEFT)

        # Downsample
        row_res = tk.Frame(self.train_frame); row_res.pack(fill=tk.X, pady=1, padx=4)
        lbl_ds = tk.Label(row_res, text="Downsample", width=14, anchor="w"); lbl_ds.pack(side=tk.LEFT)
        ToolTip(lbl_ds, "Factor to reduce projection resolution.\n2 = half resolution (faster), 1 = full resolution (slower).\nUse 2 for initial experiments, 1 for final training.")
        self.downsample = tk.StringVar(value="2")
        tk.Entry(row_res, textvariable=self.downsample, width=6).pack(side=tk.LEFT)

        # Image Size
        row_img = tk.Frame(self.train_frame); row_img.pack(fill=tk.X, pady=1, padx=4)
        lbl_img = tk.Label(row_img, text="Image Size", width=14, anchor="w"); lbl_img.pack(side=tk.LEFT)
        ToolTip(lbl_img, "Size of the reconstructed 2D CT slice (NxN pixels).\n256 = standard, 512 = high quality (needs more VRAM).")
        self.image_size = tk.StringVar(value="256")
        tk.Entry(row_img, textvariable=self.image_size, width=6).pack(side=tk.LEFT)

        # Scan method
        row3 = tk.Frame(self.train_frame); row3.pack(fill=tk.X, pady=1, padx=4)
        lbl_scan = tk.Label(row3, text="Scan Geometry", width=14, anchor="w"); lbl_scan.pack(side=tk.LEFT)
        ToolTip(lbl_scan, "How the object was positioned during X-ray scanning.\n• auto: detect automatically from settings\n• centered: object at rotation center\n• offset: object displaced from center")
        self.scan_method = tk.StringVar(value="auto")
        ttk.Combobox(row3, textvariable=self.scan_method,
                     values=["auto", "centered", "offset"], width=8,
                     state="readonly").pack(side=tk.LEFT)

        # Checkboxes
        self.flag_dry    = tk.BooleanVar()
        self.flag_infer  = tk.BooleanVar()
        self.flag_quick  = tk.BooleanVar()
        
        cb1 = tk.Checkbutton(self.train_frame, text="Dry Run (preview commands only)", variable=self.flag_dry)
        cb1.pack(anchor=tk.W, padx=8)
        ToolTip(cb1, "Shows the commands that would be executed\nwithout actually running them. Good for verification.")
        
        cb2 = tk.Checkbutton(self.train_frame, text="Run Inference After Training", variable=self.flag_infer)
        cb2.pack(anchor=tk.W, padx=8)
        ToolTip(cb2, "Automatically run 3D volume reconstruction\nafter training completes for each STL model.")
        
        cb3 = tk.Checkbutton(self.train_frame, text="Quick Test (1 STL, 2 epochs)", variable=self.flag_quick)
        cb3.pack(anchor=tk.W, padx=8)
        ToolTip(cb3, "Rapid smoke test with minimal training.\nUse this to verify everything works before a full run.")
        
        # === Frame 2: Inference Options ===
        self.infer_frame = tk.Frame(self.dynamic_frame)
        tk.Label(self.infer_frame, text="Inference Configuration",
                 font=("TkDefaultFont", 10, "bold"), anchor="w").pack(fill=tk.X, pady=(8, 0))
        ttk.Separator(self.infer_frame).pack(fill=tk.X)
        tk.Label(self.infer_frame,
                 text="Select a trained model and input projections\n"
                      "to reconstruct a 3D CT volume.",
                 fg="#666", wraplength=340, justify=tk.LEFT, anchor="w",
                 font=("TkDefaultFont", 9)).pack(fill=tk.X, padx=4, pady=(2, 4))
        
        btn_model = tk.Button(self.infer_frame, text="📂 Select Model Checkpoint (.pt)",
                              command=self._pick_model, width=30)
        btn_model.pack(pady=4)
        ToolTip(btn_model, "Choose a trained PyTorch model file (.pt)\nfrom the outputs directory.")
        tk.Label(self.infer_frame, textvariable=self.infer_model_path,
                 fg="gray", wraplength=320).pack(fill=tk.X, pady=(0,8))

        btn_proj = tk.Button(self.infer_frame, text="📂 Select Input Projection Folder",
                             command=self._pick_sample, width=30)
        btn_proj.pack(pady=4)
        ToolTip(btn_proj, "Choose a folder containing X-ray projections\n(TIFF files + settings.cto configuration).")
        tk.Label(self.infer_frame, textvariable=self.infer_sample_dir,
                 fg="gray", wraplength=320).pack(fill=tk.X)
        
        row_inf = tk.Frame(self.infer_frame); row_inf.pack(fill=tk.X, pady=4, padx=4)
        lbl_inf = tk.Label(row_inf, text="Batch Size", width=14, anchor="w"); lbl_inf.pack(side=tk.LEFT)
        ToolTip(lbl_inf, "Number of slices to process simultaneously during inference.\nHigher = faster, but uses more GPU memory.")
        self.infer_batch = tk.StringVar(value="8")
        tk.Entry(row_inf, textvariable=self.infer_batch, width=6).pack(side=tk.LEFT)

        # === Frame 3: Batch Pipeline Info ===
        self.main_info_frame = tk.Frame(self.dynamic_frame)
        tk.Label(self.main_info_frame, text="Batch Pipeline",
                 font=("TkDefaultFont", 10, "bold"), anchor="w").pack(fill=tk.X, pady=(8, 0))
        ttk.Separator(self.main_info_frame).pack(fill=tk.X)
        tk.Label(self.main_info_frame,
                 text="Executes the full pipeline script (run_full_pipeline.sh).\n\n"
                      "This processes ALL datasets in one batch:\n"
                      "  1. Simulate projections (gVXR)\n"
                      "  2. Build training datasets\n"
                      "  3. Train the neural network\n"
                      "  4. Run 3D volume inference\n\n"
                      "⚠️ This does not accept custom parameters.",
                 fg="#555", justify=tk.LEFT, wraplength=320).pack(pady=10, fill=tk.X)

        # ── STL Status ──
        self.stl_section = tk.Frame(parent)
        tk.Label(self.stl_section, text="STL Model Status",
                 font=("TkDefaultFont", 10, "bold"), anchor="w").pack(fill=tk.X, pady=(8, 0))
        ttk.Separator(self.stl_section).pack(fill=tk.X)
        tk.Label(self.stl_section,
                 text="Training progress for each CAD model in DATACREATION/STL/.",
                 fg="#666", wraplength=340, justify=tk.LEFT, anchor="w",
                 font=("TkDefaultFont", 9)).pack(fill=tk.X, padx=4, pady=(2, 2))
        self.stl_frame = tk.Frame(self.stl_section)
        self.stl_frame.pack(fill=tk.X)
        btn_refresh = tk.Button(self.stl_section, text="↻ Refresh STL Status",
                                command=self._refresh_stl)
        btn_refresh.pack(anchor=tk.W, pady=2, padx=4)
        ToolTip(btn_refresh, "Rescan the STL folder and update which models\nhave completed training.")
        
        self.stl_section.pack(fill=tk.X)

        # ── Pipeline Controls ──
        section("Pipeline Controls", "Start or stop the selected pipeline mode.")
        
        self.run_btn = tk.Button(parent, text="▶  Start Pipeline",
                                 command=self._run, width=26,
                                 bg="#2d7a2d", fg="white",
                                 activebackground="#3d9a3d")
        self.run_btn.pack(pady=4)
        ToolTip(self.run_btn, "Launches the selected pipeline mode.\n"
                "• Sequential: trains on STL models one-by-one\n"
                "• Batch: runs the full pipeline shell script\n"
                "• Inference: reconstructs 3D volume from projections")
        
        self.stop_btn = tk.Button(parent, text="■  Stop Pipeline",
                                  command=self._stop, state=tk.DISABLED, width=26,
                                  bg="#a03030", fg="white",
                                  activebackground="#c04040")
        self.stop_btn.pack(pady=2)
        ToolTip(self.stop_btn, "Terminates the currently running pipeline process.")

        # ── Results & Export ──
        section("Results & Export",
                "View reconstruction results or export to CAD format.")
        
        btn_dl = tk.Button(parent, text="📂 Open DL Volume",
                           command=self._open_dl, width=26)
        btn_dl.pack(pady=2)
        ToolTip(btn_dl, "Opens the DL-reconstructed 3D volume (.tif)\nin your system's default viewer.")
        
        btn_fdk = tk.Button(parent, text="📂 Open FDK Reference Volume",
                            command=self._open_fdk, width=26)
        btn_fdk.pack(pady=2)
        ToolTip(btn_fdk, "Opens the classical FDK reconstruction volume\nfor visual quality comparison.")
        
        btn_cad = tk.Button(parent, text="📦 Export 3D CAD File (STL/OBJ)",
                            command=self._export_cad, width=26)
        btn_cad.pack(pady=2)
        ToolTip(btn_cad, "Converts the DL reconstruction into a 3D mesh file\n"
                "(STL/OBJ) that can be opened in FreeCAD, SolidWorks,\n"
                "AutoCAD, or 3D Slicer.")

        # ── Training State Management ──
        self.state_section = tk.Frame(parent)
        tk.Label(self.state_section, text="Training State",
                 font=("TkDefaultFont", 10, "bold"), anchor="w").pack(fill=tk.X, pady=(8, 0))
        ttk.Separator(self.state_section).pack(fill=tk.X)
        tk.Label(self.state_section,
                 text="View or reset the sequential training progress tracker.",
                 fg="#666", wraplength=340, justify=tk.LEFT, anchor="w",
                 font=("TkDefaultFont", 9)).pack(fill=tk.X, padx=4, pady=(2, 4))
        
        btn_state = tk.Button(self.state_section, text="📋 View Training Progress",
                              command=self._view_state, width=26)
        btn_state.pack(pady=2)
        ToolTip(btn_state, "Shows which STL models have been trained\nand their completion status (JSON state file).")
        
        btn_reset = tk.Button(self.state_section, text="🔄 Reset All Training",
                              command=self._reset_state, width=26)
        btn_reset.pack(pady=2)
        ToolTip(btn_reset, "⚠️ WARNING: Deletes all training progress!\n"
                "All STL models will be re-trained from scratch\non the next pipeline run.")
        self.state_section.pack(fill=tk.X)

    def _build_right(self, parent):
        hdr = tk.Frame(parent)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="Pipeline Log Output",
                 font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)
        btn_clear = tk.Button(hdr, text="Clear Log", command=self._clear_log)
        btn_clear.pack(side=tk.RIGHT)
        ToolTip(btn_clear, "Clears all text from the log console.")

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
    # Dynamic UI toggle
    # ------------------------------------------------------------------
    def _toggle_ui(self):
        m = self.mode.get()
        
        self.train_frame.pack_forget()
        self.infer_frame.pack_forget()
        self.main_info_frame.pack_forget()
        self.stl_section.pack_forget()
        self.state_section.pack_forget()
        
        if m == "inference":
            self.infer_frame.pack(fill=tk.X)
        elif m == "sequential":
            self.train_frame.pack(fill=tk.X)
            self.stl_section.pack(fill=tk.X)
            self.state_section.pack(fill=tk.X)
        elif m == "main":
            self.main_info_frame.pack(fill=tk.X, pady=10)

    # ------------------------------------------------------------------
    # File Pickers
    # ------------------------------------------------------------------
    def _pick_model(self):
        start_dir = OUTPUTS / "pure_dl_training_centered"
        if not start_dir.exists(): start_dir = REPO_ROOT
        f = filedialog.askopenfilename(initialdir=start_dir, title="Select Trained Model Checkpoint",
                                       filetypes=[("PyTorch Model", "*.pt")])
        if f:
            self.infer_model_path.set(f)
            
    def _pick_sample(self):
        start_dir = REPO_ROOT / "data"
        if not start_dir.exists(): start_dir = REPO_ROOT
        d = filedialog.askdirectory(initialdir=start_dir,
                                    title="Select Projection Folder (must contain settings.cto)")
        if d:
            self.infer_sample_dir.set(d)

    # ------------------------------------------------------------------
    # STL status list
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
            tk.Label(self.stl_frame, text="  (no STL files found in DATACREATION/STL/)",
                     fg="gray").pack(anchor=tk.W)
            return
        for stl in stls:
            done   = stl.name in completed
            symbol = "✓ Trained" if done else "○ Pending"
            color  = "#2d7a2d" if done else "#888"
            tk.Label(self.stl_frame, text=f"  {symbol}  —  {stl.name}",
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
        m = self.mode.get()
        if m == "inference":
            mp = self.infer_model_path.get()
            sd = self.infer_sample_dir.get()
            if not Path(mp).exists():
                messagebox.showerror("Error", "Please select a valid model checkpoint (.pt file).")
                return None
            if not Path(sd).exists():
                messagebox.showerror("Error", "Please select a valid input projection folder\n"
                                     "(must contain settings.cto).")
                return None
            
            out_path = OUTPUTS / "dl_reconstruction" / "dl_volume.tif"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            cmd = conda_python() + [
                str(INFER_SCRIPT),
                "--model-path", mp,
                "--sample-dir", sd,
                "--output-path", str(out_path),
                "--batch-size", self.infer_batch.get()
            ]
            return cmd
            
        elif m == "main":
            return ["bash", str(FULL_SH)]

        else: # sequential
            cmd = conda_python() + [str(SEQ_SCRIPT),
                  "--epochs",      self.epochs.get(),
                  "--batch-size",  self.batch_size.get(),
                  "--downsample",  self.downsample.get(),
                  "--image-size",  self.image_size.get(),
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
        if cmd is None: return
        
        self._log(f"$ {' '.join(str(c) for c in cmd)}\n\n", "info")
        self.run_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)

        def worker():
            try:
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                
                if shutil.which("nvidia-smi") is not None:
                    env["__NV_PRIME_RENDER_OFFLOAD"] = "1"
                    env["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
                
                self.proc = subprocess.Popen(
                    [str(c) for c in cmd],
                    cwd=str(REPO_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env
                )
                
                os.set_blocking(self.proc.stdout.fileno(), False)
                
                while True:
                    line = self.proc.stdout.readline()
                    if line:
                        tag = "err" if any(w in line.lower() for w in ("error", "traceback", "failed")) \
                              else "ok" if any(w in line for w in ("✅", "Complete", "saved", "OK", "✓")) \
                              else None
                        self.root.after(0, self._log, line, tag)
                    else:
                        if self.proc.poll() is not None:
                            while True:
                                remainder = self.proc.stdout.readline()
                                if not remainder: break
                                self.root.after(0, self._log, remainder, None)
                            break
                        time.sleep(0.05)
                        
                rc = self.proc.wait()
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
            self._log("\n[Pipeline stopped by user]\n", "err")

    def _done(self):
        self.proc = None
        self.run_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        if self.mode.get() != "inference":
            self._refresh_stl()

    # ------------------------------------------------------------------
    # Open results
    # ------------------------------------------------------------------
    def _open(self, path: Path):
        if not path.exists():
            messagebox.showinfo("File Not Found",
                                f"The file does not exist yet:\n{path}\n\n"
                                "Run inference first to generate the reconstruction.")
            return
        subprocess.Popen(["xdg-open", str(path)])

    def _open_dl(self):
        self._open(OUTPUTS / "dl_reconstruction" / "dl_volume.tif")

    def _open_fdk(self):
        hits = sorted(OUTPUTS.rglob("fdk_volume.tif"))
        if not hits:
            messagebox.showinfo("Not Found", "No FDK volume found in outputs/.\n"
                                "Run the classical FDK reconstruction first.")
            return
        if len(hits) == 1:
            self._open(hits[0])
            return
        win = tk.Toplevel(self.root)
        win.title("Select FDK Volume")
        tk.Label(win, text="Multiple FDK volumes found. Click to open:").pack(padx=8, pady=4)
        for h in hits:
            label = str(h.relative_to(REPO_ROOT))
            tk.Button(win, text=label, anchor="w",
                      command=lambda p=h: [self._open(p), win.destroy()]).pack(fill=tk.X, padx=8, pady=1)

    # ------------------------------------------------------------------
    # CAD Export
    # ------------------------------------------------------------------
    def _export_cad(self):
        """Export the DL reconstruction volume to STL/OBJ for CAD software."""
        dl_volume = OUTPUTS / "dl_reconstruction" / "dl_volume.tif"
        if not dl_volume.exists():
            messagebox.showinfo("No Volume Found",
                                "No DL reconstruction volume found.\n\n"
                                "Run inference first to generate dl_volume.tif,\n"
                                "then use this button to export it as STL/OBJ.")
            return
        
        # Ask user for output path
        out_path = filedialog.asksaveasfilename(
            initialdir=OUTPUTS / "dl_reconstruction",
            title="Save CAD Mesh As",
            defaultextension=".stl",
            filetypes=[("STL Mesh", "*.stl"), ("OBJ Mesh", "*.obj"), ("All Files", "*.*")]
        )
        if not out_path:
            return
        
        self._log(f"\n[CAD Export] Converting volume to mesh: {out_path}\n", "info")
        
        cmd = conda_python() + [
            str(CAD_EXPORT),
            str(dl_volume),
            "--output-stl", out_path
        ]
        
        # Run in the same worker thread pattern
        self.run_btn.configure(state=tk.DISABLED)
        
        def worker():
            try:
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                proc = subprocess.Popen(
                    [str(c) for c in cmd], cwd=str(REPO_ROOT),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, env=env
                )
                out, _ = proc.communicate()
                if out:
                    self.root.after(0, self._log, out, "ok" if proc.returncode == 0 else "err")
                tag = "ok" if proc.returncode == 0 else "err"
                msg = f"\n[CAD Export] {'Complete!' if proc.returncode == 0 else 'Failed.'}\n"
                self.root.after(0, self._log, msg, tag)
            except Exception as e:
                self.root.after(0, self._log, f"\n[CAD Export ERROR] {e}\n", "err")
            finally:
                self.root.after(0, lambda: self.run_btn.configure(state=tk.NORMAL))
        
        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def _view_state(self):
        if not STATE_FILE.exists():
            messagebox.showinfo("No Training State",
                                "No training state file found.\n"
                                "Run the sequential pipeline first to generate one.")
            return
        win = tk.Toplevel(self.root)
        win.title("Training Progress State")
        txt = scrolledtext.ScrolledText(win, width=56, height=16,
                                        font=("Courier", 9))
        txt.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        txt.insert(tk.END, STATE_FILE.read_text())
        txt.configure(state=tk.DISABLED)

    def _reset_state(self):
        if not messagebox.askyesno(
            "⚠️ Reset All Training?",
            "This will delete ALL training progress.\n"
            "Every STL model will be re-trained from scratch.\n\n"
            "This cannot be undone. Continue?"
        ):
            return
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        self._refresh_stl()
        self._log("[State reset — all STL models marked as pending]\n", "info")


# ------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()

    # Enable High DPI scaling for Linux/Windows
    try:
        root.tk.call('tk', 'scaling', 2.0)
    except Exception:
        pass

    # Increase default font size for readability
    from tkinter import font
    default_font = font.nametofont("TkDefaultFont")
    default_font.configure(size=12)
    text_font = font.nametofont("TkTextFont")
    text_font.configure(size=12)
    fixed_font = font.nametofont("TkFixedFont")
    fixed_font.configure(size=12)

    App(root)
    root.mainloop()
