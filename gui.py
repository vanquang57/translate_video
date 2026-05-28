"""Simple tkinter GUI for Video Text Translator.

Usage:
    .venv\\Scripts\\python gui.py
    (or via run_gui.bat)

Features:
- File chooser for input video
- Progress bar
- Log output area
- Output saved next to input file
- Perturb Video button with preset selection dialog
- CLI still works independently via main.py
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# ---------------------------------------------------------------------------
# Bootstrap: ensure src/ is importable and .env is loaded
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Reuse the .env loader from main.py
from main import _load_dotenv  # noqa: E402

_load_dotenv(_PROJECT_ROOT / ".env")

from video_text_translator.config import build_config, deep_merge, load_yaml  # noqa: E402
from video_text_translator.detector import PaddleOCRDetector  # noqa: E402
from video_text_translator.inpainter import OpenCVInpainter  # noqa: E402
from video_text_translator.logging_config import setup_logging  # noqa: E402
from video_text_translator.perturbation_config import (  # noqa: E402
    PRESETS,
    PerturbationConfig,
    load_perturbation_config,
)
from video_text_translator.perturbation_pipeline import PerturbationPipeline  # noqa: E402
from video_text_translator.pipeline import Pipeline  # noqa: E402
from video_text_translator.renderer import PillowRenderer  # noqa: E402
from video_text_translator.tracker import IoUContentTracker  # noqa: E402
from video_text_translator.translator import GoogleTranslator  # noqa: E402
from video_text_translator.translator_llm import LlmTranslator  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom ProgressReporter that pushes updates to a queue
# ---------------------------------------------------------------------------


class GuiProgressReporter:
    """Progress reporter that sends updates to a thread-safe queue."""

    def __init__(self, msg_queue: queue.Queue) -> None:
        self._queue = msg_queue
        self._total = 0
        self._current = 0
        self._stage = ""

    def start(self, total: int, stage_name: str) -> None:
        self._total = total
        self._current = 0
        self._stage = stage_name
        self._queue.put(("stage", stage_name, total))

    def update(self, n: int = 1) -> None:
        self._current += n
        self._queue.put(("progress", self._current, self._total))

    def set_stage(self, name: str) -> None:
        self._stage = name
        self._queue.put(("stage", name, self._total))

    def set_info(self, key: str, value: str) -> None:
        """Send an info key-value pair to display on the GUI."""
        self._queue.put(("info", key, value))

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Custom log handler that sends records to a queue
# ---------------------------------------------------------------------------


class QueueLogHandler(logging.Handler):
    """Logging handler that puts formatted messages into a queue."""

    def __init__(self, msg_queue: queue.Queue) -> None:
        super().__init__()
        self._queue = msg_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._queue.put(("log", msg, None))
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# Main GUI Application
# ---------------------------------------------------------------------------


class TranslatorApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Video Text Translator")
        self.root.geometry("700x500")
        self.root.resizable(True, True)

        self._msg_queue: queue.Queue = queue.Queue()
        self._running = False

        self._build_ui()
        self._setup_logging()

    def _build_ui(self) -> None:
        # --- File selection frame ---
        file_frame = ttk.LabelFrame(self.root, text="Input Video", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        self._file_var = tk.StringVar()
        entry = ttk.Entry(file_frame, textvariable=self._file_var, state="readonly")
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        browse_btn = ttk.Button(file_frame, text="Browse...", command=self._browse)
        browse_btn.pack(side=tk.RIGHT)

        # --- Progress frame ---
        prog_frame = ttk.LabelFrame(self.root, text="Progress", padding=10)
        prog_frame.pack(fill=tk.X, padx=10, pady=5)

        self._stage_var = tk.StringVar(value="Idle")
        ttk.Label(prog_frame, textvariable=self._stage_var).pack(anchor=tk.W)

        self._info_var = tk.StringVar(value="")
        self._info_label = ttk.Label(
            prog_frame, textvariable=self._info_var, foreground="gray"
        )
        self._info_label.pack(anchor=tk.W)

        self._progress = ttk.Progressbar(prog_frame, mode="determinate")
        self._progress.pack(fill=tk.X, pady=(5, 0))

        # --- Log frame ---
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self._log_text = tk.Text(log_frame, height=12, state=tk.DISABLED, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=scrollbar.set)
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # --- Buttons ---
        btn_frame = ttk.Frame(self.root, padding=10)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self._start_btn = ttk.Button(btn_frame, text="Start", command=self._start)
        self._start_btn.pack(side=tk.LEFT, padx=(0, 5))

        self._perturb_btn = ttk.Button(
            btn_frame, text="Perturb Video", command=self._show_perturb_dialog,
            state=tk.DISABLED,
        )
        self._perturb_btn.pack(side=tk.LEFT, padx=(0, 5))

        self._quit_btn = ttk.Button(btn_frame, text="Quit", command=self._quit)
        self._quit_btn.pack(side=tk.RIGHT)

        # --- File variable trace: enable/disable perturb button ---
        self._file_var.trace_add("write", self._on_file_changed)

    def _setup_logging(self) -> None:
        setup_logging(verbose=False, quiet=False)
        handler = QueueLogHandler(self._msg_queue)
        handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        logging.getLogger().addHandler(handler)

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mkv *.mov *.webm"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._file_var.set(path)

    def _on_file_changed(self, *_args: object) -> None:
        """Enable/disable perturb button based on file selection."""
        if self._running:
            return
        if self._file_var.get():
            self._perturb_btn.config(state=tk.NORMAL)
        else:
            self._perturb_btn.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Perturbation dialog and execution
    # ------------------------------------------------------------------

    def _show_perturb_dialog(self) -> None:
        """Open a modal dialog for preset selection."""
        input_path = self._file_var.get()
        if not input_path:
            messagebox.showwarning("No file", "Please select an input video first.")
            return
        if self._running:
            messagebox.showinfo("Running", "A process is already in progress.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Perturbation Preset")
        dialog.geometry("420x380")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        # Preset selection
        ttk.Label(dialog, text="Select perturbation preset:", font=("", 10, "bold")).pack(
            anchor=tk.W, padx=15, pady=(15, 5)
        )

        preset_var = tk.StringVar(value="medium")

        presets_frame = ttk.Frame(dialog, padding=(15, 0))
        presets_frame.pack(fill=tk.X)

        for preset_name in ("light", "medium", "heavy"):
            ttk.Radiobutton(
                presets_frame, text=preset_name.capitalize(),
                variable=preset_var, value=preset_name,
                command=lambda: self._toggle_custom_fields(custom_frame, preset_var),
            ).pack(anchor=tk.W, pady=2)

        ttk.Radiobutton(
            presets_frame, text="Custom",
            variable=preset_var, value="custom",
            command=lambda: self._toggle_custom_fields(custom_frame, preset_var),
        ).pack(anchor=tk.W, pady=2)

        # Custom fields frame
        custom_frame = ttk.LabelFrame(dialog, text="Custom Parameters", padding=10)
        custom_frame.pack(fill=tk.X, padx=15, pady=(10, 5))

        # Custom parameter fields
        custom_fields: dict[str, tk.StringVar] = {}
        param_defs = [
            ("max_crop_percent", "Max Crop %", "5.0"),
            ("speed_min", "Speed Min", "0.99"),
            ("speed_max", "Speed Max", "1.01"),
            ("max_zoom", "Max Zoom", "1.05"),
            ("eq_range_db", "EQ Range (dB)", "2.0"),
            ("change_interval", "Change Interval (s)", "10.0"),
        ]

        for i, (key, label, default) in enumerate(param_defs):
            row = i // 2
            col = (i % 2) * 2
            ttk.Label(custom_frame, text=label).grid(
                row=row, column=col, sticky=tk.W, padx=(0, 5), pady=2
            )
            var = tk.StringVar(value=default)
            custom_fields[key] = var
            ttk.Entry(custom_frame, textvariable=var, width=10).grid(
                row=row, column=col + 1, sticky=tk.W, padx=(0, 15), pady=2
            )

        # Initially hide custom fields
        self._toggle_custom_fields(custom_frame, preset_var)

        # OK / Cancel buttons
        btn_frame = ttk.Frame(dialog, padding=(15, 10))
        btn_frame.pack(fill=tk.X)

        def on_ok() -> None:
            selected = preset_var.get()
            custom_params: dict[str, float] | None = None

            if selected == "custom":
                # Parse custom fields
                custom_params = {}
                for key, var in custom_fields.items():
                    try:
                        custom_params[key] = float(var.get())
                    except ValueError:
                        messagebox.showerror(
                            "Invalid value",
                            f"Invalid numeric value for '{key}': {var.get()}",
                            parent=dialog,
                        )
                        return
                # Use medium as base preset for custom
                selected = "medium"

            dialog.destroy()
            self._start_perturb(selected, custom_params)

        def on_cancel() -> None:
            dialog.destroy()

        ttk.Button(btn_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT)

        # Center dialog on parent
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")

    @staticmethod
    def _toggle_custom_fields(custom_frame: ttk.LabelFrame, preset_var: tk.StringVar) -> None:
        """Show/hide custom parameter fields based on preset selection."""
        if preset_var.get() == "custom":
            for child in custom_frame.winfo_children():
                child.configure(state="normal") if hasattr(child, "configure") else None
            custom_frame.pack(fill=tk.X, padx=15, pady=(10, 5))
        else:
            custom_frame.pack_forget()

    def _start_perturb(self, preset: str, custom_params: dict[str, float] | None) -> None:
        """Start perturbation in a background thread."""
        input_path = self._file_var.get()
        inp = Path(input_path)

        # Generate output path: {stem}_perturbed_{preset}{suffix}
        preset_label = "custom" if custom_params else preset
        output_path = str(
            inp.parent / f"{inp.stem}_perturbed_{preset_label}{inp.suffix}"
        )

        self._running = True
        self._start_btn.config(state=tk.DISABLED)
        self._perturb_btn.config(state=tk.DISABLED)
        self._clear_log()
        self._stage_var.set("Starting perturbation...")
        self._info_var.set("")
        self._progress["value"] = 0

        thread = threading.Thread(
            target=self._run_perturb_pipeline,
            args=(input_path, output_path, preset, custom_params),
            daemon=True,
        )
        thread.start()
        self._poll_queue()

    def _run_perturb_pipeline(
        self,
        input_path: str,
        output_path: str,
        preset: str,
        custom_params: dict[str, float] | None,
    ) -> None:
        """Run the perturbation pipeline in a background thread."""
        try:
            yaml_path = _PROJECT_ROOT / "configs" / "perturbation.yaml"

            # Build param overrides
            overrides: dict[str, object] = {
                "input_path": input_path,
                "output_path": output_path,
            }
            if custom_params:
                overrides.update(custom_params)

            config = load_perturbation_config(
                yaml_path=yaml_path,
                preset_override=preset,
                param_overrides=overrides,
            )

            progress = GuiProgressReporter(self._msg_queue)
            pipeline = PerturbationPipeline(config=config, progress=progress)
            exit_code = pipeline.run()

            if exit_code == 0:
                self._msg_queue.put(("done", output_path, None))
            else:
                self._msg_queue.put(
                    ("error", f"Perturbation pipeline exited with code {exit_code}", None)
                )

        except Exception as exc:
            self._msg_queue.put(("error", str(exc), None))

    def _start(self) -> None:
        input_path = self._file_var.get()
        if not input_path:
            messagebox.showwarning("No file", "Please select an input video first.")
            return
        if self._running:
            messagebox.showinfo("Running", "Translation is already in progress.")
            return

        # Output path: same folder, append _translated, auto-increment if exists
        inp = Path(input_path)
        output_path = self._unique_output_path(inp)

        self._running = True
        self._start_btn.config(state=tk.DISABLED)
        self._perturb_btn.config(state=tk.DISABLED)
        self._clear_log()
        self._stage_var.set("Starting...")
        self._info_var.set("")
        self._progress["value"] = 0

        thread = threading.Thread(
            target=self._run_pipeline,
            args=(input_path, output_path),
            daemon=True,
        )
        thread.start()
        self._poll_queue()

    def _run_pipeline(self, input_path: str, output_path: str) -> None:
        """Run the translation pipeline in a background thread."""
        try:
            import cv2

            yaml_path = _PROJECT_ROOT / "configs" / "default.yaml"
            yaml_dict = load_yaml(yaml_path) if yaml_path.is_file() else {}
            overrides = {"input_path": input_path, "output_path": output_path}
            merged = deep_merge(yaml_dict, overrides)
            config = build_config(merged)

            # Probe video
            cap = cv2.VideoCapture(config.input_path)
            if not cap.isOpened():
                self._msg_queue.put(("error", f"Cannot open video: {input_path}", None))
                return
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            detector = PaddleOCRDetector(
                compute_mode=config.compute_mode,
                confidence_threshold=config.detector.confidence_threshold,
                downscale=config.performance.ocr_downscale,
                model_variant=config.detector.model_variant,
                cpu_threads=config.detector.cpu_threads,
            )
            tracker = IoUContentTracker(
                frame_width=width,
                frame_height=height,
                iou_threshold=config.tracker.iou_threshold,
                content_similarity_threshold=config.tracker.content_similarity_threshold,
                center_distance_ratio=config.tracker.center_distance_ratio,
                n_inactive=config.tracker.n_inactive,
                ocr_stride=config.performance.ocr_stride,
                max_active_segments=config.tracker.max_active_segments,
                smooth_lock_threshold=config.tracker.smooth_lock_threshold,
                smooth_ema_alpha=config.tracker.smooth_ema_alpha,
            )
            inpainter = OpenCVInpainter(
                algorithm=config.inpainter.algorithm,
                radius=config.inpainter.radius,
                padding=config.inpainter.padding,
            )

            if config.translator.backend == "llm":
                translator = LlmTranslator(
                    config=config.translator.llm,
                    max_retries=config.translator.max_retries,
                )
            else:
                translator = GoogleTranslator(
                    timeout_seconds=config.translator.timeout_seconds,
                    max_chars=config.translator.max_chars,
                    max_retries=config.translator.max_retries,
                )

            renderer = PillowRenderer(default_font_path=config.renderer.font_path)
            progress = GuiProgressReporter(self._msg_queue)

            pipeline = Pipeline(
                config=config,
                detector=detector,
                tracker=tracker,
                inpainter=inpainter,
                translator=translator,
                renderer=renderer,
                progress=progress,
            )

            exit_code = pipeline.run()

            if exit_code == 0:
                self._msg_queue.put(("done", output_path, None))
            else:
                self._msg_queue.put(("error", f"Pipeline exited with code {exit_code}", None))

        except Exception as exc:
            self._msg_queue.put(("error", str(exc), None))

    def _poll_queue(self) -> None:
        """Process messages from the background thread."""
        try:
            while True:
                msg_type, data1, data2 = self._msg_queue.get_nowait()

                if msg_type == "progress":
                    current, total = data1, data2
                    if total > 0:
                        self._progress["maximum"] = total
                        self._progress["value"] = current

                elif msg_type == "stage":
                    stage_name, total = data1, data2
                    self._stage_var.set(stage_name)
                    if total > 0:
                        self._progress["maximum"] = total
                        self._progress["value"] = 0

                elif msg_type == "info":
                    key, value = data1, data2
                    self._info_var.set(f"{key}: {value}")

                elif msg_type == "log":
                    self._append_log(data1)

                elif msg_type == "done":
                    self._stage_var.set("Done!")
                    self._progress["value"] = self._progress["maximum"]
                    self._append_log(f"\n✓ Output saved: {data1}")
                    self._running = False
                    self._start_btn.config(state=tk.NORMAL)
                    self._perturb_btn.config(
                        state=tk.NORMAL if self._file_var.get() else tk.DISABLED
                    )
                    return

                elif msg_type == "error":
                    self._stage_var.set("Error")
                    self._append_log(f"\n✗ ERROR: {data1}")
                    self._running = False
                    self._start_btn.config(state=tk.NORMAL)
                    self._perturb_btn.config(
                        state=tk.NORMAL if self._file_var.get() else tk.DISABLED
                    )
                    return

        except queue.Empty:
            pass

        if self._running:
            self.root.after(100, self._poll_queue)

    def _append_log(self, text: str) -> None:
        self._log_text.config(state=tk.NORMAL)
        self._log_text.insert(tk.END, text + "\n")
        self._log_text.see(tk.END)
        self._log_text.config(state=tk.DISABLED)

    def _clear_log(self) -> None:
        self._log_text.config(state=tk.NORMAL)
        self._log_text.delete("1.0", tk.END)
        self._log_text.config(state=tk.DISABLED)

    @staticmethod
    def _unique_output_path(inp: Path) -> str:
        """Generate output path like name_translated.mp4, name_translated(1).mp4, etc."""
        candidate = inp.parent / f"{inp.stem}_translated{inp.suffix}"
        if not candidate.exists():
            return str(candidate)
        n = 1
        while True:
            candidate = inp.parent / f"{inp.stem}_translated({n}){inp.suffix}"
            if not candidate.exists():
                return str(candidate)
            n += 1

    def _quit(self) -> None:
        if self._running:
            if not messagebox.askyesno("Confirm", "Translation is running. Quit anyway?"):
                return
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    app = TranslatorApp()
    app.run()
