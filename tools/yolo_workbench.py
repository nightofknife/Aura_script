from __future__ import annotations

import argparse
import os
import queue
import threading
from collections import deque
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from tools._shared import discover_plan_names, sanitize_filename
from tools.yolo_project_lib import (
    YoloProjectError,
    activate_model,
    check_environment,
    complete_labelimg_session,
    create_labelimg_session,
    create_project,
    export_training_dataset,
    import_images,
    launch_labelimg,
    list_projects,
    load_project_config,
    load_samples,
    project_display_payload,
    project_root,
    run_worker_subprocess,
    summarize_samples,
    update_after_relabel,
    update_after_train,
)


class YoloWorkbenchApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("YOLO Workbench")
        self.root.geometry("1180x780")

        self.plan_var = tk.StringVar()
        self.project_var = tk.StringVar()
        self.worker_status_var = tk.StringVar(value="Idle")
        self.summary_var = tk.StringVar(value="No project selected.")
        self.model_var = tk.StringVar(value="-")
        self.classes_var = tk.StringVar(value="-")
        self.latest_train_var = tk.StringVar(value="-")
        self.latest_relabel_var = tk.StringVar(value="-")
        self.env_var = tk.StringVar(value="-")
        self.log_lines = deque(maxlen=200)
        self.worker_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.worker_running = False

        self._build_ui()
        self.refresh_plan_choices()
        self.root.after(100, self._poll_worker_queue)

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill="x")

        ttk.Label(top, text="Plan").grid(row=0, column=0, sticky="w")
        self.plan_box = ttk.Combobox(top, textvariable=self.plan_var, state="readonly", width=24)
        self.plan_box.grid(row=0, column=1, sticky="we", padx=(6, 12))
        self.plan_box.bind("<<ComboboxSelected>>", lambda _e: self.on_plan_changed())

        ttk.Label(top, text="Project").grid(row=0, column=2, sticky="w")
        self.project_box = ttk.Combobox(top, textvariable=self.project_var, state="readonly", width=28)
        self.project_box.grid(row=0, column=3, sticky="we", padx=(6, 12))
        self.project_box.bind("<<ComboboxSelected>>", lambda _e: self.refresh_project_view())

        ttk.Button(top, text="Refresh", command=self.refresh_project_view).grid(row=0, column=4, sticky="ew")

        button_bar = ttk.Frame(self.root, padding=(12, 0))
        button_bar.pack(fill="x")
        buttons = [
            ("Create Project", self.create_project_dialog),
            ("Import Images", self.import_images_action),
            ("Open Manual Label Session", self.open_manual_session),
            ("Complete Manual Session", self.complete_manual_session),
            ("Train Model", self.train_model),
            ("Relabel Unlabeled", self.relabel_unlabeled),
            ("Open Draft Review Session", self.open_review_session),
            ("Complete Draft Review", self.complete_review_session),
            ("Activate Model", self.activate_model_action),
            ("Open Project Folder", self.open_project_folder),
            ("Copy Model Path", self.copy_model_path),
        ]
        for index, (label, callback) in enumerate(buttons):
            ttk.Button(button_bar, text=label, command=callback).grid(row=index // 4, column=index % 4, sticky="ew", padx=4, pady=4)

        info = ttk.LabelFrame(self.root, text="Project Summary", padding=12)
        info.pack(fill="x", padx=12, pady=(8, 8))
        ttk.Label(info, textvariable=self.summary_var, justify="left").grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(info, text="Classes").grid(row=1, column=0, sticky="nw", pady=(8, 0))
        ttk.Label(info, textvariable=self.classes_var, justify="left").grid(row=1, column=1, sticky="w", pady=(8, 0), padx=(6, 24))
        ttk.Label(info, text="Active Model").grid(row=1, column=2, sticky="nw", pady=(8, 0))
        ttk.Label(info, textvariable=self.model_var, justify="left").grid(row=1, column=3, sticky="w", pady=(8, 0))
        ttk.Label(info, text="Latest Train").grid(row=2, column=0, sticky="nw", pady=(8, 0))
        ttk.Label(info, textvariable=self.latest_train_var, justify="left").grid(row=2, column=1, sticky="w", pady=(8, 0), padx=(6, 24))
        ttk.Label(info, text="Latest Relabel").grid(row=2, column=2, sticky="nw", pady=(8, 0))
        ttk.Label(info, textvariable=self.latest_relabel_var, justify="left").grid(row=2, column=3, sticky="w", pady=(8, 0))
        ttk.Label(info, text="Environment").grid(row=3, column=0, sticky="nw", pady=(8, 0))
        ttk.Label(info, textvariable=self.env_var, justify="left").grid(row=3, column=1, columnspan=3, sticky="w", pady=(8, 0))

        log_frame = ttk.LabelFrame(self.root, text="Worker Log", padding=12)
        log_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        status_row = ttk.Frame(log_frame)
        status_row.pack(fill="x")
        ttk.Label(status_row, text="Worker Status").pack(side="left")
        ttk.Label(status_row, textvariable=self.worker_status_var).pack(side="left", padx=(8, 0))

        self.log_text = tk.Text(log_frame, height=24, wrap="word", font=("Consolas", 10))
        self.log_text.pack(fill="both", expand=True, pady=(8, 0))
        self.log_text.configure(state="disabled")

    def refresh_plan_choices(self) -> None:
        plans = discover_plan_names()
        self.plan_box["values"] = plans
        if not self.plan_var.get() and plans:
            self.plan_var.set(plans[0])
        self.on_plan_changed()

    def on_plan_changed(self) -> None:
        plan_name = self.plan_var.get().strip()
        projects = list_projects(plan_name) if plan_name else []
        self.project_box["values"] = projects
        if projects:
            if self.project_var.get() not in projects:
                self.project_var.set(projects[0])
        else:
            self.project_var.set("")
        self.refresh_project_view()

    def current_project_dir(self) -> Path | None:
        plan_name = self.plan_var.get().strip()
        project_name = self.project_var.get().strip()
        if not plan_name or not project_name:
            return None
        return project_root(plan_name, project_name)

    def refresh_project_view(self) -> None:
        project_dir = self.current_project_dir()
        if project_dir is None or not project_dir.is_dir():
            self.summary_var.set("No project selected.")
            self.classes_var.set("-")
            self.model_var.set("-")
            self.latest_train_var.set("-")
            self.latest_relabel_var.set("-")
            self.env_var.set(self._render_env())
            return

        payload = project_display_payload(project_dir)
        project = payload["project"]
        sample_summary = payload["sample_summary"]
        self.summary_var.set(
            f"Plan={project.get('plan_name')}  Project={project.get('project_name')}\n"
            f"Samples: unlabeled={sample_summary.get('unlabeled', 0)}  "
            f"draft={sample_summary.get('draft_generated', 0)}  "
            f"approved={sample_summary.get('approved', 0)}  "
            f"manual={sample_summary.get('in_manual_session', 0)}  "
            f"review={sample_summary.get('in_review_session', 0)}"
        )
        self.classes_var.set(", ".join(project.get("class_names") or []) or "-")
        self.model_var.set(str(payload.get("current_model_path") or "-"))
        latest_train = project.get("latest_train_run") or {}
        latest_relabel = project.get("latest_relabel_run") or {}
        self.latest_train_var.set(
            "-"
            if not latest_train
            else f"run={latest_train.get('run_id')} model={latest_train.get('best_model_path')}"
        )
        self.latest_relabel_var.set(
            "-"
            if not latest_relabel
            else f"run={latest_relabel.get('run_id')} drafts={latest_relabel.get('draft_count')}"
        )
        self.env_var.set(self._render_env())

    def _render_env(self) -> str:
        env = check_environment()
        parts = [
            f"python={'ok' if env.runtime_python else 'missing'}",
            f"labelImg={'ok' if env.labelimg_command else 'missing'}",
            f"ultralytics={'ok' if env.has_ultralytics else 'missing'}",
            f"torch={'ok' if env.has_torch else 'missing'}",
        ]
        if env.torch_cuda is not None:
            parts.append(f"cuda={'yes' if env.torch_cuda else 'no'}")
        if env.messages:
            parts.append("messages=" + "; ".join(env.messages))
        return " | ".join(parts)

    def create_project_dialog(self) -> None:
        plan_name = self.plan_var.get().strip()
        if not plan_name:
            messagebox.showwarning("Missing plan", "Please choose a plan first.", parent=self.root)
            return
        project_name = simpledialog.askstring("Create Project", "Project name:", parent=self.root)
        if not project_name:
            return
        classes_raw = simpledialog.askstring(
            "Create Project",
            "Classes (comma or newline separated):",
            parent=self.root,
        )
        if not classes_raw:
            messagebox.showwarning("Missing classes", "At least one class is required.", parent=self.root)
            return
        class_names = [item.strip() for item in classes_raw.replace("\n", ",").split(",") if item.strip()]
        try:
            create_project(plan_name=plan_name, project_name=project_name, class_names=class_names)
        except Exception as exc:
            messagebox.showerror("Create failed", str(exc), parent=self.root)
            return
        self.on_plan_changed()
        self.project_var.set(sanitize_filename(project_name, fallback="project"))
        self.refresh_project_view()

    def import_images_action(self) -> None:
        project_dir = self.current_project_dir()
        if project_dir is None:
            messagebox.showwarning("No project", "Please select a project first.", parent=self.root)
            return
        file_paths = filedialog.askopenfilenames(
            parent=self.root,
            title="Import Images",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp")],
        )
        if not file_paths:
            return
        try:
            result = import_images(project_dir, [Path(path) for path in file_paths])
        except Exception as exc:
            messagebox.showerror("Import failed", str(exc), parent=self.root)
            return
        messagebox.showinfo(
            "Import complete",
            f"Imported: {result['imported_count']}\nSkipped: {len(result['skipped'])}",
            parent=self.root,
        )
        self.refresh_project_view()

    def open_manual_session(self) -> None:
        self._open_label_session("manual")

    def open_review_session(self) -> None:
        self._open_label_session("review")

    def _open_label_session(self, session_type: str) -> None:
        project_dir = self.current_project_dir()
        if project_dir is None:
            messagebox.showwarning("No project", "Please select a project first.", parent=self.root)
            return
        env = check_environment()
        if env.labelimg_command is None:
            messagebox.showerror("LabelImg missing", "LabelImg was not found in config, .venv, or PATH.", parent=self.root)
            return
        batch_size = simpledialog.askinteger(
            "Batch Size",
            f"{'Manual' if session_type == 'manual' else 'Review'} batch size:",
            parent=self.root,
            minvalue=1,
            initialvalue=50,
        )
        if batch_size is None:
            return
        try:
            session = create_labelimg_session(project_dir, session_type=session_type, batch_size=batch_size)
            launched = launch_labelimg(project_dir, session)
        except Exception as exc:
            messagebox.showerror("Session failed", str(exc), parent=self.root)
            return
        messagebox.showinfo(
            "Session opened",
            f"Session: {session.session_id}\nLabelImg pid: {launched['pid']}",
            parent=self.root,
        )
        self.refresh_project_view()

    def complete_manual_session(self) -> None:
        self._complete_label_session("manual")

    def complete_review_session(self) -> None:
        self._complete_label_session("review")

    def _complete_label_session(self, session_type: str) -> None:
        project_dir = self.current_project_dir()
        if project_dir is None:
            messagebox.showwarning("No project", "Please select a project first.", parent=self.root)
            return
        try:
            result = complete_labelimg_session(project_dir, session_type=session_type)
        except Exception as exc:
            messagebox.showerror("Complete failed", str(exc), parent=self.root)
            return
        messagebox.showinfo(
            "Session completed",
            f"Approved: {result['approved_count']}\nReverted: {result['reverted_count']}",
            parent=self.root,
        )
        self.refresh_project_view()

    def train_model(self) -> None:
        project_dir = self.current_project_dir()
        if project_dir is None:
            messagebox.showwarning("No project", "Please select a project first.", parent=self.root)
            return
        env = check_environment()
        if not env.runtime_python or not env.has_ultralytics or not env.has_torch:
            messagebox.showerror("Environment not ready", self._render_env(), parent=self.root)
            return
        try:
            samples = load_samples(project_dir)
            if summarize_samples(samples).get("approved", 0) <= 0:
                raise YoloProjectError("There are no approved samples available for training.")
            export_info = export_training_dataset(project_dir)
        except Exception as exc:
            messagebox.showerror("Training blocked", str(exc), parent=self.root)
            return
        self._start_worker(
            worker_kind="train",
            script_path=REPO_ROOT / "tools" / "yolo_train_worker.py",
            args=["--project-root", str(project_dir), "--export-dir", export_info["export_dir"]],
        )

    def relabel_unlabeled(self) -> None:
        project_dir = self.current_project_dir()
        if project_dir is None:
            messagebox.showwarning("No project", "Please select a project first.", parent=self.root)
            return
        env = check_environment()
        if not env.runtime_python or not env.has_ultralytics or not env.has_torch:
            messagebox.showerror("Environment not ready", self._render_env(), parent=self.root)
            return
        try:
            project = load_project_config(project_dir)
            model_path = project.get("active_model_path")
            if not model_path:
                raise YoloProjectError("Current project has no active model.")
            if summarize_samples(load_samples(project_dir)).get("unlabeled", 0) <= 0:
                raise YoloProjectError("There are no unlabeled samples to relabel.")
        except Exception as exc:
            messagebox.showerror("Relabel blocked", str(exc), parent=self.root)
            return
        self._start_worker(
            worker_kind="relabel",
            script_path=REPO_ROOT / "tools" / "yolo_relabel_worker.py",
            args=["--project-root", str(project_dir)],
        )

    def activate_model_action(self) -> None:
        project_dir = self.current_project_dir()
        if project_dir is None:
            messagebox.showwarning("No project", "Please select a project first.", parent=self.root)
            return
        file_path = filedialog.askopenfilename(
            parent=self.root,
            title="Choose YOLO model",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
            initialdir=str(project_dir),
        )
        if not file_path:
            return
        try:
            result = activate_model(project_dir, Path(file_path))
        except Exception as exc:
            messagebox.showerror("Activate failed", str(exc), parent=self.root)
            return
        self.refresh_project_view()
        messagebox.showinfo("Model activated", f"Active model: {result['active_model_path']}", parent=self.root)

    def open_project_folder(self) -> None:
        project_dir = self.current_project_dir()
        if project_dir is None:
            messagebox.showwarning("No project", "Please select a project first.", parent=self.root)
            return
        try:
            if hasattr(os, "startfile"):
                os.startfile(str(project_dir))  # type: ignore[attr-defined]
            else:
                raise YoloProjectError("Opening folders is only supported on Windows for this tool.")
        except Exception as exc:
            messagebox.showerror("Open failed", str(exc), parent=self.root)

    def copy_model_path(self) -> None:
        project_dir = self.current_project_dir()
        if project_dir is None:
            messagebox.showwarning("No project", "Please select a project first.", parent=self.root)
            return
        payload = project_display_payload(project_dir)
        model_path = str(payload.get("current_model_path") or "")
        self.root.clipboard_clear()
        self.root.clipboard_append(model_path)
        messagebox.showinfo("Copied", f"Copied model path:\n{model_path}", parent=self.root)

    def _start_worker(self, *, worker_kind: str, script_path: Path, args: list[str]) -> None:
        if self.worker_running:
            messagebox.showwarning("Worker running", "Please wait for the current worker to finish.", parent=self.root)
            return
        project_dir = self.current_project_dir()
        if project_dir is None:
            messagebox.showwarning("No project", "Please select a project first.", parent=self.root)
            return

        self.worker_running = True
        self.worker_status_var.set(f"Running {worker_kind}...")
        self._append_log_line(f"[workbench] starting {worker_kind}: {script_path.name}")

        def worker_thread() -> None:
            try:
                result = run_worker_subprocess(
                    script_path=script_path,
                    args=args,
                    on_line=lambda line: self.worker_queue.put(("log", line)),
                )
                self.worker_queue.put(("success", {"kind": worker_kind, "result": result, "project_dir": str(project_dir)}))
            except Exception as exc:
                self.worker_queue.put(("error", {"kind": worker_kind, "error": str(exc)}))
            finally:
                self.worker_queue.put(("done", {"kind": worker_kind}))

        thread = threading.Thread(target=worker_thread, daemon=True)
        thread.start()

    def _poll_worker_queue(self) -> None:
        while True:
            try:
                kind, payload = self.worker_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self._append_log_line(str(payload))
            elif kind == "success":
                self._handle_worker_success(payload)
            elif kind == "error":
                self._handle_worker_error(payload)
            elif kind == "done":
                self.worker_running = False
                self.worker_status_var.set("Idle")
        self.root.after(100, self._poll_worker_queue)

    def _handle_worker_success(self, payload: dict[str, Any]) -> None:
        kind = payload["kind"]
        result = payload["result"]
        project_dir = Path(payload["project_dir"])
        if kind == "train":
            update_after_train(project_dir, result)
        elif kind == "relabel":
            update_after_relabel(project_dir, result)
        self._append_log_line(f"[workbench] {kind} completed successfully.")
        self.refresh_project_view()
        messagebox.showinfo("Worker complete", f"{kind} completed successfully.", parent=self.root)

    def _handle_worker_error(self, payload: dict[str, Any]) -> None:
        kind = payload["kind"]
        error = payload["error"]
        self._append_log_line(f"[workbench] {kind} failed: {error}")
        messagebox.showerror("Worker failed", f"{kind} failed:\n{error}", parent=self.root)

    def _append_log_line(self, line: str) -> None:
        self.log_lines.append(str(line))
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, "\n".join(self.log_lines))
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO data-engineering workbench for Aura plans.")
    return parser


def run_cli(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    root = tk.Tk()
    app = YoloWorkbenchApp(root)
    root.mainloop()
    return 0


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
