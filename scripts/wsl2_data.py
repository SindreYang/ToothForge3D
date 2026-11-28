"""WSL2 数据回归的本地持久化入口，不覆盖原始数据或历史运行。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone


class ResultStore:
    """管理本地主机回归记录，已完成的数据库直接复用。"""

    SINDRE_SHA = "5c872863b8ae96464a8924356cac12337db1445c"

    def __init__(self, project: Path):
        self.project = project.resolve()
        self.root = self.project / "outputs" / "regression"
        self.runs = self.root / "runs"
        self.runs.mkdir(parents=True, exist_ok=True, mode=0o700)

    @staticmethod
    def read_completed(run: Path) -> dict | None:
        """读取完成标记，忽略未结束或无法解析的回归记录。"""
        for name in ("aggregate.json", "summary.json"):
            try:
                result = json.loads((run / name).read_text(encoding="utf-8"))
                if isinstance(result.get("passed"), bool):
                    return result
            except (OSError, ValueError, AttributeError):
                continue
        return None

    def archive_completed(self):
        """将同主机已完成缓存迁入持久目录，保留原路径，不重复复制 LMDB。"""
        legacy = Path.home() / ".cache" / "toothforge3d-regression"
        if legacy.is_dir():
            for run in sorted(legacy.iterdir()):
                if run.is_symlink() or not run.is_dir() or self.read_completed(run) is None:
                    continue
                target = self.runs / run.name
                if target.exists() or target.is_symlink():
                    continue
                # WSL 同盘原子重命名；跨盘时保留原记录，不做昂贵复制。
                try:
                    run.rename(target)
                except OSError:
                    print(f"Legacy run retained at {run}")
                    continue
                run.symlink_to(target, target_is_directory=True)
        self.publish()

    def publish(self):
        """分别更新最近完成、逐条数据通过和全量通过的稳定入口。"""
        completed = []
        for run in self.runs.iterdir():
            if not run.is_dir() or run.is_symlink():
                continue
            report = self.read_completed(run)
            if report is not None:
                stamp = run / "aggregate.json"
                if not stamp.is_file():
                    stamp = run / "summary.json"
                completed.append((stamp.stat().st_mtime_ns, run, report))
        completed.sort(key=lambda item: (item[0], item[1].name))
        for name in ("latest", "latest_checked", "latest_passed"):
            candidates = []
            for _, run, report in completed:
                data = report.get("preprocess", {})
                checked = (data.get("status") == "passed"
                           and data.get("all_records_checked") is True
                           and data.get("written", 0) > 0
                           and data.get("roundtrip_records_pass") == data.get("written")
                           and (run / "fdg_v3_full.db").exists())
                if (name == "latest" or (checked and name == "latest_checked")
                        or (checked and report.get("passed") is True and name == "latest_passed")):
                    candidates.append(run)
            if candidates:
                link = self.root / name
                if link.exists() and not link.is_symlink():
                    raise FileExistsError(f"Refusing to overwrite non-symlink: {link}")
                temporary = self.root / f".{name}-{os.getpid()}"
                temporary.unlink(missing_ok=True)
                temporary.symlink_to(Path("runs") / candidates[-1].name, target_is_directory=True)
                temporary.replace(link)
        print(f"Local regression records: {self.root}")

    def write_manifest(self, run: Path, database: Path, sindre_sha: str):
        """保存运行代码版本、原始库路径和环境，便于下次定位。"""
        code_root = Path(__file__).resolve().parents[1]
        sha = subprocess.check_output(
            ["git", "-C", str(code_root), "rev-parse", "HEAD"], text=True,
        ).strip()
        payload = {
            "code_commit": sha, "sindre_commit": sindre_sha,
            "source_database": str(database.resolve()),
            "python": sys.executable, "python_version": sys.version,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run.name, "fdg_resolution": 128,
        }
        (run / "manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def run(self, database: Path | None = None, *, refresh_cache: bool = False) -> int:
        """直接使用项目虚拟环境及持久依赖快照执行一轮真实回归。"""
        os.umask(0o077)
        self.archive_completed()
        database = database or self.project / "datasets" / "全牙_5.db"
        if not database.exists():
            raise FileNotFoundError(database)
        python = self.project / ".venv" / "bin" / "python"
        snapshot = self.root / "dependencies" / self.SINDRE_SHA
        marker = snapshot / ".source-sha"
        if not python.is_file():
            raise FileNotFoundError(python)
        if not marker.is_file() or marker.read_text().strip() != self.SINDRE_SHA:
            raise RuntimeError("缺少固定版本 SindrePy 快照，请先运行 WSL2 CI 的环境准备步骤")
        run = self.runs / datetime.now(timezone.utc).strftime("local-%Y%m%dT%H%M%S%fZ")
        run.mkdir(mode=0o700)
        self.write_manifest(run, database, self.SINDRE_SHA)
        env = dict(os.environ, TF_DB=str(database.resolve()), TF_RESULTS=str(run),
                   PYTHONPATH=os.pathsep.join((str(self.project), str(snapshot))),
                   MPLBACKEND="Agg", PYTHONUTF8="1", PYTHONUNBUFFERED="1",
                   SINDRE_SHA=self.SINDRE_SHA, TF_CASE_CACHE=str(self.root / "case_cache"),
                   TF_REFRESH_CACHE="1" if refresh_cache else "0")
        cmd = [str(python), "-m", "tests.run_wsl2_regression"]
        if not env.get("DISPLAY") and shutil.which("xvfb-run"):
            cmd = ["xvfb-run", "-a", *cmd]
        print(f"Run: {run}", flush=True)
        with (run / "real.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(cmd, cwd=self.project, env=env, stdout=log, stderr=subprocess.STDOUT)
        self.publish()
        return result.returncode


def main():
    store = ResultStore(Path(__file__).resolve().parents[1])
    return store.run()


if __name__ == "__main__":
    raise SystemExit(main())
