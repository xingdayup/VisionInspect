import subprocess
import sys

import psutil

from visioninspect.resources import kill_tree, resident_tree_memory, tree_memory


def test_windows_launcher_child_memory_is_included():
    command = [
        sys.executable,
        "-B",
        "-u",
        "-c",
        'import time; allocation=bytearray(40*1024*1024); print("ready",flush=True); time.sleep(30)',
    ]
    child = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    monitored = psutil.Process(child.pid)
    try:
        assert child.stdout.readline().strip() == "ready"
        peaks = {}
        memory = tree_memory(monitored, peaks)
        assert memory > 40 * 1024 * 1024
        assert child.pid in peaks
    finally:
        kill_tree(monitored)
        child.wait(timeout=10)
        child.stdout.close()
    assert child.poll() is not None


def test_resident_limit_uses_current_rss_instead_of_previous_job_peak():
    from types import SimpleNamespace

    class ProcessFixture:
        def children(self, recursive):
            return []

        def memory_info(self):
            return SimpleNamespace(rss=1234, peak_wset=20 * 1024**3)

    assert resident_tree_memory(ProcessFixture()) == 1234
