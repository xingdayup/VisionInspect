"""Sample and reclaim the complete inference process tree."""
import psutil


def resident_tree_memory(process: psutil.Process,tracked=None) -> int:
    """Current tree RSS, excluding lifetime peaks from earlier resident jobs."""
    try:members=[process]+process.children(recursive=True)
    except psutil.NoSuchProcess:return 0
    total=0
    for member in members:
        try:
            if tracked is not None:tracked[member.pid]=member
            total+=member.memory_info().rss
        except psutil.NoSuchProcess:pass
    return total


def tree_memory(process: psutil.Process, peaks: dict[int,int]) -> int:
    try:
        members=[process]+process.children(recursive=True)
    except psutil.NoSuchProcess:
        return sum(peaks.values())
    for member in members:
        try:
            info=member.memory_info()
            peaks[member.pid]=max(peaks.get(member.pid,0),info.rss,getattr(info,'peak_wset',0))
        except psutil.NoSuchProcess:
            pass
    return sum(peaks.values())


def kill_tree(process: psutil.Process):
    try:
        members=process.children(recursive=True)+[process]
    except psutil.NoSuchProcess:
        return
    for member in reversed(members):
        try:
            member.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(members,timeout=5)
