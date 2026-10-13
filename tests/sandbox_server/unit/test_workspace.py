import pytest

from sandbox_server.workspace import WorkspaceError, WorkspaceManager


@pytest.fixture
def wm(tmp_path):
    return WorkspaceManager(tmp_path / "ws")


def test_create(wm):
    ws = wm.create("sess1", "trial1")
    assert ws.is_dir()
    assert not any(ws.iterdir())


def test_create_duplicate_fails(wm):
    wm.create("sess1", "trial1")
    with pytest.raises(FileExistsError):
        wm.create("sess1", "trial1")


def test_resolve_valid(wm):
    ws = wm.create("s", "t")
    path = wm.resolve(ws, "foo/bar.txt")
    assert str(path).startswith(str(ws))


def test_resolve_blocks_absolute(wm):
    ws = wm.create("s", "t")
    with pytest.raises(WorkspaceError, match="absolute"):
        wm.resolve(ws, "/etc/passwd")


def test_resolve_blocks_traversal(wm):
    ws = wm.create("s", "t")
    with pytest.raises(WorkspaceError, match="escapes"):
        wm.resolve(ws, "../../etc/passwd")


def test_resolve_blocks_symlink_escape(wm):
    ws = wm.create("s", "t")
    link = ws / "link"
    link.symlink_to("/etc")
    with pytest.raises(WorkspaceError):
        wm.resolve(ws, "link/passwd")


def test_resolve_blocks_intermediate_symlink(wm):
    ws = wm.create("s", "t")
    subdir = ws / "real"
    subdir.mkdir()
    (subdir / "file.txt").write_text("data")
    link = ws / "linked_dir"
    link.symlink_to(subdir)
    with pytest.raises(WorkspaceError, match="symlink"):
        wm.resolve(ws, "linked_dir/file.txt")


def test_destroy(wm):
    ws = wm.create("s", "t")
    assert ws.is_dir()
    wm.destroy(ws)
    assert not ws.exists()
    assert not (wm.root / "s").exists()


def test_destroy_keeps_non_empty_session_dir(wm):
    ws1 = wm.create("s", "t1")
    ws2 = wm.create("s", "t2")
    wm.destroy(ws1)

    assert not ws1.exists()
    assert ws2.is_dir()
    assert (wm.root / "s").is_dir()


def test_destroy_outside_root_rejected(wm, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(WorkspaceError, match="outside"):
        wm.destroy(outside)


def test_destroy_idempotent(wm):
    ws = wm.create("s", "t")
    wm.destroy(ws)
    wm.destroy(ws)  # should not raise
