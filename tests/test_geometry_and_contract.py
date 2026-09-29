import copy
import zipfile

import numpy as np
import pytest
from PIL import Image

from cozmo.capture import Capture
from cozmo.cli import unpack
from cozmo.damage import surfaces
from cozmo.evaluation import evaluate, validate, validate_geometry
from cozmo.geometry import PlanFrame
from cozmo.plan import M, PlanOut, RoomOut, WallOut
from cozmo.viewer import write_viewer


def square_plan():
    vertices = [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]]
    walls = [
        WallOut(
            i,
            a,
            vertices[(i + 1) % 4],
            M(float(np.linalg.norm(np.array(a) - vertices[(i + 1) % 4])), 0.05),
            True,
        )
        for i, a in enumerate(vertices)
    ]
    room = RoomOut(
        1, "Test room", vertices, walls, M(12.0, 0.5, "m2"), M(14.0, 0.2), M(2.5, 0.1), []
    )
    plan = PlanOut("synthetic", "lidar", [room], [], M(12.0, 0.5, "m2"), {"units": "m"})
    surfaces(plan)
    return plan


@pytest.mark.parametrize("angle", [0.0, 0.5, 1.4])
def test_world_plan_round_trip(angle):
    p = np.array([[1.0, 2.0, 3.0], [-2.0, 5.0, 0.5]])
    frame = PlanFrame(angle, np.zeros(2), 0.02, (100, 100))
    np.testing.assert_allclose(frame.to_world_xz(frame.to_plan(p)), p[:, [0, 2]], atol=1e-12)


@pytest.mark.parametrize("new_format", [True, False])
def test_depth_units_intrinsics_and_sparse_frame_ids(tmp_path, new_format):
    (tmp_path / "depth").mkdir()
    (tmp_path / "confidence").mkdir()
    Image.fromarray(np.full((12, 16), 2000, np.uint16)).save(tmp_path / "depth/000007.png")
    Image.fromarray(np.full((12, 16), 2, np.uint8)).save(tmp_path / "confidence/000007.png")
    (tmp_path / "camera_matrix.csv").write_text("1200,0,960\n0,1200,720\n0,0,1\n")
    header = "timestamp,frame,x,y,z,qx,qy,qz,qw"
    row = "0,7,0,0,0,0,0,0,1"
    if new_format:
        header += ",fx,fy,cx,cy"
        row += ",1200,1200,960,720"
    (tmp_path / "odometry.csv").write_text(header + "\n" + row + "\n")
    capture = Capture(tmp_path)
    np.testing.assert_allclose(capture.depth(0), 2.0)
    np.testing.assert_allclose(capture.K_depth(0), [[10, 0, 8], [0, 10, 6], [0, 0, 1]])
    p, n = capture.points_world(0)
    np.testing.assert_allclose(p[:, 2], 2.0)
    assert np.all(n[:, 2] < -0.99)


def test_contract_and_geometric_consistency(tmp_path):
    plan = square_plan()
    path = tmp_path / "plan.json"
    plan.save(path)
    assert validate(path)["valid"]
    data = plan.to_json()
    data["rooms"][0]["walls"][0]["length"]["value"] = 7
    assert not validate_geometry(data)["valid"]


def test_overlaps_are_not_silently_accepted():
    data = square_plan().to_json()
    other = copy.deepcopy(data["rooms"][0])
    other["id"] = 2
    data["rooms"].append(other)
    assert validate_geometry(data)["overlaps"][0]["area_m2"] == 12.0


def test_opening_gate_counts_missed_and_phantom():
    plan = square_plan().to_json()
    opening = {
        "kind": "door",
        "wall": 0,
        "offset": M(0.5, 0.1).to_json(),
        "width": M(0.9, 0.05).to_json(),
        "height": M(2.0, 0.1).to_json(),
        "sill": None,
        "connects_to": None,
    }
    plan["rooms"][0]["openings"] = [opening, copy.deepcopy(opening)]
    truth = {
        "source": "laser",
        "openings_exhaustive": True,
        "measurements": {"room-1/opening-0/width": 0.9, "room-1/opening-2/width": 0.8},
    }
    report = evaluate(plan, truth)
    assert report["opening_gate"]["success_fraction"] == pytest.approx(1 / 3)
    assert not report["opening_gate"]["passes"]
    with pytest.raises(ValueError, match="ground truth"):
        evaluate(plan, {"source": "lidar", "measurements": truth["measurements"]})


def test_archive_traversal_rejected(tmp_path):
    archive = tmp_path / "capture.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../outside.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        unpack(archive, tmp_path / "work")
    assert not (tmp_path / "work/outside.txt").exists()


def test_viewer_does_not_interpret_capture_as_script(tmp_path):
    data = square_plan().to_json()
    data["capture"] = "</script><script>alert(1)</script>"
    path = tmp_path / "index.html"
    write_viewer(data, path)
    assert "</script><script>alert(1)" not in path.read_text()


def test_invalid_measurement_fails_and_physical_interval_is_nonnegative():
    assert M(0.01, 0.5).to_json()["ci95"][0] == 0
    with pytest.raises(ValueError):
        M(float("nan"), 0.1).to_json()


def test_fusion_cold_and_cached_results_are_identical(tmp_path):
    from cozmo.cache import fused_cloud

    class SyntheticCapture:
        fingerprint = "test"
        cache_tag = "raw"

        def fused_points(self, **kwargs):
            return np.array(
                [[0.123456789, 0.0, 0.0], [0.126345678, 0.0, 0.0], [1.0, 2.0, 3.0]]
            ), np.array([[0.0, 1.0, 0.0]] * 3)

    cold = fused_cloud(SyntheticCapture(), tmp_path)
    cached = fused_cloud(SyntheticCapture(), tmp_path)
    for a, b in zip(cold, cached, strict=True):
        np.testing.assert_array_equal(a, b)
