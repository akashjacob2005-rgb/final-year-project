"""Graph pathway tests: graph construction, numpy GAT, validation and the /api/graph routes.

No real data is used or needed. Small generated arrays (10 regions, 60 time
points) and randomly initialised weights exist ONLY as test fixtures: they
exercise the code paths and are never written to ml/artifacts or used to
train anything.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

_TMP_DB = Path(tempfile.mkdtemp()) / "test_graph.db"
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMP_DB}")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-for-production")

from fastapi.testclient import TestClient  # noqa: E402

from app.database import Base, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.ml import graph_model  # noqa: E402
from graph import gat_numpy  # noqa: E402  (ml/ dir via app.ml.paths)
from graph.graph_build import MatrixError, build_graph, select_edges, validate_matrix  # noqa: E402

N_REGIONS = 10
N_TIMEPOINTS = 60
HEADS, HIDDEN, LAYERS = 2, 4, 2


def _user(tag: str) -> dict:
    return {"email": f"graph-{tag}@example.com", "password": "testpass123",
            "full_name": f"Graph {tag}", "age": 70, "education_years": 14}


# --------------------------------------------------------------------------
# fixtures (generated arrays: test inputs only, never training data)
# --------------------------------------------------------------------------
def fc_matrix(seed: int = 0) -> np.ndarray:
    ts = np.random.default_rng(seed).standard_normal((N_TIMEPOINTS, N_REGIONS))
    return np.corrcoef(ts.T)


def sc_matrix(seed: int = 1) -> np.ndarray:
    m = np.abs(np.random.default_rng(seed).standard_normal((N_REGIONS, N_REGIONS))) * 100
    return m + m.T


AFFINE = np.diag([8.0, 8.0, 8.0, 1.0])
AFFINE[:3, 3] = [-40, -56, -32]


def labels_img():
    import nibabel as nib

    labels = np.zeros((10, 14, 10), np.int16)
    for i in range(5):
        labels[i * 2:(i + 1) * 2, :7, :] = i + 1
        labels[i * 2:(i + 1) * 2, 7:, :] = i + 6
    return nib.Nifti1Image(labels, AFFINE)


def bold_bytes(shape=(10, 14, 10, N_TIMEPOINTS)) -> bytes:
    import nibabel as nib

    data = (np.random.default_rng(2).standard_normal(shape) + 100).astype(np.float32)
    img = nib.Nifti1Image(data, AFFINE)
    img.header.set_zooms((8.0, 8.0, 8.0, 2.0)[: len(shape)])
    return gzip.compress(img.to_bytes())


def random_weights(seed: int = 0) -> dict:
    """Random parameters with the trained model's shapes (fixture only)."""
    rng = np.random.default_rng(seed)
    w, d = {}, 2 * N_REGIONS + 2
    for i in range(LAYERS):
        w[f"convs.{i}.lin.weight"] = rng.standard_normal((HEADS * HIDDEN, d)) * 0.3
        for a in ("att_src", "att_dst", "att_edge"):
            w[f"convs.{i}.{a}"] = rng.standard_normal((1, HEADS, HIDDEN)) * 0.3
        w[f"convs.{i}.lin_edge.weight"] = rng.standard_normal((HEADS * HIDDEN, 2)) * 0.3
        w[f"convs.{i}.bias"] = np.zeros(HEADS * HIDDEN)
        d = HEADS * HIDDEN
    w["gate.weight"], w["gate.bias"] = rng.standard_normal((1, d)), np.zeros(1)
    w["head.0.weight"], w["head.0.bias"] = rng.standard_normal((6, 2 * d)), np.zeros(6)
    w["head.3.weight"], w["head.3.bias"] = rng.standard_normal((3, 6)), np.zeros(3)
    return w


CONFIG = {"atlas": "test-atlas", "region_names": [f"Region {i + 1}" for i in range(N_REGIONS)],
          "n_regions": N_REGIONS, "top_k": 3, "threshold": None, "with_sc": True,
          "hidden": HIDDEN, "heads": HEADS, "layers": LAYERS, "version": "test"}


@pytest.fixture
def no_model(tmp_path, monkeypatch):
    """Artifacts point at an empty dir: the untrained state."""
    monkeypatch.setattr(graph_model, "GRAPH_MODEL_PATH", tmp_path / "graph_model.npz")
    monkeypatch.setattr(graph_model, "GRAPH_CONFIG_PATH", tmp_path / "graph_config.json")
    monkeypatch.setattr(graph_model, "GRAPH_METRICS_PATH", tmp_path / "graph_metrics.json")
    monkeypatch.setenv("GRAPH_ENABLED", "true")
    graph_model.reset()
    yield tmp_path
    graph_model.reset()


@pytest.fixture
def fixture_model(no_model, monkeypatch):
    """A randomly initialised model in a temp dir (exercises the code path only)."""
    np.savez(no_model / "graph_model.npz", **random_weights())
    (no_model / "graph_config.json").write_text(json.dumps(CONFIG))
    monkeypatch.setattr(graph_model, "_atlas", lambda name: labels_img())
    graph_model.reset()
    return no_model


@pytest.fixture(scope="module")
def client():
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as c:
        yield c


def _signed_in(tag: str) -> TestClient:
    c = TestClient(app)
    r = c.post("/api/auth/signup", json=_user(tag))
    assert r.status_code in (201, 409), r.text
    if r.status_code == 409:
        c.post("/api/auth/login", json={"email": _user(tag)["email"], "password": "testpass123"})
    return c


@pytest.fixture(scope="module")
def alice(client):
    return _signed_in("alice")


@pytest.fixture(scope="module")
def bob(client):
    return _signed_in("bob")


def _files(fmri: bytes | None = None, sc: np.ndarray | None = None, fmri_name="rest_bold.nii.gz",
           sc_name="sc.npy", sc_raw: bytes | None = None):
    files = {"fmri": (fmri_name, bold_bytes() if fmri is None else fmri, "application/gzip")}
    if sc_raw is not None or sc is not None:
        if sc_raw is None:
            buf = io.BytesIO()
            np.save(buf, sc)
            sc_raw = buf.getvalue()
        files["sc"] = (sc_name, sc_raw, "application/octet-stream")
    return files


# --------------------------------------------------------------------------
# graph construction
# --------------------------------------------------------------------------
def test_graph_nodes_features_and_edge_attr():
    g = build_graph(fc_matrix(), sc_matrix(), top_k=3)
    assert g["num_nodes"] == N_REGIONS
    assert g["x"].shape == (N_REGIONS, 2 * N_REGIONS + 2)
    e = g["edge_index"].shape[1]
    assert g["edge_attr"].shape == (e, 2)
    # one self-loop per node, appended after the real edges
    loops = g["edge_index"][0] == g["edge_index"][1]
    assert loops.sum() == N_REGIONS and np.allclose(g["edge_attr"][loops, 0], 1.0)
    assert 0.0 <= g["edge_attr"][:, 1].min() and g["edge_attr"][:, 1].max() <= 1.0  # SC normalised


def test_threshold_edges_respected():
    fc = fc_matrix()
    g = build_graph(fc, top_k=None, threshold=0.15)
    src, dst = g["edge_index"]
    real = src != dst
    assert real.any()
    assert np.all(np.abs(fc[src[real], dst[real]]) > 0.15)


def test_edges_are_symmetric():
    g = build_graph(fc_matrix(), top_k=2)
    pairs = set(zip(*g["edge_index"].tolist()))
    assert all((d, s) in pairs for s, d in pairs)
    adj = select_edges(fc_matrix(), top_k=2)
    assert (adj == adj.T).all() and not adj.diagonal().any()


def test_matrix_validation_rejects_bad_sc():
    with pytest.raises(MatrixError, match="must be 10x10"):
        validate_matrix(np.zeros((5, 5)), N_REGIONS)
    bad = sc_matrix()
    bad[0, 1] = np.nan
    with pytest.raises(MatrixError, match="NaN"):
        validate_matrix(bad, N_REGIONS)
    asym = sc_matrix()
    asym[0, 1] += 50
    with pytest.raises(MatrixError, match="symmetric"):
        validate_matrix(asym, N_REGIONS)


# --------------------------------------------------------------------------
# GAT forward pass
# --------------------------------------------------------------------------
def test_gat_forward_probabilities_and_attention():
    g = build_graph(fc_matrix(), sc_matrix(), top_k=3)
    out = gat_numpy.forward(g, random_weights(), CONFIG)
    p = out["probabilities"]
    assert p.shape == (3,) and abs(p.sum() - 1.0) < 1e-6 and (p >= 0).all()
    assert len(out["attention"]) == LAYERS
    for alpha in out["attention"]:
        assert alpha.shape == (g["edge_index"].shape[1], HEADS)
        # attention over each node's incoming edges sums to 1 per head
        sums = np.zeros((N_REGIONS, HEADS))
        np.add.at(sums, g["edge_index"][1], alpha)
        assert np.allclose(sums, 1.0)
    top = gat_numpy.top_edges(g, out["attention"], CONFIG["region_names"], k=5)
    assert 0 < len(top) <= 5
    assert all(t["region_a"] != t["region_b"] for t in top)


def test_numpy_forward_matches_torch_geometric():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from graph.gat import GraphGAT, export_weights, to_data

    torch.manual_seed(0)
    g = build_graph(fc_matrix(), sc_matrix(), top_k=3)
    model = GraphGAT(g["x"].shape[1], hidden=HIDDEN, heads=HEADS, layers=LAYERS).eval()
    d = to_data(g)
    with torch.no_grad():
        pt = torch.softmax(model(d.x, d.edge_index, d.edge_attr, torch.zeros(N_REGIONS, dtype=torch.long)), 1)
    w = {k: v.astype(np.float64) for k, v in export_weights(model).items()}
    pn = gat_numpy.forward(g, w, CONFIG)["probabilities"]
    assert np.abs(pt[0].numpy() - pn).max() < 1e-5


def test_fc_from_bold_shape_and_rejects_3d():
    from graph.connectivity import BoldError, fc_from_bold, load_bold_bytes

    fc = fc_from_bold(load_bold_bytes(bold_bytes()), labels_img())
    assert fc.shape == (N_REGIONS, N_REGIONS) and np.allclose(fc, fc.T) and np.isfinite(fc).all()
    with pytest.raises(BoldError, match="4-D"):
        load_bold_bytes(bold_bytes(shape=(10, 14, 10)))


# --------------------------------------------------------------------------
# flags and the untrained state
# --------------------------------------------------------------------------
def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("GRAPH_ENABLED", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "development")
    assert graph_model.enabled() is False


def test_enabled_fails_closed_in_production(monkeypatch):
    from app.config import settings

    monkeypatch.delenv("GRAPH_ENABLED", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "graph_enabled", True)  # a stray .env value
    assert graph_model.enabled() is False
    monkeypatch.setenv("GRAPH_ENABLED", "true")
    assert graph_model.enabled() is True


def test_analyze_requires_auth(client, no_model):
    fresh = TestClient(app)
    assert fresh.post("/api/graph/analyze", files=_files()).status_code == 401


def test_analyze_disabled_is_503(alice, no_model, monkeypatch):
    monkeypatch.setenv("GRAPH_ENABLED", "false")
    r = alice.post("/api/graph/analyze", files=_files())
    assert r.status_code == 503 and "not enabled" in r.json()["detail"]


def test_untrained_model_is_503_not_fabricated(alice, no_model):
    r = alice.post("/api/graph/analyze", files=_files(sc=sc_matrix()))
    assert r.status_code == 503
    assert "not been trained" in r.json()["detail"]
    assert "probabilities" not in r.json()


def test_info_without_metrics_omits_them(client, no_model):
    body = client.get("/api/graph/info").json()
    assert body["available"] is False and body["trained"] is False
    assert "cv" not in body and "not been trained" in body["message"]


def test_info_with_metrics_reports_them(client, fixture_model):
    metrics = {"model": "GAT", "classes": ["Normal (CDR 0)", "MCI (CDR 0.5)", "Dementia (CDR >= 1)"],
               "cv": {"folds": 5, "accuracy": {"mean": 0.5, "std": 0.1}}, "caveats": ["x"], "version": "t"}
    (fixture_model / "graph_metrics.json").write_text(json.dumps(metrics))
    graph_model.reset()
    body = client.get("/api/graph/info").json()
    assert body["available"] is True and body["cv"]["accuracy"]["mean"] == 0.5
    assert body["caveats"] == ["x"]


# --------------------------------------------------------------------------
# input validation (422)
# --------------------------------------------------------------------------
def test_wrong_suffix_is_422(alice, fixture_model):
    r = alice.post("/api/graph/analyze", files=_files(fmri_name="scan.txt", sc=sc_matrix()))
    assert r.status_code == 422
    r = alice.post("/api/graph/analyze", files=_files(sc=sc_matrix(), sc_name="sc.json"))
    assert r.status_code == 422


def test_empty_file_is_422(alice, fixture_model):
    r = alice.post("/api/graph/analyze", files=_files(fmri=b"", sc=sc_matrix()))
    assert r.status_code == 422 and "empty" in r.json()["detail"]


def test_3d_volume_is_422(alice, fixture_model):
    r = alice.post("/api/graph/analyze", files=_files(fmri=bold_bytes(shape=(10, 14, 10)), sc=sc_matrix()))
    assert r.status_code == 422 and "4-D" in r.json()["detail"]


def test_sc_shape_mismatch_is_422(alice, fixture_model):
    r = alice.post("/api/graph/analyze", files=_files(sc=np.eye(5)))
    assert r.status_code == 422 and "10x10" in r.json()["detail"]


def test_sc_with_nans_is_422(alice, fixture_model):
    sc = sc_matrix()
    sc[2, 3] = sc[3, 2] = np.nan
    r = alice.post("/api/graph/analyze", files=_files(sc=sc))
    assert r.status_code == 422 and "NaN" in r.json()["detail"]


def test_sc_csv_asymmetric_is_422(alice, fixture_model):
    sc = sc_matrix()
    sc[0, 9] += 40
    raw = "\n".join(",".join(f"{v:.3f}" for v in row) for row in sc).encode()
    r = alice.post("/api/graph/analyze", files=_files(sc_raw=raw, sc_name="sc.csv"))
    assert r.status_code == 422 and "symmetric" in r.json()["detail"]


def test_missing_sc_when_model_needs_it_is_422(alice, fixture_model):
    r = alice.post("/api/graph/analyze", files=_files())
    assert r.status_code == 422 and "structural connectivity" in r.json()["detail"]


# --------------------------------------------------------------------------
# happy path + owner isolation
# --------------------------------------------------------------------------
def test_analyze_stores_owner_scoped_result(alice, bob, fixture_model):
    r = alice.post("/api/graph/analyze", files=_files(sc=sc_matrix()))
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["probabilities"]) == {"Normal", "MCI", "Dementia"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3
    assert body["predicted_class"] in body["probabilities"]
    assert body["top_edges"] and {"region_a", "region_b", "attention"} <= set(body["top_edges"][0])
    rid = body["id"]

    assert alice.get(f"/api/graph/results/{rid}").status_code == 200
    assert any(x["id"] == rid for x in alice.get("/api/graph/results").json())
    assert bob.get(f"/api/graph/results/{rid}").status_code == 404
    assert all(x["id"] != rid for x in bob.get("/api/graph/results").json())


def test_fusion_hook_is_disabled():
    import fusion

    assert fusion.GRAPH_FUSION_ENABLED is False
    assert fusion.graph_component({"Normal": 0.1, "MCI": 0.5, "Dementia": 0.4}) is None
