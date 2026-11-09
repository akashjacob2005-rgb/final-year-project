"""OASIS-3 (NITRC-IR) client and graph session-list logic, against a fake XNAT.

No network: a stub requests.Session answers the handful of XNAT REST paths
the scripts use.
"""

from __future__ import annotations

import pytest

import app.ml.paths  # noqa: F401  (puts ml/ on sys.path)
from graph import build_session_list as bsl  # noqa: E402
import oasis_xnat  # noqa: E402


class _Resp:
    def __init__(self, status=200, text="", content=b""):
        self.status_code, self.text, self.content = status, text, content or text.encode()

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeXnat:
    """Minimal XNAT: one project, two sessions, a data-files subject with a CDR CSV."""

    def __init__(self, project_status=200):
        self.project_status = project_status
        self.auth = None
        self.calls = []

    def post(self, url, timeout=30):
        return _Resp(200 if url.endswith("/data/JSESSION") else 404, "token")

    def get(self, url, timeout=120, stream=False):
        self.calls.append(url)
        path = url.split("/ir", 1)[1]
        if path.startswith("/data/projects/OASIS3?"):
            return _Resp(self.project_status, "{}")
        if "experiments?xsiType=xnat:mrSessionData" in path:
            return _Resp(text="label,subject_label\nOAS30001_MR_d0129,OAS30001\n"
                              "OAS30002_MR_d2000,OAS30002\nOAS_DF_MR,0AS_data_files\n")
        if path.endswith("/subjects/OAS30001/experiments/OAS30001_MR_d0129/scans?format=csv"):
            return _Resp(text="ID,type\n1,T1w\n5,func-bold task-rest\n6,dwi\n")
        if path.endswith("/subjects/OAS30002/experiments/OAS30002_MR_d2000/scans?format=csv"):
            return _Resp(text="ID,type\n1,T1w\n2,FLAIR\n")
        if path.endswith("/subjects/0AS_data_files/experiments?format=csv"):
            return _Resp(text="ID,label\nX1,OASIS3_data_files\n")
        if path.endswith("/experiments/OASIS3_data_files/files?format=csv"):
            return _Resp(text="Name,URI\nOASIS3_UDSb4_cdr.csv,/data/files/cdr.csv\nreadme.txt,/data/files/r\n")
        if path == "/data/files/cdr.csv":
            return _Resp(text="OASISID,days_to_visit,CDRTOT\nOAS30001,0,0\nOAS30001,400,0.5\n"
                              "OAS30002,100,1\n")
        return _Resp(404)


def client(**kw):
    return oasis_xnat.OasisXnat("u", "p", base="https://www.nitrc.org/ir", session=FakeXnat(**kw))


def test_default_server_is_nitrc_ir():
    assert oasis_xnat.BASE.startswith("https://www.nitrc.org/ir")


def test_login_and_project_access_denied():
    client().login()  # ok
    with pytest.raises(oasis_xnat.AccessDenied, match="OASIS3 project"):
        client(project_status=403).login()


def test_session_and_scan_listing():
    x = client()
    labels = {s["label"] for s in x.mr_sessions()}
    assert "OAS30001_MR_d0129" in labels
    scans = x.scans("OAS30001", "OAS30001_MR_d0129")
    assert [s["ID"] for s in scans if bsl.is_rest_bold(s["type"])] == ["5"]
    assert [s["ID"] for s in scans if bsl.is_dwi(s["type"])] == ["6"]
    assert not bsl.is_rest_bold("T1w") and not bsl.is_rest_bold("func-bold task-nback")


def test_clinical_table_found_and_parsed():
    rows = bsl.find_clinical_rows(client())
    table = bsl.parse_cdr_table(rows)
    assert table["OAS30001"] == [(0, 0.0), (400, 0.5)]


def test_nearest_cdr_labelling_and_window():
    table = {"OAS30001": [(0, 0.0), (400, 0.5)], "OAS30002": [(100, 1.0)]}
    assert bsl.session_days("OAS30001_MR_d0129") == 129
    assert bsl.nearest_label("OAS30001_MR_d0129", "OAS30001", table, 365) == (0, 0.0, 129)
    assert bsl.nearest_label("OAS30001_MR_d0350", "OAS30001", table, 365)[0] == 1  # 0.5 -> MCI
    assert bsl.nearest_label("OAS30002_MR_d2000", "OAS30002", table, 365) is None  # too far
    assert bsl.cdr_to_label(2.0) == 2


def test_adrc_layout_without_days_column():
    rows = [{"ADRC_ADRCCLINICALDATA ID": "OAS30003_ClinicalData_d0042", "Subject": "OAS30003", "cdr": "0.5"}]
    assert bsl.parse_cdr_table(rows) == {"OAS30003": [(42, 0.5)]}


def test_build_session_list_end_to_end(tmp_path, monkeypatch, capsys):
    import csv
    import sys

    fake = client()  # build before patching: client() uses the real class
    monkeypatch.setattr(oasis_xnat, "OasisXnat", lambda *a, **k: fake)
    monkeypatch.setattr(sys, "argv", ["build_session_list.py", "--data-root", str(tmp_path), "--workers", "1"])
    bsl.main()
    sessions = list(csv.DictReader(open(tmp_path / "processed" / "graph_sessions.csv")))
    labels = list(csv.DictReader(open(tmp_path / "processed" / "graph_session_labels.csv")))
    # OAS30002 has no rs-fMRI; OAS30001 does, with a CDR 0 visit 129 days away.
    assert [s["experiment_id"] for s in sessions] == ["OAS30001_MR_d0129"]
    assert sessions[0]["bold_ids"] == "5" and sessions[0]["dwi_ids"] == "6"
    assert labels[0]["label"] == "0"
    assert "Normal 1" in capsys.readouterr().out


class _ZipCdrXnat(FakeXnat):
    """CDR CSV only available inside a zip on the subject itself."""

    def get(self, url, timeout=120, stream=False):
        import io as _io
        import zipfile as _zip

        path = url.split("/ir", 1)[1]
        if path.endswith("/subjects/0AS_data_files/files?format=csv"):
            return _Resp(text="Name,URI\nOASIS3_clinical.zip,/data/files/clin.zip\n")
        if path.endswith("/experiments/OASIS3_data_files/files?format=csv"):
            return _Resp(text="Name,URI\nreadme.txt,/data/files/r\n")
        if path == "/data/files/clin.zip":
            buf = _io.BytesIO()
            with _zip.ZipFile(buf, "w") as z:
                z.writestr("csv/OASIS3_UDSb4_cdr.csv", "OASISID,days_to_visit,CDRTOT\nOAS30001,0,1\n")
                z.writestr("csv/demographics.csv", "OASISID,AgeatEntry\nOAS30001,70\n")
            return _Resp(content=buf.getvalue())
        return super().get(url, timeout, stream)


class _NoCdrXnat(FakeXnat):
    def get(self, url, timeout=120, stream=False):
        if url.endswith("/experiments/OASIS3_data_files/files?format=csv"):
            return _Resp(text="Name,URI\nOASIS3_MR_json.zip,/data/files/x\nreadme.txt,/data/files/r\n")
        if url.endswith("/data/files/x"):
            import io as _io
            import zipfile as _zip

            buf = _io.BytesIO()
            with _zip.ZipFile(buf, "w") as z:
                z.writestr("a.json", "{}")
            return _Resp(content=buf.getvalue())
        return super().get(url, timeout, stream)


def test_cdr_table_found_inside_zip():
    x = oasis_xnat.OasisXnat("u", "p", base="https://www.nitrc.org/ir", session=_ZipCdrXnat())
    rows = bsl.find_clinical_rows(x)
    assert bsl.parse_cdr_table(rows) == {"OAS30001": [(0, 1.0)]}


def test_no_cdr_table_lists_files_seen(capsys):
    x = oasis_xnat.OasisXnat("u", "p", base="https://www.nitrc.org/ir", session=_NoCdrXnat())
    with pytest.raises(SystemExit, match="--clinical-csv"):
        bsl.find_clinical_rows(x)
    out = capsys.readouterr().out
    assert "OASIS3_data_files/OASIS3_MR_json.zip" in out and "readme.txt" in out


class _ScanCdrXnat(FakeXnat):
    """NITRC-IR layout: the CDR CSV is a file of a *scan* (SCANS/UDSb4/csv/...)."""

    def get(self, url, timeout=120, stream=False):
        path = url.split("/ir", 1)[1]
        if path.endswith("/experiments/OASIS3_data_files/files?format=csv"):
            return _Resp(text="Name,URI\n")
        if path.endswith("/experiments/OASIS3_data_files/scans?format=csv"):
            return _Resp(text="ID,type\nUDSb4,csv\n")
        if path.endswith("/experiments/OASIS3_data_files/scans/UDSb4/files?format=csv"):
            return _Resp(text="Name,URI\nOASIS3_UDSb4_cdr.csv,/data/files/cdr.csv\nUDSb4_catalog.xml,/data/x\n")
        return super().get(url, timeout, stream)


def test_cdr_table_found_under_scans():
    x = oasis_xnat.OasisXnat("u", "p", base="https://www.nitrc.org/ir", session=_ScanCdrXnat())
    assert "OAS30002" in bsl.parse_cdr_table(bsl.find_clinical_rows(x))


def test_selection_one_per_subject_and_normal_cap():
    from graph.download_oasis_func import select_sessions

    sessions = [{"experiment_id": e, "subject_id": e.split("_")[0], "bold_ids": "1", "dwi_ids": ""}
                for e in ("A_MR_d0001", "A_MR_d0500", "B_MR_d0001", "C_MR_d0001", "D_MR_d0001", "E_MR_d0001")]
    labels = {"A_MR_d0001": {"label": "0", "days_apart": "200"}, "A_MR_d0500": {"label": "0", "days_apart": "3"},
              "B_MR_d0001": {"label": "0", "days_apart": "0"}, "C_MR_d0001": {"label": "0", "days_apart": "0"},
              "D_MR_d0001": {"label": "1", "days_apart": "0"}, "E_MR_d0001": {"label": "2", "days_apart": "0"}}
    one = select_sessions(sessions, labels, one_per_subject=True, max_normal=0)
    assert [s["experiment_id"] for s in one if s["subject_id"] == "A"] == ["A_MR_d0500"]  # closest CDR
    capped = select_sessions(sessions, labels, one_per_subject=True, max_normal=1)
    kept = {labels[s["experiment_id"]]["label"] for s in capped}
    assert sum(labels[s["experiment_id"]]["label"] == "0" for s in capped) == 1
    assert {"1", "2"} <= kept  # every impaired person kept
    assert capped == select_sessions(sessions, labels, True, 1)  # deterministic
