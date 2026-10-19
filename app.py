"""Flask API and dashboard for the hospital bed management platform.

Two separate versions, each with its own database:
  python app.py              demo hospital with simulated patients    http://localhost:5000
  python app.py --hospital   empty hospital that staff set up         http://localhost:5001
"""
import argparse
import os
import sys

from flask import Flask, jsonify, render_template, request

from hospital import db, models, services
from hospital.seed_demo import seed

LIVE = "--hospital" in sys.argv or os.environ.get("HOSPITAL_MODE") == "live"
db.configure(db.LIVE_DB if LIVE else db.DEMO_DB)

app = Flask(__name__)
app.json.sort_keys = False


def ensure_db():
    if db.DB_PATH.exists():
        return
    with db.session() as con:
        if LIVE:
            db.init(con)
            db.set_setting(con, "mode", "live")
        else:
            seed(con)


ensure_db()


def error(msg, code=400):
    return jsonify({"error": msg}), code


@app.get("/")
def index():
    meta = models.triage_meta() or {}
    with db.session() as con:
        ward_names = [w["name"] for w in services.wards(con)]
        name = db.get_settings(con).get("hospital_name") or ("Demo hospital" if not LIVE else "")
    return render_template("index.html", wards=ward_names, complaints=meta.get("complaints", []),
                           arrival_modes=meta.get("categories", {}).get("arrivalmode", []),
                           live=LIVE, needs_setup=LIVE and not ward_names, hospital_name=name)


# ---------- setup (hospital version only) ----------

@app.post("/api/setup")
def setup():
    data = request.get_json(silent=True) or {}
    with db.session() as con:
        try:
            services.setup_hospital(con, data.get("name", ""), data.get("wards", []))
        except (ValueError, KeyError, TypeError) as e:
            return error(str(e))
    return jsonify({"ok": True}), 201


@app.get("/api/wards")
def get_wards():
    with db.session() as con:
        return jsonify({"wards": services.wards(con), "hospital_name": db.get_settings(con).get("hospital_name")})


@app.post("/api/wards")
def add_ward():
    data = request.get_json(silent=True) or {}
    with db.session() as con:
        try:
            services.add_ward(con, data.get("name"), int(data.get("beds", 0)))
        except (ValueError, TypeError) as e:
            return error(str(e))
    return jsonify({"ok": True}), 201


@app.post("/api/wards/<int:ward_id>/beds")
def add_beds(ward_id):
    data = request.get_json(silent=True) or {}
    with db.session() as con:
        try:
            services.add_beds(con, ward_id, int(data.get("count", 0)))
        except (ValueError, TypeError) as e:
            return error(str(e))
    return jsonify({"ok": True})


@app.get("/api/overview")
def overview():
    with db.session() as con:
        return jsonify(services.overview(con, int(request.args.get("days", 7))))


@app.get("/api/queue")
def get_queue():
    with db.session() as con:
        return jsonify({"queue": services.queue(con), "settings": db.get_settings(con)})


@app.post("/api/queue")
def add_to_queue():
    data = request.get_json(silent=True) or {}
    if not data.get("name"):
        return error("name is required")
    with db.session() as con:
        pid = services.add_patient(con, data)
        out = {"id": pid}
        if data.get("admit_now"):
            try:
                out.update(services.admit(con, pid))
            except ValueError as e:
                out["admit_error"] = str(e)
        return jsonify(out), 201


@app.post("/api/queue/<int:pid>/remove")
def remove(pid):
    with db.session() as con:
        try:
            services.remove_from_queue(con, pid)
        except ValueError as e:
            return error(str(e), 409)
    return jsonify({"ok": True})


@app.post("/api/queue/<int:pid>/admit")
def admit(pid):
    data = request.get_json(silent=True) or {}
    with db.session() as con:
        try:
            return jsonify(services.admit(con, pid, data.get("bed_id")))
        except ValueError as e:
            return error(str(e), 409)


@app.post("/api/queue/<int:pid>/urgency")
def set_urgency(pid):
    u = int((request.get_json(silent=True) or {}).get("urgency", 1))
    if u not in (1, 2, 3):
        return error("urgency must be 1, 2 or 3")
    with db.session() as con:
        con.execute("UPDATE patients SET urgency = ? WHERE id = ? AND status = 'waiting'", (u, pid))
    return jsonify({"ok": True})


@app.post("/api/settings")
def update_settings():
    data = request.get_json(silent=True) or {}
    with db.session() as con:
        for k in ("w_urgency", "w_wait", "w_prob", "wait_cap"):
            if k in data:
                db.set_setting(con, k, float(data[k]))
        return jsonify(db.get_settings(con))


@app.post("/api/triage")
def triage():
    values = request.get_json(silent=True) or {}
    if values.get("esi") in (None, ""):
        return error("ESI level is required")
    out = models.predict_admit(values)
    out["suggested_urgency"] = models.urgency_from_esi(values["esi"])
    return jsonify(out)


@app.get("/api/beds")
def beds():
    with db.session() as con:
        return jsonify(services.beds(con))


@app.post("/api/beds/<int:bed_id>/discharge")
def discharge(bed_id):
    with db.session() as con:
        try:
            services.discharge(con, bed_id)
        except ValueError as e:
            return error(str(e), 409)
    return jsonify({"ok": True})


@app.post("/api/simulate/next-day")
def next_day():
    if LIVE:
        return error("the clock only moves in the demo", 403)
    with db.session() as con:
        return jsonify(services.next_day(con))


@app.post("/api/simulate/reset")
def reset():
    if LIVE:
        return error("reset is only available in the demo", 403)
    with db.session() as con:
        seed(con)
    return jsonify({"ok": True})


@app.get("/api/metrics")
def metrics():
    return jsonify(models.all_metrics())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hospital", action="store_true", help="run the empty hospital version instead of the demo")
    ap.add_argument("--port", type=int)
    args = ap.parse_args()
    app.run(host="127.0.0.1", port=args.port or (5001 if LIVE else 5000), debug=False)
