"""Flask API and dashboard for the hospital bed management platform.

Run:  python app.py   then open http://localhost:5000
"""
from flask import Flask, jsonify, render_template, request

from hospital import db, models, services
from hospital.seed_demo import WARDS, seed

app = Flask(__name__)
app.json.sort_keys = False


def ensure_db():
    if not db.DB_PATH.exists():
        with db.session() as con:
            seed(con)


ensure_db()


def error(msg, code=400):
    return jsonify({"error": msg}), code


@app.get("/")
def index():
    meta = models.triage_meta() or {}
    return render_template("index.html", wards=list(WARDS), complaints=meta.get("complaints", []),
                           arrival_modes=meta.get("categories", {}).get("arrivalmode", []))


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
        return jsonify({"id": pid}), 201


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
    with db.session() as con:
        return jsonify(services.next_day(con))


@app.post("/api/simulate/reset")
def reset():
    with db.session() as con:
        seed(con)
    return jsonify({"ok": True})


@app.get("/api/metrics")
def metrics():
    return jsonify(models.all_metrics())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
