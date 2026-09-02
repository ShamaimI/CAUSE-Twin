import os
from flask import Flask, render_template, jsonify, request
from data_loader import DigitalTwinEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)
engine = DigitalTwinEngine()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/explorer")
def explorer():
    countries = engine.get_country_list()
    return render_template("explorer.html", countries=countries)


@app.route("/simulator")
def simulator():
    countries = engine.get_country_list()
    return render_template("simulator.html", countries=countries)


@app.route("/diagnostics")
def diagnostics():
    return render_template("diagnostics.html")


@app.route("/governance")
def governance():
    return render_template("governance.html")


@app.route("/api/country_data", methods=["POST"])
def api_country_data():
    data = request.json or {}
    country = data.get("country", "Algeria")
    baseline = engine.get_country_baseline(country)
    history = engine.get_country_history(country)
    return jsonify({"baseline": baseline, "history": history})


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    data = request.json or {}
    country = data.get("country", "Algeria")
    stunting_shift = float(data.get("stunting_shift", 5.0))
    sanitation_shift = float(data.get("sanitation_shift", 2.5))
    literacy_shift = float(data.get("literacy_shift", 1.0))
    gdp_shift = float(data.get("gdp_shift", 2.0))

    result = engine.simulate_multi_lever_policy(
        country, stunting_shift, sanitation_shift, literacy_shift, gdp_shift
    )
    return jsonify(result)


@app.route("/api/shap", methods=["GET"])
def api_shap():
    shap_data = engine.get_shap_importance()
    return jsonify(shap_data)


if __name__ == "__main__":
    app.run(debug=True, port=5000)