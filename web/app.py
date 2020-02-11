from flask import Flask, session, render_template, request, redirect, url_for
from flask_session import Session
from model import Store, Impulsify

app = Flask(__name__)

model = Impulsify()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/reco", methods=["POST"])
def reco():

    property_name = request.form.get('property_name')
    property_code = request.form.get('property_code')
    flag_name = request.form.get('flag_name')
    num_of_rooms = request.form.get('num_of_rooms', type=int)
    location_type = request.form.get('location_type')
    state = request.form.get('state')
    revenue = request.form.get('revenue', type=float)
    profit_margin = request.form.get('profit_margin', type=float)
    occupancy_rate = request.form.get('occupancy_rate', type=float)

    store = Store(property_name, property_code, flag_name, num_of_rooms, revenue, profit_margin, occupancy_rate, location_type, state)
    cluster = model.predict(store)
    spor = model.predict_spor(store)
    profit_margin = model.predict_profit_margin(store)
    stores = model.predict_comparable_stores(store, 5)
    store_hat = store.upgrade_to(spor=spor, profit_margin=profit_margin)

    return render_template("reco.html", store = store, cluster=cluster, store_hat = store_hat, stores=stores.tolist())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)


### location_type
#array(['Urban', 'Airport', 'Suburban', 'Resort', 'Interstate', 'Campus'],
#      dtype=object)

### flag_name
# array(['Aston', 'Avid', 'Best Western Plus', 'Candlewood Suites',
#        'Comfort Suites', 'Crowne Plaza Hotels and Resorts',
#        'Red Lion Hotel', 'Doubletree by Hilton', 'Hilton Garden Inn',
#        'Embassy Suites', 'Independent', 'Hampton Inn and Suites',
#        'Hampton Inn', 'Hilton Suites', 'Hilton',
#        'Holiday Inn Express & Suites', 'Holiday Inn Express',
#        'Holiday Inn', 'Liv Dev', 'Home2', 'Homewood Suites',
#        'Hotel Indigo', 'Intercontinental Hotels Group',
#        'La Quinta Inns and Suites', 'Curio by Hilton', 'Quality Suites',
#        'Renaissance', 'Staybridge Suites', 'Tru by Hilton',
#        'Tapestry Collection'], dtype=object)

