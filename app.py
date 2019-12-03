from collections import OrderedDict

from flask import Flask, json, request
import pandas as pd
import requests
import googleapiclient.discovery
from google.oauth2 import service_account
from google_auth_oauthlib.flow import InstalledAppFlow

from mysql import Engine

app = Flask(__name__)

google_project = 'yelp-final-260208'
google_yelp_model_logreg = 'rfc'
# google_ai_url = ' https://ml.googleapis.com/v1/projects/yelp-final-260208/models/yelp_logreg:predict'
# google_access_token = 'ya29.ImayB5jPfOOYcQjmcEKPS1Q63n33gjKlIGEn161D2HP0RtXNp3XTIajHfb8mKX8swulxJbxldVCkuk6aAM-eSsX0jJFpvaTwOVxWlou2IjqKoEN_dyzDoqPkUDEOSVI4VmYiCR0nlbQ'

SERVICE_ACCOUNT_FILE = 'credentials_service.json'
SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
MYSQL_TABLE_NAME = 'active_learning'

ordered_features = OrderedDict([
    ('review_count', 1),
    ('Alcohol', 0),
    ('Caters', 0),
    ('GoodForKids', 0),
    ('HasTV', 0),
    ('NoiseLevel', 0),
    ('OutdoorSeating', 0),
    ('RestaurantsDelivery', 0),
    ('RestaurantsGoodForGroups', 0),
    ('RestaurantsPriceRange2', 0),
    ('RestaurantsReservations', 0),
    ('RestaurantsTakeOut', 0),
    ('WiFi', 0),
    ('parking', 0),
    ('street', 0),
    ('valet', 0),
    ('casual', 0),
    ('classy', 0),
    ('divey', 0),
    ('hipster', 0),
    ('intimate', 0),
    ('romantic', 0),
    ('touristy', 0),
    ('trendy', 0),
    ('upscale', 0),
    ('breakfast', 0),
    ('brunch', 0),
    ('dessert', 0),
    ('dinner', 0),
    ('latenight', 0),
    ('lunch', 0),
    ('Vegan', 0),
    ('Mediterranean', 0),
    ('Indian', 0),
    ('Seafood', 0),
    ('Sandwiches', 0),
    ('Nightlife', 0),
    ('Salad', 0),
    ('Mexican', 0),
    ('Mongolian', 0),
    ('Thai', 0),
    ('FastFood', 0),
    ('AsianFusion', 0),
    ('Pizza', 0),
    ('Buffets', 0),
    ('Italian', 0),
    ('Korean', 0),
    ('French', 0),
    ('Vegetarian', 0),
    ('SushiBars', 0),
    ('Japanese', 0),
    ('Bars', 0),
    ('Chinese', 0),
    ('Burgers', 0),
    ('American', 0)
])


@app.route('/')
def hello_world():
    return "Hello World!"


@app.route('/predict', methods=['POST'])
def predict():
    request_map = request.json
    # This will set all missing values with defaults.
    # And change the feature values only for a few input keys given by user.
    for key in request_map:
        if ordered_features[key] is not None:
            ordered_features[key] = request_map[key]
    csv_list = get_features_csv()
    response = predict_json(google_project, google_yelp_model_logreg, csv_list)
    active_learn(csv_list, response['predictions'][0])
    return str(response['predictions'][0])


def active_learn(csv_list, predicted_stars):
    csv_list[0].insert(1, predicted_stars)
    csv_list[0].append(1)
    db = Engine.get_db_conn()
    csv_str = str(csv_list).replace('[', '').replace(']', '')
    db.execute('INSERT INTO {} VALUES ({})'.format(MYSQL_TABLE_NAME, csv_str))


def predict_json(project, model, instances, version=None):
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)

    service = googleapiclient.discovery.build('ml', 'v1', credentials=credentials)
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response


def get_features_csv():
    df_row = pd.DataFrame.from_dict(ordered_features, orient="index")
    return df_row.T.to_numpy().tolist()


if __name__ == '__main__':
    app.run()
