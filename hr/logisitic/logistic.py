from firebase_admin import credentials, initialize_app, firestore
import os
import datetime
import logging



update_data = {
            'date': datetime.datetime(2022,2,7),
            'created':firestore.SERVER_TIMESTAMP,
            'invoice_no': 'SF1426423964966',
            'tel': 22574307,
            'weight_kg': 0.5,
            'cost_hkd': 46,
            'staff_no':2
        }


class Logistic:
    def __init__(self) -> None:
        dir = os.path.dirname(__file__)
        key_path = os.path.join(dir, "../../key/001_HR_logistic_pirvate_key.json")

        cred = credentials.Certificate(key_path)
        self.app = initialize_app(cred, {
            'databaseURL': 'https://logistics-system-2ff4a-default-rtdb.firebaseio.com/'
        })
        self.db = firestore.client()
        pass

    def add_data(self,data:dict) -> None:
        try:
            print('HR Logistics Recording')
            db = firestore.client()
            ref =db.collection('shipping')
            ref.add(data)
            logging.info('insert sucess')
        except Exception as e:
            logging.error('insert fail')
        return
    
    def main(self) -> None:
        self.add_data(update_data)

