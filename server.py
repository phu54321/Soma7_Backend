from flask import Flask, jsonify, request
from sklearn.externals import joblib
from konlpy.tag import Mecab
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

text_clf = joblib.load('text_clf.dat')
mecab = Mecab()
app = Flask(__name__)


@app.route('/classify', methods=['GET'])
def classify():
    name = request.args['name'].lower()
    name = ' '.join(mecab.morphs(name))
    pred = text_clf.predict([name])[0]
    print("Classify : \"%s\" -> \"%s\"" % (name, pred))
    return jsonify({'cate': pred})

print("Running server on port 1234...")
app.run(host='0.0.0.0', port=1234)
