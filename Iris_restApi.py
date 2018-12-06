import numpy as np

# Either way, you should now be using your virtualenv (notice how the prompt of your shell has changed to show the active environment).
# And if you want to go back to the real world, use the following command:
# $ deactivate on terminal
from flask import Flask, abort,jsonify,request
import pickle
# Change model path below
my_randon_forest= pickle.load(open('../../data/iris_model.pickle','rb'))
app =Flask(__name__)
# create ReST API at port 9000
@app.route('/api', methods=['POST'])
def make_predict():
    #do error checking here
    data=request.get_json(force=True)
    # Convert Data(json) to numpy array
    predit_request = [data['sl'],data['sw'],data['pl'],data['pw']]
    predit_request = np.array(predit_request)
    predit_request = predit_request.reshape(1,-1)
    print(predit_request)
    y_predicted = my_randon_forest.predict(predit_request)
    print(y_predicted)
    output = [y_predicted[0]]
    print("output", output)
    print(type(output[0]))
    newoutput =[]
    for i in output:
        newoutput.append(int(i))

    return jsonify(results=newoutput)

if __name__ == '__main__':
    app.run(port=9000, debug=True)