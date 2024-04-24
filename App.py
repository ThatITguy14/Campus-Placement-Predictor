from flask import Flask, request, render_template
import pickle

# Unpickling the model
file = open('campusplacementpredictor.pkl', 'rb')
rf = pickle.load(file)
file.close()

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        mydict = request.form
        Internships = int(mydict['Internships'])
        HSC_Marks = int(mydict['HSC_Marks'])
        SSC_Marks = int(mydict['SSC_Marks'])
        SoftSkillsRating = float(mydict['SoftSkillsRating'])
        Projects = int(mydict['Projects'])
        Workshops_Certifications = int(mydict['Workshops_Certifications'])
        ExtracurricularActivities = int(mydict['ExtracurricularActivities'])
        PlacementTraining = int(mydict['PlacementTraining'])
        AptitudeTestScore = int(mydict['AptitudeTestScore'])
        CGPA = float(mydict['CGPA'])

        inputfeatures = [[Internships, HSC_Marks, SSC_Marks, SoftSkillsRating, Projects, Workshops_Certifications,
                          ExtracurricularActivities, PlacementTraining, AptitudeTestScore, CGPA]]

        # Predicting the class either 0 or 1
        predictedclass = rf.predict(inputfeatures)

        # Predicting the probability
        predictedprob = rf.predict_proba(inputfeatures)

        if predictedclass[0] == 1:
            proba = predictedprob[0][1] * 100
        else:
            proba = predictedprob[0][0] * 100

        placemap = {1: 'Will be Placed', 0: 'Better Luck Next Time :('}
        predictedclasssend = placemap[predictedclass[0]]

        if predictedclass[0] == 1:
            return render_template('show1.html', predictedclasssend=predictedclasssend, predictedprob=round(proba, 2),
                                   placed=True)
        else:
            return render_template('show1.html', predictedclasssend=predictedclasssend)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
