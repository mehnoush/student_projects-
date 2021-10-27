import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
#from online_detecter import predict
from utilitie import predict_emotion



UPLOAD_FOLDER = './data'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('results', filename=filename))
    return render_template('landing_page.html')




@app.route('/results')
def results():
    filename= request.args.get('filename',type=str)
    #access to image image can be processes
    #costruct the whole file path 
    #readin the image
    #preproccessing the data
    #take the model and run the mode.predict with the image
    #transform the result to labes
    output=predict_emotion(filename)
    return render_template('results.html', emotion = output)


if __name__ == "__main__":
    # executes the app on the development server which we can access via: http://127.0.0.1:5000/
    # debug=True restarts the server everytime we save changes so we can see them in real time
    # also gives us verbose debugging information in the terminal
    app.run(debug=True)