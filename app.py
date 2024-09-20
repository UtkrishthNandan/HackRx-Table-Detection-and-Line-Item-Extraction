from flask import Flask, request, render_template, redirect, url_for,flash,get_flashed_messages,send_file
from werkzeug.utils import secure_filename
import os
import io
import paddle_table_ocr
import pandas as pd

app = Flask(__name__) 
app.secret_key = "super secret key"
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['pdf', 'jpg', 'jpeg', 'png'])
df_files=pd.DataFrame(data=None,columns=['file_name','item_name','item_amount'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_files(): 
    global df_files   
    for file in os.listdir(os.path.join(app.config['UPLOAD_FOLDER'],'')):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'],file))
    if request.method == 'POST':
        files = request.files.getlist('files[]')
        
        if files:
            for file in files:
                if file and allowed_file(file.filename):
                    # Check file size
                    if file.content_length > 10 * 1024 * 1024:
                        flash('File size exceeds 10MB limit.')
                        continue  # Skip to the next file

                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    flash(f'{filename} uploaded successfully!')
                    df1=paddle_table_ocr.table_ocr(img_path=filepath)
                    print(df1)
                    print(df_files)
                    #df.reset_index(inplace=True, drop=True)
                    #df1.reset_index(inplace=True, drop=True)
                    try:
                        df_files=pd.concat([df_files,df1],axis=0)
                    except:
                        print(f'error occured while concatenating file:{filename}')
                    os.remove(filepath)
                else:
                    flash(f'{file.filename} is not a valid file.')
            print(df_files.to_string())
        else:
            flash('No files were selected.')

    return render_template('upload.html')
@app.route('/download_csv')
def download_csv():
    # Convert DataFrame to CSV string
    csv_string = df_files.to_csv(index=False)

    # Create a file-like object from the CSV string
    #csv_file = io.StringIO(csv_string)
    csv_file = io.BytesIO(csv_string.encode('utf-8'))

    # Send the CSV file as a download
    return send_file(csv_file, mimetype='text/csv', download_name='data.csv')
if __name__ == "__main__":
    app.run(debug=True)