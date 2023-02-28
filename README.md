# flask-deployment
Flask is a web application framework written in Python. Building/Training a model using various algorithms on a large dataset is one part of the data. But using these models within the different applications is the second part of deploying machine learning in the real world.

Construction:
1.	Import necessary libraries using requirments.txt
2.	Create ML Model and save (pickle) it
3.	Create Flask files for UI and python main file (app.py) that can unpickle the machine learning model from step 1 and do predictions
4.	Create requirements.txt to setup Flask web app with all python dependencies
5.	Create Procfile to initiate Flask app command
6.	Commit files from Step 1, 2, 3 & 4 in the GitHub repo
7.	Create account/Login on Heroku, create an app, connect with GitHub repo, and select branch
8.	Select manual deploy (or enable Automatic deploys) on Heroku

