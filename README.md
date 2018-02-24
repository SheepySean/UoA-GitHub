# UoA-NLP - GitHub misunderstanding project

**Tools for improving source code understanding**

The GitHub Understanding project is a project to identify confusion experienced by software developers during application development to better tailor tools to help them.

The Project uses both natural language processing and cluster analysis to attempt to identify confusion. In current implementation is a C4.5 Decision Tree Classifier that attempts to predictively classify developer comments for confusion. There is also clustering algorithms to group confusing comments together. The suite of algorithms includes DBScan, k-means, and Affinity Propogation.

## Getting Started

These instructions will get you a running version of the project on your local machine.

### Prerequisites

The project should be able to run on all operating systems.

The project requires:
```
- MongoDB (see Preparing The Database for full install instructions)
- Python (2.7 or 3.6) with PiP installed.
(The following Python Packages must also be installed, run pip install ... for the following:)
-- NLTK, don't forget to run nltk.download() from the python shell and install all packages
-- pymongo
-- gensim for LDA/LSA
-- scikit-learn
-- pandas and redis for recommender
```

### Preparing the Database

The Misunderstanding project uses MongoDB as its database management system. This is due to its performance, non-relational structure, and the way in which it stores string data.

#### Installing MongoDB

To download MongoDB click [Here](https://www.mongodb.com/download-center?jmp=nav#community)

Follow the instructions from the installer and MongoDB should successfully install.

#### Configuring PATH (Windows)

MongoDB is run from the command line via Mongo Commands. Typically during the install, your **PATH** Environment Variable will not be configured correctly, therefore you have to enter the absolute path to MongoDB Commands to run them.

In to the start menu on windows type **environment** and click **Edit the system environment variables**. With Environment Variables open, click **Edit...** under **System Variables**. Click **New**, then paste the path to the MongoDB bin and click **OK**. The path to the bin should look something similar to:
*C:\Program Files\MongoDB\Server\3.4\bin\*.

**Your MongoDB should now be working, to test this type the command *mongod* to start the server in the Command Prompt. To run commands on the database, open a new Command Prompt and enter the command *mongo* to open the mongo client which then can be queried**

#### Installing MongoDB Compass (Recommended)
 
MongoDB provides a GUI to via MongoDB Data called MongoDB Compass.

To install MongoDB Compass click [Here](https://www.mongodb.com/download-center?jmp=nav#compass)

With Compass installed, you can open from the start menu as any appilcation and click connect to connect to your Server. 
*Note: The server must be running in a command prompt to connect to it. The server can be started with the command "mongod"*

#### Importing the Database

Now that MongoDB is installed, download the **data** folder that comes with this project and unzip the **uoa-gh** folder.

On the command line, navigate to the data folder such that you can see the uoa-gh folder but not in that folder and run the following command:

```shell
mongorestore -d uoa-gh uoa-gh
```

This will create the database and name it **uoa-gh**. You should now be able to see it in MongoDB Compass or from the Mongo Client on the command line.


## Running the Code

With Python (plus dependant packages) and the database installed you can run the project code. 

To do this, start the MongoDB Server via the command from the command line:
```shell
mongod
```

Then run the main script file **break_classify.py**. If any dependencies have been missed, install them with pip from the command line. 

You should see console output which consists of the classifier and clusterizer results. The clusters will then be saved to text files in the same directory as break_classify.py

## Folder Structure

The projects folder structure is as follows:

```
UoA-NLP-GitHub
|
|__data (backup of the database)
|  |__uoa-gh
|	  |__ ...
|  
|__Database (database related .py files)
|  |__Configuration.py (connection configuration to the database)
|
|__ML (machine learning related .py files including clustering and classifying scripts)
|  |__affinity_propagation_clusterizer.py
|  |__classifier.py
|  |__classifier_data.py
|  |__classifier_data.py
|  |__dbscan_clusterizer.py
|  |__decision_tree_classifier.py
|  |__hierarchical_clusterizer.py
|  |__kmeans_clusterizer.py
|  |__recommender.py
|  |__sgd_classifier.py
|  |__silhouette_validator.py
|
|__NLP (natural language processing related .py files)
|  |__input_preprocessor.py
|  |__keyword_extractor.py
|  |__latent_dirichlet_allocation.py
|  |__latent_semantic_analysor.py
|  |__summarizer.py
|
|__Utils (utility related .py files)
|  |__generate_config.py
|  |__logger.py
|
|__break_classify.py (Main .py script for the project)
|
|__categories.txt (data categories for classification, including the corresponding category ID)
|
|__keywords_lda.py
|
|__sandbox.py
|
|__README.md (You Are Here)
```

## Helpful Links

*  [Python](https://www.python.org/) - Python Dowload with choice between versions 2.7 and 3.6
* [MongoDB Manual](https://docs.mongodb.com/manual/) - Manual to using MongoDB
* [PyCharm](https://www.jetbrains.com/pycharm/download/) - PyCharm IDE for Python Devlopement (Recommended)
* [NLTK](http://www.nltk.org/book/ch06.html) - A guide to using NLTK to classify text
* [Topic Modelling](https://www.youtube.com/watch?v=BuMu-bdoVrU) - A video guide to Topic Modelling with Python

## Authors

* **Sean Oldfield** - *Development* - [SheepySean](https://github.com/SheepySean)
* **Austin Sutherland** - *Development* - [a-suth](https://github.com/a-suth)
* **Cyril Bos** - *Development* - [CyrilBos](https://github.com/CyrilBos)


