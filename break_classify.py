import numpy as np
import re

from nltk.tokenize import sent_tokenize
from pymongo import MongoClient, errors

from pymongo import MongoClient

from Database.Configuration import host, port


from ML.affinity_propagation_clusterizer import AffinityPropagationClusterizer
from ML.dbscan_clusterizer import DBSCANClusterizer
from ML.decision_tree_classifier import C45DecisionTreeClassifier
from ML.hierarchical_clusterizer import HierarchicalClusterizer
from ML.sgd_classifier import SGDClassifier
from ML.kmeans_clusterizer import KMeansClusterizer

''' 
Script that utilises document clustering and decision tree classification to identify confusion based on comments
retrieved from developers on GitHub.

Authors: Austin Sutherland (https://github.com/a-suth), Cyril Bos (https://github.com/CyrilBos), and Sean Oldfield (https://github.com/SheepySean)
Date: 10/02/2018
'''

PREPROCESS = True # True if the data should be preprocessed (stemmed, bag of words, etc.)
N_SPLITS_KFOLD_CROSS_VALIDATION = 20 #Number of chunks to split the data into to cross validate
MAX_NUMBER_DATA = 5000 # Max Range of Data to be retrieved from the database
MIN_NUMBER_DATA = 0 # Min Range of Data to be retrieved from the database
NUM_DISPLAY_PER_CATEGORY = 20 # How many sentences should be displayed per category from the classifier
COLLECTION = "issue_comments" # For Data extraction i.e generating training data

# Clustering params, tweak as necessary
N_FEATURES = 20
JOBS = 3
VERBOSE = 1

def connect_to_db(db_host, db_port, timeout=10):
    """
    Connect to a specified MongoDB
    :param db_host: hostname of the database location
    :type db_host: String
    :param db_port: port to connect the database under a TCP Connection
    :type db_port: int
    :param timeout: millisecond delay to abort the connection the database if it exceeds
    :type timeout: int
    :return: an instance of a MongoDB Client
    :rtype: MongoClient
    """

    # Connect to the Database
    try:
        db_client = MongoClient(db_host, db_port, serverSelectionTimeoutMS=timeout)
        db_client.server_info()
    except errors.ServerSelectionTimeoutError as err:  # Failed connection
        db_client = None
        print("Error: Unable to connect to the specified database")
        print("PyMongo's Error: {0}".format(err))

    return db_client

def get_project_data(db, attribute, project="", collection=COLLECTION):
    """
    This function is not currently in use in this script but is useful if you need to retrieve all data related to a certain
    project/repo on github
    :param db: database storing the comments
    :type db: database
    :param attribute: Attribute of the dictionary that contains the comment (normally "body")
    :type attribute: String
    :param project: Name of the project to retrieve data for (if left blank then all data is retrieved from db)
    :type project: String
    :param collection: Collection to inspect for comments, e.g. "issue_comments"
    :type collection: String
    :return: list containing all comments related to that project
    :rtype: list
    """
    print("Searching for projects matching '{}'...".format(project))
    to_search_regx = re.compile("/.*" + project + ".*/", re.IGNORECASE)

    project_data = []
    i = 0
    for document in db[collection].find({"html_url": to_search_regx}):
        project_data.append(document[attribute])
        i += 1

    print("Found {0} records matching the project '{1}'...".format(i, project))
    return project_data



##########################################################################################################


### Compute and print clusters ###

#question_recommender = Recommender()

def kmeans(data, category):
    """
    Run K-Means Clustering on supplied data and classify into a category
    :param data: Data to cluster
    :type data: list
    :param category: Category the current data is matched too (for labelled clustering)
    :type category: String
    :return:
    """
    divide_by = int(MAX_NUMBER_DATA / 100)
    n_clusters = int(len(data) / divide_by )
    print(n_clusters)
    clusterizer = KMeansClusterizer(data, n_features=N_FEATURES, preprocess=PREPROCESS, jobs=JOBS,
                                    verbose=VERBOSE)
    clusterizer.lda_clusterize(n_clusters=n_clusters, n_features=N_FEATURES, kmeans_iter=3)
    clusterizer.print_clusters(filename='kmeans_{}_{}.txt'.format(category, n_clusters))
    #clusterizer.get_avg_silhouette()
    return clusterizer

def dbscan(data, category):
    """
    Run DBScan Clustering on supplied data and classify into a category
    :param data: Data to cluster
    :type data: list
    :param category: Category the current data is matched too (for labelled clustering)
    :type category: String
    :return:
    """
    clusterizer = DBSCANClusterizer(data, n_features=N_FEATURES, preprocess=PREPROCESS, jobs=JOBS, verbose=VERBOSE)
    db = clusterizer.compute(eps=0.5, min_samples=5)

    #sihouette = clusterizer.get_avg_silhouette();

    clusterizer.print_clusters(filename='dbscan_{}_{}.txt'.format(category, len(clusterizer.get_clusters())))

    return clusterizer

def affinity(data, category):
    """
    Run Affinity Propagation Clustering on supplied data and classify into a category
    :param data: Data to cluster
    :type data: list
    :param category: Category the current data is matched too (for labelled clustering)
    :type category: String
    :return:
    """
    clusterizer = AffinityPropagationClusterizer(data)

    clusterizer.compute(n_features=10, max_iter=1)

    clusterizer.print_clusters(filename='affinity_{}_{}.txt'.format(category, len(clusterizer.get_clusters())))

    return clusterizer

def main():
    """
    Main routine of script, could be split and refactored as necessary
    :return:
    """

    # Attempt to connect to the database and end the script if unable to
    client = connect_to_db(host, port)
    if client is None:
        return
    db = client["uoa-gh"]

    training_data = db.training_data.find({}) # Get training data from the database

    # Prep the data for classifier training
    data = []
    target = []
    target_names = []

    for item in training_data:
        data.append(item["content"])
        target.append(item["category_id"])
        if item["category"] not in target_names:
            target_names.append(item["category"])

    # Generate and train the classifier
    classifier = C45DecisionTreeClassifier(data, target, target_names, preprocess=PREPROCESS)  # SGDClassifier(data, target, target_names)
    classifier.train()
    print('Precision of the classifier on its training data set: ', classifier.evaluate_precision(1))
    print("{}-Fold precision evaluation on the training data set: \n".format(N_SPLITS_KFOLD_CROSS_VALIDATION),
          classifier.evaluate_precision(N_SPLITS_KFOLD_CROSS_VALIDATION))

    ### predict the category of every question, appending it into the corresponding list of the dictionary ###

    # Prep some empty dictionaries for clustering later
    predicted_categories = {}
    cluster_data = {}
    cluster_target = {}
    for category_name in target_names:
        predicted_categories[category_name] = []
        cluster_data[category_name] = []
        cluster_target[category_name] = []

    # Get some developer comments (this could be changed to issue or commit comments)
    pull_request_comments = db.pull_request_comments.find({})[MIN_NUMBER_DATA:MAX_NUMBER_DATA]


    for comment in pull_request_comments: # For each comment
        # comment = comment.replace('.', '. ').replace('.  ', '. ').replace('?', '? ').replace('?  ', '? ').replace('!', '! ').replace('!  ', '! ')
        for sentence in sent_tokenize(comment["body"]): # For each sentence in the comment
            predicted_category_i = classifier.predict([sentence])[0] # Predict the sentence and get the predicted index
            # Add the prediction to the list of same categorised sentences
            predicted_categories[classifier.target_names[predicted_category_i]].append(sentence)

    # Display the results of the classifier
    for category in predicted_categories:
        print()
        print("-" * 100)
        print('#### Category: ' + category.upper() + " #####")
        nb = NUM_DISPLAY_PER_CATEGORY if len(predicted_categories[category]) >= NUM_DISPLAY_PER_CATEGORY else len(predicted_categories[category])
        for i in range(nb):
            print(predicted_categories[category][i])

    cluster_target_names = []
    # For each category, attempt to cluster its data
    for category in predicted_categories:
        if len(predicted_categories[category]) > 0 and category != 'outroduction':
            cluster_target_names.append(category)
            for sentence in predicted_categories[category]:
                cluster_data[category].append(sentence)
                cluster_target[category].append(category)

            # Algorithm choice, potentially make this either console or command line input
            clusterizer =  kmeans(cluster_data[category], category)
            #clusterizer = dbscan(cluster_data[category], category)
            # affinity(cluster_data[category], category)
            # clusterizer.print_to_csv()
            """
            #"Clustering" the documents using a recommender
            recommend_data = {'id':[], 'description':[]}

            for i in range(len(cluster_data[category])):
                recommend_data['id'].append(i)

                recommend_data['description'].append(cluster_data[category][i])

            question_recommender.train(recommend_data)

            save_file = open('broken_idf_recommendations_{}.txt'.format(category), 'w')
            rnd = random.randint(0, 20)
            for item_to_recommend_index in range(rnd, rnd + rnd):
                print("Current item: ", recommend_data['description'][item_to_recommend_index])
                save_file.write("Current item: " + recommend_data['description'][item_to_recommend_index])
                recommended_items = question_recommender.predict(recommend_data['id'][item_to_recommend_index], 5)
                print("Recommended items: ")
                for recommended_item in recommended_items:
                    print(recommend_data['description'][recommend_data['id'].index(int(recommended_item[1]))])
                    save_file.write(recommend_data['description'][recommend_data['id'].index(int(recommended_item[1]))])

            category_i += 1
            """

if __name__ == '__main__':
    main()
