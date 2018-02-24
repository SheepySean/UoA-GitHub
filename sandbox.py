from pymongo import MongoClient

from Database.Configuration import host, port

client = MongoClient(host, port)

db = client['uoa-gh']

pull_request_comments = db.pull_request_comments.find({})

threads = {}
for comment in pull_request_comments:
    pullreq_id = comment['pullreq_id']
    if pullreq_id not in threads:
        threads[pullreq_id] = []
    threads[pullreq_id].append(comment)

file = open('pull_request_comments_threads.txt', 'w')

print(threads.keys())
for thread in threads:
    file.write('##### thread {} #####\n'.format(thread))
    print(thread)
    threads[thread] = sorted(threads[thread], key=lambda k: 0 if k['position'] is None else k['position'])
    for comment in threads[thread]:
        file.write('# Comment {} pos {}\n'.format(comment['_id'], comment['position']) + comment['body'] + '\n')
