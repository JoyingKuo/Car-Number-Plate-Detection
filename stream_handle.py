import string
import random
import pyrebase
import subprocess
import os


config = {
  "apiKey": "AIzaSyAyu0tIQgWiU13sF6NBsdhFHy3FdU9oHLY",
  "authDomain": "car-number-plate.firebaseapp.com",
  "databaseURL": "https://car-number-plate.firebaseio.com",
  "storageBucket": "car-number-plate.appspot.com",
  "serviceAccount": "/home/joying/car-number-plate-firebase-adminsdk-x0gm2-7601fd9d3d.json"
}

firebase = pyrebase.initialize_app(config)
count=0
#authentication
auth = firebase.auth()
user = auth.sign_in_with_email_and_password("nctucscar108@gmail.com", "CAR108car108")
db = firebase.database()


storage = firebase.storage()

def id_generator(size=18, chars = string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
    
def stream_handler(message):
  global count 
  print(message["event"]) # put
  print(message["path"]) # /-K7yGTTEp7O549EzTYtI
  print(message["data"]) # {'title': 'Pyrebase', "body": "etc..."}
  if (count!=0) &(count%2!=0):
    get_image = "images/" + "%s"%(message["data"]) + ".jpg"
    #print get_image
    store_image = "%s"%(message["data"]) + ".jpg"
    storage.child(get_image).download(store_image)
    analyze_image = id_generator() + ".jpg"
    os.system("python analyze.py %s weights6.npz weights7.npz weights8.npz %s" % (store_image, analyze_image))
    print "pass"
    #os.rename(analyze_image,detect.code+".jpg")
    #print detect.code
    #put to firebase
    return_image= "result/" + analyze_image
    print return_image#reuturn_str=sys.argv[1][0:lenG-4]+"_return"
    storage.child(return_image).put(analyze_image)
    lenG=len(analyze_image)
    data = {"%s"%(analyze_image[0:lenG-4]): "%s"%(analyze_image[0:lenG-4])}
    #db.child("%s"%(reuturn_str)).set(data)
    db.push(data)
    
    
    
  
  count=count+1
  
  print"ok"
  
my_stream = db.stream(stream_handler)