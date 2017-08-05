import cv2
import numpy as np
import os
import requests
import shutil

def store_raw_images():
    negative_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03319745'
    r = requests.get(negative_link)
    negative_urls = r.text.encode("utf-8")

    if not os.path.exists('neg'):
        os.makedirs('neg')

    num = 1
    split_links = negative_urls.split('\n')
    for url in split_links:
        try:
            print str(num), url
            response = requests.get(url, stream=True)

            complete_path = os.path.join('/home/michael/terrapin/haar_cascade/neg', str(num)+".jpg")
            with open(complete_path, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            del response

            img = cv2.imread(complete_path, cv2.IMREAD_GRAYSCALE)
            resized_img = cv2.resize(img, (100,100))
            cv2.imwrite("neg/"+str(num)+".jpg", resized_img)

        except Exception as e:
            print e
        num += 1


def check_if_white(current_path):
    img = cv2.imread(current_path)
    non_whites = 0
    for i in range(100):
        for j in range(100):
            if img[i][j][0] != 255 or img[i][j][1] != 255 or img[i][j][2] != 255:
                non_whites +=1
    return True if non_whites < 400 else False


def remove_images():
    for item in os.listdir("neg"):
        current_path = "neg/" + str(item)
        print current_path
        try:
            img = cv2.imread(current_path)
            if img.shape[0] != 100 or img.shape[1] != 100:
                print "not correct size"
                os.remove(current_path)
            elif check_if_white(current_path):
                print "item is flickr placeholder"
                os.remove(current_path)
            else:
                print "image okay"
        except:
            print "cannot load image"
            os.remove(current_path)

        print



# store_raw_images()
remove_images()
# print check_if_white()
