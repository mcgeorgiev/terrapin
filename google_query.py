from google.cloud import vision
from google.cloud.vision.feature import Feature
from google.cloud.vision.feature import FeatureTypes
import io
client = vision.Client()
with io.open("/home/michael/terrapin/objects.jpg", 'rb') as image_file:
    content = image_file.read()

image = client.image(content = content)
features = [Feature(FeatureTypes.LABEL_DETECTION, 1), Feature(FeatureTypes.FACE_DETECTION,1)]
annotations = image.detect(features)

things = []
for thing in annotations:
    for label in thing.labels:
        print label.description, label.score
