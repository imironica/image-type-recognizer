# Code source by Ionuț Mironică
# This script download a list of images from Google Images
from google_images_download import google_images_download

directories = ['clip-art', 'photo']
classes = 'aquatic, lion'

classes = 'mammals,beaver, dolphin, otter, seal, whale, fish, aquarium fish, flatfish, ray, shark, trout,' \
          'flowers, orchids, poppies, roses, sunflowers, tulips, food, containers, bottles, bowls, cans, cups, plates,' \
          'fruit, vegetables, apples, mushrooms, oranges, pears, sweet peppers, household electrical, devices, clock,' \
          'computer keyboard, lamp, telephone, television, household furniture, bed, chair, couch, table, wardrobe,' \
          'insects, bee, beetle, butterfly, caterpillar, cockroach, large, carnivores, bear, leopard, lion, ' \
          'tiger, wolf, large, man-made, outdoor things, bridge, castle, house, road, skyscraper, large, ' \
          'natural, outdoor scenes, cloud, forest,mountain, plain, sea, large, omnivores, herbivores, camel, ' \
          'cattle, chimpanzee, elephant, kangaroo, medium-sized mammals, fox, porcupine, possum, raccoon, ' \
          'skunk, non-insect invertebrates, crab, lobster, snail, spider, worm, people, baby,' \
          'boy, girl, man, woman, reptiles, crocodile, dinosaur, lizard, snake, turtle, small mammals, ' \
          'hamster, mouse, rabbit, shrew, squirrel, trees,maple, oak, palm, pine, willow, vehicles, ' \
          'bicycle, bus, motorcycle, pickup truck, train, lawn-mower, rocket, streetcar, tank, tractor'

for directory in directories:
    response = google_images_download.googleimagesdownload()
    arguments = {"keywords": classes, "limit": 99, "print_urls": True, "format": "jpg", 'type': directory,
                 "output_directory": directory}
    print(arguments)
    paths = response.download(arguments)
