import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

# Path to images
img_dir = 'image_data\\abstract\\'
img_paths = glob.glob(img_dir+'*')

# Create a new dataframe for the ratings or read it from a file if it already exists
ratings_path = 'art_ratings.csv'
ratings_df = pd.DataFrame(columns=['image', 'rating']) if not os.path.isfile(
    ratings_path) else pd.read_csv(ratings_path)
ratings_df['image'] = ratings_df['image'].astype(str)

print('Enter rating for the images on a scale from 0 to 5.')
for i in img_paths:
    # Skip the image if it has already been given a rating
    img_name = i.replace(img_dir, '')
    if ratings_df['image'].str.contains(img_name).any():
        continue

    # Show the image
    img = mpimg.imread(i)
    plt.imshow(img)
    plt.show(block=False)

    # Ask for the rating
    try:
        rating = int(input('Rating: '))
    except ValueError:
        print('The rating must be an integer from 0 to 5. Exiting!')
        break
    if ((rating < 0) or (rating > 5)):
        print('Value outside of the rating scale. Exiting!')
        break
    plt.close()

    # Append the rating to the dataframe
    new_entry = pd.DataFrame([[img_name, rating]], columns=['image', 'rating'])
    ratings_df = pd.concat([ratings_df, new_entry], ignore_index=True)

ratings_df.to_csv(ratings_path, header=True, index=False)
