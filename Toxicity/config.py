import os

BASE_DIR = 'data'
TEST_PATH = os.path.join(BASE_DIR, 'test.csv')
TRAIN_PATH = os.path.join(BASE_DIR, 'train.csv')

COLUMNS = ['id', 'target', 'comment_text', 'severe_toxicity', 'obscene', 'identity_attack', 
           'insult', 'threat', 'asian', 'atheist', 'bisexual', 'black', 'buddhist', 'christian', 
           'female', 'heterosexual', 'hindu', 'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability', 
           'jewish', 'latino', 'male', 'muslim', 'other_disability', 'other_gender', 'other_race_or_ethnicity', 
           'other_religion', 'other_sexual_orientation', 'physical_disability', 'psychiatric_or_mental_illness', 
           'transgender', 'white', 'created_date', 'publication_id', 'parent_id', 'article_id', 'rating', 'funny', 
           'wow', 'sad', 'likes', 'disagree', 'sexual_explicit', 'identity_annotator_count', 'toxicity_annotator_count']


BATCH_SIZE=32
