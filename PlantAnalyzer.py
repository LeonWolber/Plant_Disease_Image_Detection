import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class PlantAnalyzer:
	def __init__(self, train_dir):


		self.train_dir = train_dir
		self.plant_data = pd.read_csv(train_dir)
		self.sampled_data = pd.DataFrame()
		le = LabelEncoder()

		# pick first label if multiple are present per image
		self.plant_data['single_label'] = [i.split(' ')[0] for i in self.plant_data['labels']]

		# create new column with LabelEncoder
		self.plant_data['encoded_labels'] = le.fit_transform(self.plant_data.single_label)
		self.n_classes = self.plant_data['encoded_labels'].nunique()

		self.balance_classes()

		self.train_test()




	def balance_classes(self):
		# find out # obs. of least represented class
		min_represented = self.plant_data['encoded_labels'].value_counts().min() - 1

		small = []
		# split df into small df's per label and prune them to have uniform size
		for label in range(0,5):
			l.append(self.plant_data[self.plant_data['encoded_labels'] == label].sample(min_represented))

		# create full dataframe
		self.sampled_data = pd.concat([small[0], small[1], small[2], small[3], small[4]]).reset_index(drop=True)


	def train_test(self):
		# split data X = image_names (DataGen will load images according to the name), y = label
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.sampled_data['image'], self.sampled_data['encoded_labels'], test_size=0.33, random_state=42)

		# reset index to make use of DataGenerator later on
		[i.reset_index(drop=True, inplace=True) for i in [self.x_train, self.x_test, self.y_train, self.y_test]]
		

		