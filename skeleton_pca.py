import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
import numpy as np

def plot_vector_as_image(image, h, w):
	"""
	utility function to plot a vector as image.
	Args:
	image - vector of pixels
	h, w - dimesnions of original pi
	"""
	plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
	plt.title('title', size=12)

def get_pictures_by_name(name='Ariel Sharon'):
	"""
	Given a name returns all the pictures of the person with this specific name.
	YOU CAN CHANGE THIS FUNCTION!
	THIS IS JUST AN EXAMPLE, FEEL FREE TO CHANGE IT!
	"""
	lfw_people = load_data()
	selected_images = []
	n_samples, h, w = lfw_people.images.shape
	print("Got: ", list(lfw_people.target_names))
	target_label = list(lfw_people.target_names).index(name)
	for image, target in zip(lfw_people.images, lfw_people.target):
		if (target == target_label):
			image_vector = image.reshape((h*w, ))
			selected_images.append(image_vector)
	return selected_images, h, w

def load_data():
	# Don't change the resize factor!!!
	lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
	return lfw_people

######################################################################################
"""
Other then the PCA function below the rest of the functions are yours to change.
"""

def PCA(X, k):
	"""
	Compute PCA on the given matrix.

	Args:
		X - Matrix of dimesions (n,d). Where n is the number of sample points and d is the dimension of each sample.
		For example, if we have 10 pictures and each picture is a vector of 100 pixels then the dimesion of the matrix would be (10,100).
		k - number of eigenvectors to return

	Returns:
	  U - Matrix with dimension (k,d). The matrix should be composed out of k eigenvectors corresponding to the largest k eigenvectors
	  		of the covariance matrix.
	  S - k largest eigenvalues of the covariance matrix. vector of dimension (k, 1)
	"""
	n = X.shape[0]
	normX = X - np.mean(X, axis=0)
	
	Sigma = (1/n) * np.matmul(normX.T, normX)

	vals, vecs = np.linalg.eig(Sigma)
	vecs = vecs.T

	indices = np.argpartition(vals, 1*k)[-1*k:]
	indices.sort()
	U = np.array(vecs[indices])
	S = np.array(vals[indices])

	return U,S
	

def main():
	images, h,w = get_pictures_by_name("Gerhard Schroeder")
	X = np.array([images[i] for i in range(len(images))])
	U,S = PCA(X, 10)


	figure, axis = plt.subplots(2, 5) 
	for i in range(2):
		for j in range(5):
			axis[i,j].imshow(U[i*5 + j].reshape((h, w)), cmap=plt.cm.gray)
			axis[i,j].set_title("PC No."+str(i*5 + j))
			axis[i,j].axis("off")

	plt.suptitle("Principal components")
	plt.show()

	
	
	


if __name__ == '__main__':
	main()